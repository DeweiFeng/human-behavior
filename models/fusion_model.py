# models.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoFeatureExtractor


class MultiTaskMentalHealthModel(nn.Module):
    def __init__(self, text_model, audio_model, hidden_dim=768, sampling_rate=16000):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained(text_model)
        self.audio_encoder = AutoModel.from_pretrained(audio_model)
        self.audio_tokenizer = AutoFeatureExtractor.from_pretrained(audio_model)

        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)
        self.audio_proj = nn.Linear(self.audio_encoder.config.hidden_size, hidden_dim)
        self.face_proj = nn.Sequential(nn.Linear(24, hidden_dim), nn.ReLU())
        self.gesture_proj = nn.Sequential(nn.Linear(10, hidden_dim), nn.ReLU())

        self.fusion_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4), num_layers=2
        )

        self.task_heads = nn.ModuleDict(
            {
                "phq8_nointerest": nn.Linear(hidden_dim, 4),
                "phq8_depressed": nn.Linear(hidden_dim, 4),
                "phq8_sleep": nn.Linear(hidden_dim, 4),
                "phq8_tired": nn.Linear(hidden_dim, 4),
                "phq8_appetite": nn.Linear(hidden_dim, 4),
                "phq8_failure": nn.Linear(hidden_dim, 4),
                "phq8_concentrating": nn.Linear(hidden_dim, 4),
                "phq8_moving": nn.Linear(hidden_dim, 4),
            }
        )
        self.task_heads["emotion_meld"] = nn.Linear(hidden_dim, 7)

        self.sampling_rate = sampling_rate
        self.chunk_seconds = 10
        self.stride_seconds = 1

    def forward(self, text_inputs, audio_waveform, face_input, gesture_input, task):
        # === TEXT ===
        text_emb = self.text_encoder(**text_inputs).last_hidden_state[:, 0]

        # === AUDIO: split waveform into chunks, encode, pool ===
        audio_chunks = self._split_audio(audio_waveform)

        cls_sum = None
        count = 0

        for chunk in audio_chunks:
            inputs = self.audio_tokenizer(
                chunk,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True,
            ).to(text_emb.device)

            out = self.audio_encoder(**inputs)
            cls = out.last_hidden_state[
                :, 0, :
            ].detach()  # 🚨 critical to prevent memory accumulation

            if cls_sum is None:
                cls_sum = cls
            else:
                cls_sum += cls
            count += 1

        audio_emb = (
            (cls_sum / count)
            if count > 0
            else torch.zeros((1, text_emb.size(1)), device=text_emb.device)
        )

        text_emb = self.text_proj(text_emb)
        audio_emb = self.audio_proj(audio_emb)

        # === FACE / GESTURE ===
        # Face input: [B, T, F] → Projected: [B, T, D]
        face_proj_seq = self.face_proj(face_input)
        # Pool across time dimension (e.g., mean)
        face_emb = face_proj_seq.mean(dim=1)  # [B, D]

        gesture_proj_seq = self.gesture_proj(gesture_input)
        gesture_emb = gesture_proj_seq.mean(dim=1)  # [B, D]

        # === FUSION ===
        fused = torch.stack([text_emb, audio_emb, face_emb, gesture_emb], dim=1)
        fused = self.fusion_encoder(fused)
        fused_pooled = fused.mean(dim=1)

        return self.task_heads[task](fused_pooled)

    def _split_audio(self, waveform):
        sr = self.sampling_rate
        chunk_size = self.chunk_seconds * sr
        stride_size = self.stride_seconds * sr

        chunks = []
        for start in range(0, len(waveform), chunk_size - stride_size):
            chunk = waveform[start : start + chunk_size]
            if len(chunk) >= sr:
                chunks.append(chunk)
        return chunks

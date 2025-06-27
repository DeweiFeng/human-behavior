import torch
import torch.nn as nn
from transformers import AutoModel

class MultiTaskMentalHealthModel(nn.Module):
    def __init__(self, text_model="meta-llama/Llama-3.1-8b", audio_model="microsoft/wavlm-base-plus"):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model)
        self.audio_encoder = AutoModel.from_pretrained(audio_model)

        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, 256)
        self.audio_proj = nn.Linear(self.audio_encoder.config.hidden_size, 256)

        self.face_proj = nn.Linear(35, 256)
        self.pose_proj = nn.Linear(6, 256)

        self.fusion_proj = nn.Sequential(
            nn.Linear(256 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Task-specific heads
        self.phq8_heads = nn.ModuleList([nn.Linear(512, 4) for _ in range(8)])  # PHQ-8 item classifiers
        self.phq_binary_head = nn.Linear(512, 2)  # Binary depression classifier
        self.emotion_head = nn.Linear(512, 7)     # Emotion classifier (MELD, IEMOCAP)
        # Add more heads as needed...

    def forward(self, text_input, audio_input, face_input, pose_input, task="phq8"):
        # Encodings
        text_feat = self.text_encoder(**text_input).last_hidden_state[:, 0, :]
        audio_feat = self.audio_encoder(**audio_input).last_hidden_state.mean(dim=1)

        text_feat = self.text_proj(text_feat)
        audio_feat = self.audio_proj(audio_feat)

        face_feat = self.face_proj(face_input)
        pose_feat = self.pose_proj(pose_input)

        # Fusion
        x = torch.cat([text_feat, audio_feat, face_feat, pose_feat], dim=1)
        fused = self.fusion_proj(x)

        # Task routing
        if task == "phq8":
            return [head(fused) for head in self.phq8_heads]  # List of logits for each item
        elif task == "phq_binary":
            return self.phq_binary_head(fused)
        elif task == "emotion":
            return self.emotion_head(fused)
        else:
            raise ValueError(f"Unknown task: {task}")

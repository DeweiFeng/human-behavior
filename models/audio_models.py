from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torchaudio

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

waveform, sr = torchaudio.load(audio_path)  # (1, T)
inputs = processor(waveform, sampling_rate=sr, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
audio_embedding = outputs.last_hidden_state.mean(dim=1)

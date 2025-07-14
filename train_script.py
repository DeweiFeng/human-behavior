# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoFeatureExtractor
from models import MultiTaskMentalHealthModel
from dataloaders.mosei_loader import MOSEIDataset
from tqdm import tqdm
from utils.collate_utils import unified_collate_fn as custom_collate
import numpy as np
import os

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT_MODEL = "meta-llama/Llama-3.2-1B"
AUDIO_MODEL = "microsoft/wavlm-base-plus"
BATCH_SIZE = 1
EPOCHS = 5
SAVE_PATH = "checkpoints/multitask_model.pt"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)


# def custom_collate(batch):
#     return batch  # returns a list of samples


# === Tokenizers ===
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
text_tokenizer.pad_token = text_tokenizer.eos_token

audio_tokenizer = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)

# === Init Dataset + Loader ===
dataset = MOSEIDataset(
    data_dir="../orcd/pool/MOSEI",  # TODO: Update to your MOSEI data path
    split="train",
    modalities=["vision", "audio", "text"]
)
dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate
)

# === Validation Dataset + Loader ===
val_dataset = MOSEIDataset(
    data_dir="../orcd/pool/MOSEI",  # TODO: Update to your MOSEI data path
    split="valid",
    modalities=["vision", "audio", "text"]
)
val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate
)


@torch.no_grad()
def evaluate(model, dataloader, tokenizer, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        for sample in batch:
            # Prepare inputs
            if isinstance(sample["text"], list):
                text_body = " ".join(str(t) for t in sample["text"])
            else:
                text_body = sample["text"]

            prompt = sample["task"].get("prompt", "")
            full_text = f"{prompt}\n\n{text_body}" if prompt else text_body

            text_inputs = tokenizer(
                full_text, return_tensors="pt", padding=True, truncation=True
            ).to(DEVICE)
            audio_inputs = sample["audio"]

            audio_inputs = audio_inputs / np.abs(audio_inputs).max()

            face = sample.get("face")
            gesture = sample.get("gesture")
            face_input = (
                torch.tensor(
                    face if face is not None else np.zeros((35,)), dtype=torch.float32
                )
                .unsqueeze(0)
                .to(DEVICE)
            )
            pose_input = (
                torch.tensor(
                    gesture if gesture is not None else np.zeros((6,)),
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .to(DEVICE)
            )

            label = torch.tensor([sample["label"]], dtype=torch.long).to(DEVICE)
            task_name = sample["task"].get("name", sample["task"]["type"])

            logits = model(
                text_inputs, audio_inputs, face_input, pose_input, task=task_name
            )
            loss = criterion(logits, label)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == label).sum().item()
            total += 1

            del text_inputs, audio_inputs, face_input, pose_input, logits, loss
            torch.cuda.empty_cache()

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


# === Model, Optimizer, Loss ===
model = MultiTaskMentalHealthModel(text_model=TEXT_MODEL, audio_model=AUDIO_MODEL).to(
    DEVICE
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()


# === Training Loop ===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        batch_loss = 0.0

        for sample in batch:
            # === Extract Text ===
            if isinstance(sample["text"], list):
                text_body = " ".join(str(t) for t in sample["text"])
            else:
                text_body = sample["text"]

            # If prompt exists, prepend it
            prompt = sample["task"].get("prompt", "")
            full_text = f"{prompt}\n\n{text_body}" if prompt else text_body

            text_inputs = text_tokenizer(
                full_text, return_tensors="pt", padding=True, truncation=True
            ).to(DEVICE)
            audio_inputs = np.nan_to_num(sample["audio"], nan=0.0, posinf=0.0, neginf=0.0)

            audio_inputs = audio_inputs / np.abs(audio_inputs).max()

            # === Face / Gesture ===
            face = sample.get("face")
            gesture = sample.get("gesture")
            face = np.nan_to_num(face, nan=0.0, posinf=0.0, neginf=0.0)
            gesture = np.nan_to_num(gesture, nan=0.0, posinf=0.0, neginf=0.0)

            face_input = (
                torch.tensor(
                    face if face is not None else np.zeros((35,)), dtype=torch.float32
                )
                .unsqueeze(0)
                .to(DEVICE)
            )
            pose_input = (
                torch.tensor(
                    gesture if gesture is not None else np.zeros((6,)),
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .to(DEVICE)
            )

            assert not torch.isnan(face_input).any(), "NaN in face_input"
            assert not torch.isinf(face_input).any(), "Inf in face_input"
            assert not torch.isnan(pose_input).any(), "NaN in pose_input"
            
            # === Label ===
            label = torch.tensor([sample["label"]], dtype=torch.long).to(DEVICE)

            assert label.item() >= 0, f"Invalid label: {label.item()}"

            # === Task Routing ===
            task_type = sample["task"]["type"]
            task_name = sample["task"].get(
                "name", task_type
            )  # e.g., "phq8_sleep", or "emotion_meld"

            logits = model(
                text_inputs, audio_inputs, face_input, pose_input, task=task_name
            )

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("⚠️ NaN or Inf in logits!")
                continue

            # === Loss and Backprop ===
            loss = criterion(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            batch_loss += loss.item()

            del text_inputs, audio_inputs, face_input, pose_input, logits, loss
            torch.cuda.empty_cache()

        optimizer.step()
        total_loss += batch_loss

    avg_loss = total_loss / len(dataloader)
    val_loss, val_acc = evaluate(model, val_dataloader, text_tokenizer, criterion)
    print(
        f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}"
    )
torch.save(model.state_dict(), SAVE_PATH)

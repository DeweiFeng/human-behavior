# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoFeatureExtractor
from models import MultiTaskMentalHealthModel
from datasets import DAICWOZDataset
from tqdm import tqdm
import os

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT_MODEL = "meta-llama/Llama-3.1-8b"
AUDIO_MODEL = "microsoft/wavlm-base-plus"
BATCH_SIZE = 4
EPOCHS = 5
SAVE_PATH = "checkpoints/multitask_model.pt"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# === Tokenizers ===
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
audio_tokenizer = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)

# === Init Dataset + Loader ===
dataset = DAICWOZDataset(root_dir="/home/dewei/workspace/dewei/dataset/daicwoz", split_csv="/home/dewei/workspace/dewei/dataset/daicwoz/train_split_Depression_AVEC2017.csv")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Model, Optimizer, Loss ===
model = MultiTaskMentalHealthModel(text_model=TEXT_MODEL, audio_model=AUDIO_MODEL).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# === Training Loop ===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        batch_loss = 0.0

        for sample in batch:
            try:
                # Preprocess inputs
                text = " ".join(sample["text"]) if isinstance(sample["text"], list) else sample["text"]
                text_inputs = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                audio_inputs = audio_tokenizer(sample["audio"], return_tensors="pt", padding=True, truncation=True).to(DEVICE)

                face_input = torch.tensor(sample["face"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                pose_input = torch.tensor(sample["gesture"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                label = torch.tensor([sample["label"]], dtype=torch.long).to(DEVICE)

                # Task routing
                task_type = sample["task"]["type"]
                if task_type == "qa_generation":
                    task_type = "phq8"
                    task_idx = list(DAICWOZDataset.PHQ8_LABEL_COLUMNS.values()).index(sample["id"].split("_", 1)[-1])
                    logits = model(text_inputs, audio_inputs, face_input, pose_input, task="phq8")[task_idx]
                else:
                    logits = model(text_inputs, audio_inputs, face_input, pose_input, task=task_type)

                # Compute loss and backprop
                loss = criterion(logits, label)
                loss.backward()
                batch_loss += loss.item()

            except Exception as e:
                print(f"Skipping sample due to error: {e}")
                continue

        optimizer.step()
        total_loss += batch_loss

    avg_loss = total_loss / len(dataloader)
    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")
torch.save(model.state_dict(), SAVE_PATH)
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoFeatureExtractor
from datasets import DAICWOZDataset
from models import MultiTaskMentalHealthModel
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import pearsonr


# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT_MODEL = "meta-llama/Llama-3.2-1B"
AUDIO_MODEL = "microsoft/wavlm-base-plus"
BATCH_SIZE = 1
CHECKPOINT_PATH = "checkpoints/multitask_model.pt"
TEST_CSV = "/home/dewei/workspace/dewei/dataset/daicwoz/full_test_split.csv"
ROOT_DIR = "/home/dewei/workspace/dewei/dataset/daicwoz"

# === Load Tokenizers ===
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
text_tokenizer.pad_token = text_tokenizer.eos_token
audio_tokenizer = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)

# === Dataset + Loader ===
test_dataset = DAICWOZDataset(
    root_dir=ROOT_DIR,
    split_csv=TEST_CSV,
    train=False,
    modalities=("transcript", "covarep", "clnf_aus", "clnf_pose"),
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

for batch in test_loader:
    print(batch)
    

# === Load Model ===
model = MultiTaskMentalHealthModel(text_model=TEXT_MODEL, audio_model=AUDIO_MODEL)
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.to(DEVICE)
model.eval()

# === Evaluation ===
# Store predictions and labels per subject
subject_preds = defaultdict(lambda: {})
subject_labels = defaultdict(lambda: {})

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        for sample in batch:
            sid = sample["id"].split("_")[0]
            task_name = sample["task"].get("name", sample["task"]["type"])

            # === Prepare Inputs ===
            if isinstance(sample["text"], list):
                text_body = " ".join(str(t) for t in sample["text"])
            else:
                text_body = sample["text"]
            prompt = sample["task"].get("prompt", "")
            full_text = f"{prompt}\n\n{text_body}" if prompt else text_body

            text_inputs = text_tokenizer(full_text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            audio_inputs = sample["audio"]
            face_input = torch.tensor(sample.get("face", np.zeros((35,))), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pose_input = torch.tensor(sample.get("gesture", np.zeros((6,))), dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # === Model Prediction ===
            logits = model(text_inputs, audio_inputs, face_input, pose_input, task=task_name)
            pred = torch.argmax(logits, dim=-1).cpu().item()  # predicted class 0–3

            subject_preds[sid][task_name] = pred
            subject_labels[sid] = sample["metadata"]["phq8_score"]

# === Final Aggregated Results ===
total_preds = []
total_labels = []

for sid in subject_preds:
    if set(subject_preds[sid].keys()) == set(subject_labels[sid].keys()) == set(DAICWOZDataset.PHQ8_LABEL_COLUMNS.keys()):
        score_pred = sum(subject_preds[sid].values())
        score_label = subject_labels[sid]
        total_preds.append(score_pred)
        total_labels.append(score_label)

# === Output and Evaluation ===
for sid, p, l in zip(subject_preds.keys(), total_preds, total_labels):
    print(f"Subject {sid}: Predicted PHQ-8 = {p}, Ground Truth = {l}")

if total_preds:
    y_pred = np.array(total_preds)
    y_true = np.array(total_labels)
    corr, _ = pearsonr(y_true, y_pred)
    print(f"\n[Evaluation] PHQ-8 Pearson r: {corr:.4f}")
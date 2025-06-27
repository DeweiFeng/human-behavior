from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Suppose `text` is a single string: full transcript or concatenated utterances
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
text_embedding = outputs.last_hidden_state.mean(dim=1)  # (B, H)

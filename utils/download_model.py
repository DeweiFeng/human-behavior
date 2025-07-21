from transformers import AutoModel, AutoTokenizer, AutoConfig
import argparse
from pathlib import Path

def download_model(model_name_or_path: str, output_dir: str):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model: {model_name_or_path}")
    print(f"Saving to: {output_dir}")

    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=output_dir)

    # Download config
    print("Downloading config...")
    config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=output_dir)

    # Download model weights
    print("Downloading model...")
    model = AutoModel.from_pretrained(model_name_or_path, cache_dir=output_dir)

    print("\n✅ Done! Model downloaded to:")
    print(f"{output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path (e.g., Qwen/Qwen1.5-7B)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./downloaded_model",
        help="Directory to save the downloaded model"
    )

    args = parser.parse_args()
    download_model(args.model, args.output_dir)

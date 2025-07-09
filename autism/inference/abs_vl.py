from abc import ABC, abstractmethod
import json
import os

class VisionLanguageModel(ABC):
    @abstractmethod
    def fetch_inputs(self, *args, **kwargs):
        pass

    @abstractmethod
    def process_inputs(self, *args, **kwargs):
        pass

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

    def save_outputs(self, output_text, save_path, meta_data=None):
        """
        Shared implementation: Append outputs + meta-data to JSON (or create if not exists)
        """
        meta_data = meta_data or {}
        save_entry = {"outputs": output_text, "meta": meta_data}

        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        print(f"Warning: Existing JSON is not a list — overwriting.")
                        existing_data = []
                except json.JSONDecodeError:
                    print(f"Warning: JSON decode error — overwriting.")
                    existing_data = []
        else:
            existing_data = []

        existing_data.append(save_entry)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2)

        print(f"Appended output to {save_path} (total entries: {len(existing_data)})")

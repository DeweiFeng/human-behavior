import os
import json

def normalize_outputs_in_json(directory: str):
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                modified = False
                # Ensure we're dealing with a list of dicts
                if isinstance(data, list):
                    for item in data:
                        outputs = item.get("outputs", [])
                        if isinstance(outputs, list):
                            for i, val in enumerate(outputs):
                                if isinstance(val, str) and val.lower() == "o":
                                    outputs[i] = "0"
                                    modified = True

                if modified:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)
                    print(f"Updated outputs in {filename}")

            except Exception as e:
                print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    JSON_DIR = "/home/human-behavior/autism/data/ados_outputs"
    normalize_outputs_in_json(JSON_DIR)
    print("Normalization complete.")
import json

def clean_newlines_in_json(input_path, output_path):
    # Load the original JSON file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Recursively replace '\n' with ' ' in all string values
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean(i) for i in obj]
        elif isinstance(obj, str):
            return obj.replace('\n', ' ')
        else:
            return obj

    cleaned_data = clean(data)

    # Save the cleaned JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

# Example usage
input_file = '/Users/keane/Desktop/research/human_behaviour/human-behavior/data/autism/ados_prompts/scoring.json'
output_file = '/Users/keane/Desktop/research/human_behaviour/human-behavior/data/autism/ados_prompts/cleaned_scoring.json'
clean_newlines_in_json(input_file, output_file)
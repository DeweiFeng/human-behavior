import json

def filter_jsonl_by_key(input_path, output_path, key, value):
    """
    Reads a JSONL file, writes lines where `key == value` to output.
    """
    count_in = 0
    count_out = 0
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            count_in += 1
            try:
                obj = json.loads(line)
                if obj.get(key) == value:
                    outfile.write(json.dumps(obj) + '\n')
                    count_out += 1
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {count_in}")
    print(f"Processed {count_in} lines. Found {count_out} entries with {key} == {value}. Saved to {output_path}.")

if __name__ == "__main__":
    input_file = "/home/keaneong/human-behavior/data/instruction/instruction_prompts.jsonl"
    output_file = "/home/keaneong/human-behavior/data/instruction/meld_instruction_prompts.jsonl"
    key_to_check = "dataset"
    value_to_match = "meld"

    filter_jsonl_by_key(input_file, output_file, key_to_check, value_to_match)
def truncate_jsonl(input_path, output_path, max_lines):
    """
    Truncate a JSONL file to the first `max_lines` and save to another file.
    """
    count = 0
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if count >= max_lines:
                break
            outfile.write(line)
            count += 1
    print(f"Truncated to {count} lines and saved to {output_path}")

if __name__ == "__main__":
    # You can hard-code these or pass them as arguments
    input_file = "/home/keaneong/human-behavior/data/instruction/geom_train_upsampled.jsonl"
    output_file = "/home/keaneong/human-behavior/data/instruction/truncated_geom_train_upsampled.jsonl"
    max_lines = 100  # adjust this to your desired number

    truncate_jsonl(input_file, output_file, max_lines)
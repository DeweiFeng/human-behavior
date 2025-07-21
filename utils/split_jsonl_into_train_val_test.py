import json

def split_jsonl(input_path, train_output, val_output, test_output):
    n_train = 0
    n_val = 0
    n_test = 0
    n_other = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(train_output, 'w', encoding='utf-8') as train_out, \
         open(val_output, 'w', encoding='utf-8') as val_out, \
         open(test_output, 'w', encoding='utf-8') as test_out:

        for line in infile:
            line = line.strip()
            if not line:
                continue  # skip empty lines

            try:
                obj = json.loads(line)
                videos = obj.get("videos", [])
                if not videos:
                    n_other += 1
                    continue

                video_path = videos[0]

                if "train/train_splits" in video_path:
                    train_out.write(json.dumps(obj) + "\n")
                    n_train += 1
                elif "val/val_splits" in video_path:
                    val_out.write(json.dumps(obj) + "\n")
                    n_val += 1
                elif "test/test_splits" in video_path:
                    test_out.write(json.dumps(obj) + "\n")
                    n_test +=1
                else:
                    n_other += 1

            except json.JSONDecodeError:
                print("⚠️ Skipping invalid JSON line.")
                n_other += 1

    print(f"✅ Done.")
    print(f"→ {n_train} lines written to {train_output}")
    print(f"→ {n_val} lines written to {val_output}")
    print(f"→ {n_test} lines written to {test_output}")
    if n_other > 0:
        print(f"⚠️ {n_other} lines did not match any split and were skipped.")

if __name__ == "__main__":
    input_file = "/home/keaneong/human-behavior/data/instruction/meld_instruction_prompts.jsonl"
    train_file = "/home/keaneong/human-behavior/data/instruction/meld_instruction_train.jsonl"
    val_file = "/home/keaneong/human-behavior/data/instruction/meld_instruction_val.jsonl"
    test_file = "/home/keaneong/human-behavior/data/instruction/meld_instruction_test.jsonl"
    split_jsonl(input_file, train_file, val_file, test_file)
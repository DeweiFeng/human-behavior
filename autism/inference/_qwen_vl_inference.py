import argparse
import yaml
import json
import os

from qwen_vl import QwenVL
from utils import get_frame_paths

# TODO: SAVE THE PROMPT CONFIG SUCH THAT IT ALSO INCLUDES DETAILS ABOUT THE TEST EXAMINED
# TODO: ALSO SAVE THE INPUT FILE SEGMENT NAME (perhaps as the json corresponding to that segment)

def load_test_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def find_test_prompt(data, module, test_type):
    for test in data:
        if test["module"] == module and test["test type"] == test_type:
            desc = test["description"]
            labels = test["labels"]
            labels_str = "\n".join(f"- {label}" for label in labels)

            # Construct a clear classification prompt
            prompt = (
                f"DESCRIPTION:\n{desc}\n\n"
                f"TASK:\nClassify the following video according to the possible labels below.\n\n"
                f"LABELS:\n{labels_str}\n\n"
                f"Please output only the classification label."
            )
            return prompt
    raise ValueError(f"Module '{module}' and test type '{test_type}' not found in test JSON.")

def load_non_test_prompt(non_test_prompt):
    if os.path.isfile(non_test_prompt):
        with open(non_test_prompt, "r") as f:
            return f.read().strip()
    else:
        return non_test_prompt.strip()

def main(config):
    # Instantiate model
    qwen = QwenVL(use_flash_attention=config.get("use_flash_attention", False))

    # Determine input type
    test_segment_dir = config["test_segment_dir"]
    segment_name = os.path.basename(test_segment_dir)

    if config["use_video_or_frames"] == "video":
        video_path = os.path.join(test_segment_dir, f"{segment_name}.mp4")
        print(f"Using video: {video_path}")

        messages, meta_data = qwen.prepare_message(
            video_input=video_path,
            prompt=config["prompt"],
            is_video_path=True,
            fps=config.get("fps", 1.0)
        )
    elif config["use_video_or_frames"] == "frames":
        frames_dir = os.path.join(test_segment_dir, f"{segment_name}_frames")
        print(f"Using frames in: {frames_dir}")

        frame_paths = get_frame_paths(
            directory=frames_dir,
            slice_start=config.get("slice_start", 0.0),
            slice_end=config.get("slice_end", 1.0)
        )

        frame_config = {
            "directory": frames_dir,
            "slice_start": config.get("slice_start", 0.0),
            "slice_end": config.get("slice_end", 1.0),
            "frame_paths": frame_paths
        }

        messages, meta_data = qwen.prepare_message(
            video_input=frame_paths,
            prompt=config["prompt"],
            is_video_path=False,
            fps=config.get("fps", 1.0)
        )
        meta_data["frame_config"] = frame_config
    
    else:
        raise ValueError("Invalid use_video_or_frames option. Must be 'video' or 'frames'.")

    # Run inference
    inputs = qwen.process_inputs(messages)
    outputs = qwen.generate(inputs)

    # Save outputs
    os.makedirs(os.path.dirname(config["output_path"]), exist_ok=True)
    qwen.save_outputs(outputs, save_path=config["output_path"], meta_data=meta_data)

    print(f"Inference complete! Saved to {config['output_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QwenVL Inference Script")

    # for shell, set --config None to use CLI args
    parser.add_argument("--config", type=str, default="/home/human-behavior/autism/inference/config_inference.yaml", help="Path to YAML config file")
    
    parser.add_argument("--test_segment_dir", type=str, help="Path to test segment directory")
    parser.add_argument("--use_video_or_frames", type=str, choices=["video", "frames"], help="Choose 'video' or 'frames'")
    parser.add_argument("--fps", type=float, default=1.0, help="FPS for video/frame input")
    parser.add_argument("--slice_start", type=float, default=0.0, help="Frame slice start (0.0 - 1.0)")
    parser.add_argument("--slice_end", type=float, default=1.0, help="Frame slice end (0.0 - 1.0)")
    parser.add_argument("--test_json", type=str, help="Path to test json (if using module + test_type)")
    parser.add_argument("--module", type=str, help="Module to retrieve prompt")
    parser.add_argument("--test_type", type=str, help="Test type to retrieve prompt")
    parser.add_argument("--non_test_prompt", type=str, help="Manual override of prompt (txt file or string)")
    parser.add_argument("--output_path", type=str, help="Path to save outputs")

    args = parser.parse_args()

    # 1️⃣ Priority: YAML config if provided
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        # 2️⃣ Else, use CLI args
        config = vars(args)

    # 3️⃣ Now set prompt
    if config.get("non_test_prompt"):
        config["prompt"] = load_non_test_prompt(config["non_test_prompt"])
    elif config.get("test_json") and config.get("module") and config.get("test_type"):
        data = load_test_json(config["test_json"])
        config["prompt"] = find_test_prompt(data, config["module"], config["test_type"])
    else:
        raise ValueError("You must either provide 'non_test_prompt' OR ('test_json' + 'module' + 'test_type') in config or CLI.")

    # Run main
    main(config)
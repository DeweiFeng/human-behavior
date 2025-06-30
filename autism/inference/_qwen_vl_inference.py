import argparse
import yaml
import os
from qwen_vl import QwenVL
import json
from utils import get_frame_paths, load_rubrics_json, find_test_prompt, load_non_test_prompt

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

    # Appending the remaining meta data
    meta_data["segment_name"] = segment_name

    # If the prompt is from rubrics_json, save module + test_type
    if config.get("rubrics_json") and config.get("module") and config.get("test_type"):
        meta_data["rubrics_json"] = config["rubrics_json"]
        meta_data["module"] = config["module"]
        meta_data["test_type"] = config["test_type"]

    # Save outputs
    os.makedirs(os.path.dirname(config["output_path"]), exist_ok=True)
    qwen.save_outputs(outputs, save_path=config["output_path"], meta_data=meta_data)

    print(f"Inference complete! Saved to {config['output_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QwenVL Inference Script")

    # for shell, set --config None to use CLI args
    parser.add_argument("--config", type=str, default="/home/human-behavior/autism/inference/config_inference.yaml", help="Path to YAML config file")
    
    # set the test segment directory
    parser.add_argument("--test_segment_dir", type=str, help="Path to test segment directory")
    
    # data configuration
    parser.add_argument("--use_video_or_frames", type=str, choices=["video", "frames"], help="Choose 'video' or 'frames'")
    parser.add_argument("--fps", type=float, default=1.0, help="FPS for video/frame input")
    parser.add_argument("--slice_start", type=float, default=0.0, help="Frame slice start (0.0 - 1.0)")
    parser.add_argument("--slice_end", type=float, default=1.0, help="Frame slice end (0.0 - 1.0)")
    
    # prompt configuration
    parser.add_argument("--rubrics_json", type=str, help="Path to test json (if using module + test_type)")
    parser.add_argument("--module", type=str, help="Module to retrieve prompt")
    parser.add_argument("--test_type", type=str, help="Test type to retrieve prompt")
    parser.add_argument("--non_test_prompt", type=str, help="Manual override of prompt (txt file or string)")
    
    # output configuration
    parser.add_argument("--output_path", type=str, help="Path to save outputs")
    args = parser.parse_args()

    # Priority: YAML config if provided
    if args.config:
        print(f"WARNING: Loading YAML config from {args.config}")
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        # Else, use CLI args
        config = vars(args)

    # Now set prompt; if we aren't scoring via the rubrics, use non_test_prompt
    # Otherwise, we score via the rubrics

    if config.get("non_test_prompt"):
        config["prompt"] = load_non_test_prompt(config["non_test_prompt"])
    elif config.get("rubrics_json") and config.get("module") and config.get("test_type"):
        data = load_rubrics_json(config["rubrics_json"])
        config["prompt"] = find_test_prompt(data, config["module"], config["test_type"])
    else:
        raise ValueError(
            f"You must either provide 'non_test_prompt' OR ('rubrics_json' + 'module' + 'test_type') in config or CLI. "
            f"Your configurations are:\n{json.dumps(config, indent=2)}"
        )

    # Run main
    main(config)
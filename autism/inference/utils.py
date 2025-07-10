import os
import glob
import json

def get_frame_paths(directory, slice_start=0.0, slice_end=1.0):
    """
    Retrieve sorted frame paths from a directory and slice by %.

    Args:
        directory (str): Directory containing frames named like frame_0001.jpg
        slice_start (float): Start % between 0.0 and 1.0 (default: 0.0)
        slice_end (float): End % between 0.0 and 1.0 (default: 1.0)

    Returns:
        List[str]: Sorted list of frame paths
    """
    frame_paths = sorted(
        glob.glob(os.path.join(directory, "frame_*.jpg"))
    )

    num_frames = len(frame_paths)

    if num_frames == 0:
        raise RuntimeError(f"No frames found in: {directory}")

    # Clamp values
    slice_start = max(0.0, min(1.0, slice_start))
    slice_end = max(0.0, min(1.0, slice_end))
    if slice_start >= slice_end:
        raise ValueError(f"slice_start must be < slice_end (got {slice_start}, {slice_end})")

    start_idx = int(slice_start * num_frames)
    end_idx = int(slice_end * num_frames)

    selected = frame_paths[start_idx:end_idx]

    print(f"Loaded {len(selected)} frames from {directory} [{slice_start:.2f} - {slice_end:.2f}]")

    return selected


def load_rubrics_json(json_path):
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
import os
import glob

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

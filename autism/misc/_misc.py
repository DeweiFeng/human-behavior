import os
import shutil

def reorganize_segments(frame_root, video_root, output_root):
    # Make sure output root exists
    os.makedirs(output_root, exist_ok=True)

    # List frame folders
    frame_folders = [f for f in os.listdir(frame_root) if f.endswith("_frames")]

    # List video files
    video_files = [f for f in os.listdir(video_root) if f.endswith(".mp4")]

    # Extract segment names
    frame_segments = {f.replace("_frames", ""): f for f in frame_folders}
    video_segments = {os.path.splitext(f)[0]: f for f in video_files}

    # Union of all segments
    all_segments = set(frame_segments.keys()).union(set(video_segments.keys()))

    print(f"Found {len(all_segments)} segments.")

    for segment in sorted(all_segments):
        segment_out_dir = os.path.join(output_root, segment)
        os.makedirs(segment_out_dir, exist_ok=True)

        # Move frame folder if exists
        if segment in frame_segments:
            src_frame = os.path.join(frame_root, frame_segments[segment])
            dst_frame = os.path.join(segment_out_dir, frame_segments[segment])
            print(f"Copying frames: {src_frame} → {dst_frame}")
            shutil.copytree(src_frame, dst_frame, dirs_exist_ok=True)

        # Move video file if exists
        if segment in video_segments:
            src_video = os.path.join(video_root, video_segments[segment])
            dst_video = os.path.join(segment_out_dir, video_segments[segment])
            print(f"Copying video: {src_video} → {dst_video}")
            shutil.copy2(src_video, dst_video)

    print("Done.")

# Example usage
if __name__ == "__main__":
    frame_root = "/home/human-behavior/autism/data/ados_videos_and_frames/sample_1"
    video_root = "/home/human-behavior/autism/data/ados_videos_and_frames/sample_1/video_segments"
    output_root = "/home/human-behavior/autism/data/ados_videos_and_frames/sample_1"

    reorganize_segments(frame_root, video_root, output_root)
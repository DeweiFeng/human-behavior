import os
import subprocess
from yt_dlp import YoutubeDL
from tqdm import tqdm

# ------------------- CONFIG -------------------
VIDEO_URL = "https://www.youtube.com/watch?v=US90ZQyKHR8"
OUTPUT_DIR = "/home/human-behavior/autism/data/ados_segments"
VIDEO_FILE = os.path.join(OUTPUT_DIR, "full_ados_sample_video.mp4")
# Segments: (name, start_time, end_time) in seconds or "mm:ss"
SEGMENTS = [
    ("free_play", "0:00", "9:38"),
    ("response_to_name", "3:14", "3:40"),
    ("response_to_joint_attention_1", "3:59", "4:12"),
    ("response_to_joint_attention_2", "5:49", "6:15"),
    ("bubble_play", "9:38", "10:28"),
    ("anticipation_of_routine_1", "4:56", "5:20"),
    ("anticipation_of_routine_2", "6:31", "7:20"),
    ("anticipation_of_routine_3", "7:40", "7:55"),
    ("anticipation_of_routine_4", "8:05", "8:43"),
    ("anticipation_of_routine_5", "9:38", "11:04"),
    ("anticipation_of_routine_6", "11:05", "13:25"),
    ("anticipation_of_routine_7", "13:35", "14:50"),
    ("responsive_social_smile_1", "10:14", "10:20"),
    ("responsive_social_smile_2", "11:25", "11:30"),
    ("responsive_social_smile_3", "12:46", "12:56"),
    ("responsive_social_smile_4", "12:57", "13:08")
]
# ------------------------------------------------

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_youtube_video(url, output_file):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': output_file,
        'merge_output_format': 'mp4'
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def timestamp_to_seconds(ts):
    if isinstance(ts, int) or isinstance(ts, float):
        return ts
    parts = ts.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    else:
        raise ValueError(f"Bad timestamp: {ts}")

def extract_segment(input_file, start, end, output_file):
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-y",
        "-i", input_file,
        "-ss", str(start),
        "-to", str(end),
        "-c", "copy",
        output_file
    ]
    subprocess.run(cmd, check=True)

def extract_frames(video_file, output_folder, fps=1):
    ensure_dir(output_folder)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video_file,
        "-vf", f"fps={fps}",
        os.path.join(output_folder, "frame_%04d.jpg")
    ]
    subprocess.run(cmd, check=True)

def main():
    ensure_dir(OUTPUT_DIR)
    if not os.path.exists(VIDEO_FILE):
        print("Downloading video...")
        download_youtube_video(VIDEO_URL, VIDEO_FILE)
    else:
        print("Video already downloaded.")

    for name, start, end in tqdm(SEGMENTS, desc="Extracting segments"):
        seg_file = os.path.join(OUTPUT_DIR, f"{name}.mp4")
        start_sec = timestamp_to_seconds(start)
        end_sec = timestamp_to_seconds(end)
        try:
            extract_segment(VIDEO_FILE, start_sec, end_sec, seg_file)
            extract_frames(seg_file, os.path.join(OUTPUT_DIR, f"{name}_frames"))
        except Exception as e:
            print(f"Error extracting {name}: {e}")

    print(f"Segments saved in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
import os
import json
import subprocess
import multiprocessing

# Directories
INPUT_DIR = './videos'  # Change this if needed
OUTPUT_DIR = './segmented_videos'  # Change this if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output directory exists

def local_clip(filename, start_time, duration, output_filename):
    """
    Extracts a video clip using ffmpeg.

    Args:
        filename (str): Path to the input video file.
        start_time (float): Start time of the clip (in seconds).
        duration (float): Duration of the clip (in seconds).
        output_filename (str): Name of the output file.
    """
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Skip if the clip already exists
    if os.path.exists(output_path):
        print(f"Skipping {output_filename}, already exists.")
        return

    # Ensure the input file exists
    if not os.path.exists(filename):
        print(f"Warning: Source video {filename} not found. Skipping {output_filename}.")
        return

    command = [
        'ffmpeg',
        '-i', filename,  # Input file
        '-ss', str(start_time),  # Start time
        '-t', str(duration),  # Clip duration
        '-c:v', 'copy', '-an',  # Copy video, remove audio
        '-threads', '1',
        '-loglevel', 'error',  # Show only errors
        output_path
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Created clip: {output_filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {output_filename}: {e}")

def wrapper(args):
    """
    Wrapper function to process a single clip.

    Args:
        args (tuple): (clip_name, clip_data)
    """
    clip_name, clip = args
    video_id = clip['url'].split('=')[-1]  # Extract YouTube video ID
    input_file = os.path.join(INPUT_DIR, f"{video_id}.mkv.mp4")

    # Skip if input file doesn't exist
    if not os.path.exists(input_file):
        print(f"Warning: Video file {input_file} does not exist. Skipping {clip_name}.")
        return

    duration = clip['end'] - clip['start']
    output_filename = f"{clip_name}.mp4"  # Use metadata key as filename

    local_clip(input_file, clip['start'], duration, output_filename)

def process_clips(json_file, num_workers=8):
    """
    Reads JSON file and processes clips using multiprocessing.

    Args:
        json_file (str): Path to the JSON file.
        num_workers (int): Number of parallel processes to use.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    clips = list(data.items())  # Convert dictionary to list of (key, value) tuples

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(wrapper, clips)

if __name__ == "__main__":
    process_clips('data/mlb-youtube-segmented.json', num_workers=8)

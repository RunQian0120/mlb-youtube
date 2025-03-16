import os
import json
import subprocess

# Set the directory where you want to save the videos
save_dir = './videos'

# Load the JSON data
with open('data/mlb-youtube-segmented.json', 'r') as f:
    data = json.load(f)

# Iterate through each entry in the dataset
total_urls = set()

all_ytid = {}
for video_id, entry in data.items():
    yturl = entry['url']
    # total_urls.add(yturl)
    ytid = yturl.split('=')[-1]
    all_ytid[ytid] = yturl
    
print(f"Downloading {len(all_ytid)}")
for ytid, yturl in all_ytid.items():
    # total_urls.add(yturl)
    output_path = os.path.join(save_dir, f"{ytid}.mkv")

    # Skip if the file already exists
    if os.path.exists(output_path):
        print(f"Skipping {ytid}, already downloaded.")
        continue

    # yt-dlp command
    cmd = [
        'yt-dlp',
        '--cookies', 'youtube_cookies.txt',
        '-f', 'bv*+ba/best',  # Selects best video + best audio
        '-o', output_path,    # Output file path
        yturl
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Downloaded {ytid} successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {ytid}: {e}")

# print(len(total_urls))
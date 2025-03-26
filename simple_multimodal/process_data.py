import numpy as np
import h5py
from PIL import Image
import io
import json

USE_CAPTIONS = True
USE_METADATA = False

all_labels = [
    "Strike", "Ball", "Foul", "Strike out", "Swing and a miss", "Fly out", 
    "Two-base hit", "Ground out", "One-base hit", "Wild pitch", "Homerun", 
    "Home in", "Base on balls", "Double play", "Touch out", "Infield hit", 
    "Foul fly out", "Hit by pitch", "Error", "Line-drive out", 
    "Sacrifice bunt out", "Bunt foul", "Passed ball", "Stealing base", 
    "Tag out", "Caught stealing"
]

allowed_labels = ["Foul", "Swing and a miss", "Fly out", "Ground out", "Ball", "Strike"]

label_indices = {key: index for index, key in enumerate(all_labels)}

def return_sampled_frames(hdf5_file, dataset_name, start_segment, end_segment, duration, num_samples=4):
    with h5py.File(hdf5_file, 'r') as f:
        dataset = f[dataset_name]

        fps = int(len(dataset) / duration)

        start_idx = start_segment * fps
        end_idx = end_segment * fps

        # Generate evenly spaced indices
        sampled_indices = np.linspace(start_idx, end_idx, num_samples, dtype=int)
        images = []
        for idx in sampled_indices:
            images.append(Image.open(io.BytesIO(dataset[idx])))

        return images

def create_all_data():    
    with open('data/bbdb.v0.9.with.inning.min.json', 'r') as file:
        metadata = json.load(file)

    with open('data/captions.json') as file:
        caption_data = json.load(file)

    fps = 6
    video_ids = ['20160401HTNC02016', '20160408SSLT02016', '20160503LTHT02016', '20170808LGSS02017']
    # video_ids = ['20160401HTNC02016', '20160408SSLT02016', '20160503LTHT02016', '20170808LGSS02017', '20170705HTSK02017', '20170705KTOB02017']
    max_annotation_idx = 329  # Adjust as needed

    all_data = []
    for video_id in video_ids:
        for annotation_idx in range(0, max_annotation_idx + 1):
            d = metadata['database'][video_id]['annotations'][annotation_idx]

            duration = metadata['database'][video_id]['duration']

            file = f"data/{video_id}_jpegs.h5"
            start_idx = float(d['segment'][0])
            end_idx = float(d['segment'][1])

            if d["label"] in allowed_labels:
                label_idx = label_indices[d["label"]]
                if USE_CAPTIONS:
                    caption = caption_data[video_id][annotation_idx]['summary']
                elif USE_METADATA:
                    pitch_idx = float(d['pitchTime'])
                    caption = f"Time from start to pitch is {pitch_idx - start_idx} seconds. Time from pitch to end is {end_idx - pitch_idx} seconds."
                else:
                    caption = "Empty Caption"
                images = return_sampled_frames(file, 'jpegs', start_idx, end_idx, duration)
                all_data.append({'images': images, 'caption': caption, 'label': label_idx})
    
    return all_data
    

    
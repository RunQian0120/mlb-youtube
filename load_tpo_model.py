# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
import os
import copy
import torch
import sys
import warnings
import numpy as np
from decord import VideoReader, cpu
from transformers import BitsAndBytesConfig

# Set memory allocation environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Clear CUDA cache
torch.cuda.empty_cache()

# Suppress warnings
warnings.filterwarnings("ignore")

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    
    print(f"Loaded {len(spare_frames)} frames")
    return spare_frames, frame_time, video_time

# Configure 8-bit quantization properly
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf8"
)

print("Loading model...")
pretrained = "ruili0/LLaVA-Video-7B-TPO"
model_name = "llava_qwen"
device = "cuda"

# Load model with proper quantization
tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, 
    None, 
    model_name, 
    quantization_config=quantization_config,
    device_map="auto"
)
model.eval()

# Use fewer frames
print("Loading video...")
video_path = "segmented_videos/0D329CPSHKYV.mp4"
max_frames_num = 8  # Reduced from original
video, frame_time, video_time = load_video(video_path, max_frames_num, 1, force_sample=True)

# Use half precision (float16) instead of trying to use float8
print("Processing frames...")
video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().half()
video = [video]

print("Setting up conversation...")
conv_template = "qwen_1_5"
time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}. Please answer the following questions related to this video."
question = DEFAULT_IMAGE_TOKEN + f"{time_instruction}\nTell me whether this pitch is a strike, ball, hit, etc.\nAlso describe the video."

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

print("Tokenizing input...")
input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

# Clear cache before generation
torch.cuda.empty_cache()

print("Generating response (this may take a while)...")
cont = model.generate(
    input_ids,
    images=video,
    modalities=["video"],
    do_sample=False,
    temperature=0,
    max_new_tokens=512,  # Reduced significantly from 4096
)

text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
print("\nModel output:")
print("-" * 50)
print(text_outputs)
print("-" * 50)

# Final cleanup
torch.cuda.empty_cache()
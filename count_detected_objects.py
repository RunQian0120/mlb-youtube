import cv2
import numpy as np
import os

net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
print("finished loading model")

def detect_objects(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_objects = 0
    print("processing video")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        _, _ = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(out_layers)

        objects_in_frame = 0
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5: 
                    objects_in_frame += 1
        
        total_objects += objects_in_frame

    cap.release()
    avg_objects_per_frame = total_objects / frame_count if frame_count else 0
    return total_objects, avg_objects_per_frame

def process_videos_in_folder(folder_path, output_file):
    with open(output_file, "w") as f:
        for filename in os.listdir(folder_path):
            if filename.endswith(".mp4"):
                video_path = os.path.join(folder_path, filename)
                total_objects, avg_objects = detect_objects(video_path)
                f.write(f"{total_objects}, {avg_objects:.2f}\n")

folder_path = "./segmented_videos"
output_file = "detection_results.txt"
process_videos_in_folder(folder_path, output_file)
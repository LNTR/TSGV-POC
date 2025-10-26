import os
import cv2
import math
import numpy as np
from skimage import transform
from typing import Tuple

def extract_frames_tacos(visual_data_path: str, processed_visual_data_path: str, output_frame_size: Tuple):
    """Extracts frames from the raw videos of TACoS and save them as numpy arrays"""
    if not os.path.exists(processed_visual_data_path):
        os.mkdir(processed_visual_data_path)

    video_files = os.listdir(visual_data_path)

    for video_file in video_files:
        print('processing %s...' % video_file)
        cap = cv2.VideoCapture(os.path.join(visual_data_path, video_file))
        success = 1
        frames = []

        current_frame = 0
        fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))

        while success:
            success, frame = cap.read()
            if success:
                if current_frame % (fps * 5) == 0:  # sampling one frame every five seconds
                    frame = transform.resize(frame, output_frame_size)  # resize the image
                    frames.append(np.expand_dims(frame, axis=0))
            else:
                break
            current_frame += 1

        frames = np.concatenate(frames).astype(np.float32)
        output_file = os.path.join(processed_visual_data_path, video_file.replace('.avi', '.npy'))
        np.save(output_file, frames)

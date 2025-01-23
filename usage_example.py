import ffmpeg
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import video_utils as vu

model = YOLO('yolov11n-face.pt')
path = 'video.mp4'

frames = vu.extract_frames_ffmpeg_numpy(path)
for i, frame in enumerate(frames):
    if i % 10 == 0:
        results = model.predict(frame, imgsz=640)[0]
        faces = vu.extract_faces(frame.copy(), model, results)
        vu.draw_faces(faces)   

frame = cv2.imread('example.jpg')
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = model.predict(frame, imgsz=640)[0]
faces = vu.extract_faces(frame.copy(), model, results)
vu.draw_faces(faces)
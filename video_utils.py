import ffmpeg
import numpy as np
import cv2
import matplotlib.pyplot as plt


def extract_frames_ffmpeg_numpy(video_path, fps=1):
    """
    Extract frames as numpy arrays using FFmpeg bindings.

    Args:
    - video_path (str): Path to the video file.
    - fps (int): Frames per second to extract.

    Returns:
    - list: Extracted frames as numpy arrays.
    """
    # extract frames with FFmpeg and decode them directly into numpy arrays
    process = (
        ffmpeg
        .input(video_path, r=fps)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run(capture_stdout=True, capture_stderr=True)
    )

    # decode raw video stream into numpy
    width, height = 720, 1280  # modify to target dimensions
    video_frames = np.frombuffer(process[0], np.uint8).reshape([-1, height, width, 3])

    return video_frames


def draw_bboxes(frame, model, results):
    """
    Draw bounding boxes and labels on the frame based on YOLO results.

    Args:
    - frame (numpy.ndarray): The image frame to draw on.
    - model: The YOLO model used for predictions.
    - results: YOLO results containing bounding box predictions.

    Returns:
    - numpy.ndarray: The frame with drawn bounding boxes and labels.
    """
    for box in results.boxes:
        xyxy = box.xyxy.cpu().numpy()[0]
        cls = int(box.cls.cpu().numpy())
        conf = box.conf.cpu().numpy()[0]
        label = f"{model.names[cls]} {conf:.2f}"

        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame


def extract_faces(frame, model, results):
    """
    Extract faces from the frame using YOLO model.

    Args:
    - frame (numpy.ndarray): The image frame to process.
    - model: The YOLO model for face detection.
    - results: YOLO results containing bounding box predictions.

    Returns:
    - list: A list of face images as numpy arrays.
    """
    faces = []

    for box in results.boxes:
        xyxy = box.xyxy.cpu().numpy()[0]
        x1, y1, x2, y2 = map(int, xyxy)
        face_image = frame[y1:y2, x1:x2]
        faces.append(face_image)

    return faces


def draw_faces(faces):
    """
    Draw and display the list of face images.

    Args:
    - faces (list): A list of face images as numpy arrays.
    """
    plt.figure(figsize=(15, 10))
    for i, face in enumerate(faces):
        plt.subplot(1, len(faces), i + 1)
        plt.imshow(face)
        plt.axis('off')
        plt.title(f'Face {i + 1}')
    plt.show()
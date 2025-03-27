import ffmpeg
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from ultralytics import YOLO
import torch
import torchvision.models as models
from torchvision import transforms
import timm
import math

def extract_frames_ffmpeg(video_path, fps=1):
    """
    Extract frames as numpy arrays using FFmpeg.

    Arguments:
    - video_path (str): Path to the video file.
    - fps (float): Frame extraction rate (frames per second).

    Returns:
    - numpy.ndarray: Array of extracted frames with shape (num_frames, height, width, 3)
    """
    try:
        # Get video information
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])

        # Extract frames using the 'fps' filter
        process = (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=fps)  # Use filter to control frame rate
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )

        frames = np.frombuffer(process[0], np.uint8).reshape(-1, height, width, 3)
        return frames

    except Exception as e:
        print(f"FFmpeg error: {str(e)}")
        raise

def draw_bboxes(frame, model, results):
    """
    Draw bounding boxes and labels on the frame based on YOLO results.

    Arguments:
    - frame (numpy.ndarray): Image.
    - model: YOLO model for predictions.
    - results: YOLO results with predicted bounding boxes.

    Returns:
    - numpy.ndarray: Image with drawn bounding boxes and labels.
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
    Extract faces from the frame using YOLO.

    Arguments:
    - frame (numpy.ndarray): Image.
    - model: YOLO model for face detection.
    - results: YOLO results with predicted bounding boxes.

    Returns:
    - list: List of face images as numpy arrays.
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
    Draw and display face images.

    Arguments:
    - faces (list): List of face images (numpy arrays).
    """
    plt.figure(figsize=(15, 10))
    for i, face in enumerate(faces):
        plt.subplot(1, len(faces), i + 1)
        plt.imshow(face)
        plt.axis('off')
        plt.title(f'Face {i + 1}')
    plt.show()

def stream_frames(video_path, interval=1/5):
    """
    Stream frames through FFmpeg with an interval of `interval` seconds.

    Arguments:
    - video_path (str): Path to the video file.
    - interval (int): Interval between extracted frames (in seconds).

    Yields:
    - numpy.ndarray: Frames as numpy arrays.
    """
    try:
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])
        process = (
            ffmpeg
            .input(video_path)
            .filter('fps', interval)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        try:
            while True:
                in_bytes = process.stdout.read(width * height * 3)
                if not in_bytes:
                    break
                yield np.frombuffer(in_bytes, np.uint8).reshape(height, width, 3)
        finally:
            process.stdout.close()
            process.wait()
    except Exception as e:
        print(f"FFmpeg error: {str(e)}")
        raise

def get_face_feature_extractor(device='cpu'):
    """
    Load a pre-trained MViT V2 Small model with the classifier head removed for feature extraction.

    Args:
    - device (str): 'cuda' or 'cpu'

    Returns:
    - torch.nn.Module: The feature extraction model.
    """
    model = timm.create_model('mvitv2_small', pretrained=True)
    model.head = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    return model

face_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_face_features_from_faces(faces, feature_extractor, device='cpu'):
    """
    Extract feature vectors for each face image.

    Args:
    - faces (list): List of face images (numpy arrays).
    - feature_extractor (torch.nn.Module): Model used for feature extraction.
    - device (str): 'cuda' or 'cpu'.

    Returns:
    - list: List of feature vectors (torch.Tensor) for each face.
    """
    features = []
    for face in faces:
        try:
            input_tensor = face_transform(face)
        except Exception as e:
            print(f"Error transforming face: {e}")
            continue
        input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            feature = feature_extractor(input_tensor)
        features.append(feature.cpu().squeeze(0))
    return features

def extract_frame_feature(frame, feature_extractor, device='cpu'):
    """
    Extract feature vector for the entire frame.

    Args:
      frame (numpy.ndarray): Input frame.
      feature_extractor (torch.nn.Module): Model used for feature extraction.
      device (str): 'cuda' or 'cpu'.

    Returns:
      torch.Tensor: Feature vector for the frame.
    """
    try:
        input_tensor = face_transform(frame)
    except Exception as e:
        print(f"Error transforming frame: {e}")
        return None
    input_tensor = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        feature = feature_extractor(input_tensor)
    return feature.cpu().squeeze(0)

def extract_face_features_from_video(video_path, yolo_model, feature_extractor, interval=1, device='cpu'):
    """
    Extract face feature vectors from faces detected in video frames.

    Args:
    - video_path (str): Path to the video file.
    - yolo_model: YOLO model for face detection.
    - feature_extractor (torch.nn.Module): Model for extracting face features.
    - interval (int or float): Interval between frames in seconds.
    - device (str): 'cuda' or 'cpu'.

    Returns:
    - dict: A dictionary where keys are frame times and values are lists of face feature vectors.
    """
    face_features_dict = {}
    frames = extract_frames_ffmpeg(video_path, fps=1/interval)
    for idx, frame in tqdm(enumerate(frames), desc="Processing frames"):
        with torch.no_grad():
            results = yolo_model.predict(frame, imgsz=640, device=device, verbose=False)[0]
        faces = extract_faces(frame, yolo_model, results)
        if faces:
            features = extract_face_features_from_faces(faces, feature_extractor, device=device)
            face_features_dict[(idx+1)*interval] = features
        else:
            face_features_dict[(idx+1)*interval] = []
        del frame, results
        if device == 'cuda':
            torch.cuda.empty_cache()
    return face_features_dict

def process_video(video_path, annotation_path, output_path, interval=5, extraction_fps=1):
    """
    Processes a video with a given interval (in seconds):
    - Extracts all frames using the specified extraction FPS.
    - Groups the frames into intervals of 'interval' seconds.
    - For each interval:
         * Performs face detection and feature extraction for each frame.
         * Collects frame features and face features (lists, without averaging).
         * Aggregates all detected face images.
    - From the annotations (CSV), extracts the VAD label for each interval:
         'arousal' and 'valence' are taken from the annotations, with 'dominance' fixed to 0.
    - The 'categorical_label' is determined by selecting from the key emotions:
         'cheerful', 'happy', 'angry', 'nervous', and 'sad'.
         * If all values are 1, the label is set to "neutral".
         * Otherwise, the emotion with the highest value is chosen.
         * In case of a tie, a fixed priority is applied: angry > nervous > sad > happy > cheerful.
    - A dictionary is created for each interval containing all the information.
    - The result is saved to a pickle file, allowing access to each interval by index.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load the YOLO model for face detection
    model = YOLO('video_faces_utils/yolov11n-face.pt').to(device)
    # Load the feature extractor model
    face_feature_extractor = get_face_feature_extractor(device=device)

    # Load annotations from CSV
    with open(annotation_path, 'r') as f:
        headers = f.readline().strip().split(',')
        annotations = [line.strip().split(',') for line in f]

    data = {
        'speaker_id': os.path.basename(video_path).split('_')[0].upper(),
        'frames': []
    }

    # Extract all frames using the given extraction FPS
    all_frames = extract_frames_ffmpeg(video_path, fps=extraction_fps)
    total_frames = all_frames.shape[0]
    frames_per_interval = int(extraction_fps * interval)
    num_intervals = math.ceil(total_frames / frames_per_interval)

    for interval_idx in tqdm(range(num_intervals), desc="Processing intervals"):
        # Determine the range of frames for the current interval
        start_idx = interval_idx * frames_per_interval
        end_idx = min(start_idx + frames_per_interval, total_frames)
        interval_frames = all_frames[start_idx:end_idx]

        # Lists to accumulate features over the interval
        interval_frame_features = []
        interval_face_features = []
        interval_faces = []

        # Process each frame in the interval
        for frame in interval_frames:
            # Face detection for the frame
            with torch.no_grad():
                results = model.predict(frame, imgsz=640, device=device, verbose=False)[0]

            # Extract faces from the frame
            faces = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                face_img = frame[y1:y2, x1:x2]
                faces.append(face_img)
            # Add all detected faces
            interval_faces.extend(faces)

            # Extract features for the frame and its faces
            frame_feat = extract_frame_feature(frame, face_feature_extractor, device=device)
            if frame_feat is not None:
                interval_frame_features.append(frame_feat)
            if faces:
                face_feats = extract_face_features_from_faces(faces, face_feature_extractor, device=device)
                interval_face_features.extend(face_feats)

            del results
            if device == 'cuda':
                torch.cuda.empty_cache()

        # Get VAD label from annotations for the interval (assuming one annotation per interval)
        if interval_idx < len(annotations):
            ann = annotations[interval_idx]
            try:
                arousal = float(ann[1])
                valence = float(ann[2])
            except Exception:
                arousal, valence = 0.0, 0.0
        else:
            arousal, valence = 0.0, 0.0
        dominance = 0.0
        VAD = (arousal, valence, dominance)

        # Determine the categorical_label using key emotions:
        # We consider only: 'cheerful', 'happy', 'angry', 'nervous', 'sad'
        key_emotions = ['cheerful', 'happy', 'angry', 'nervous', 'sad']
        emotion_values = []
        for key in key_emotions:
            try:
                idx_key = headers.index(key)
                val = float(annotations[interval_idx][idx_key])
            except Exception:
                val = 1.0
            emotion_values.append(val)
        if all(v == 1.0 for v in emotion_values):
            categorical_label = "neutral"
        else:
            # In case of a tie, use fixed priority: angry > nervous > sad > happy > cheerful
            priority = {'angry': 5, 'nervous': 4, 'sad': 3, 'happy': 2, 'cheerful': 1}
            max_val = max(emotion_values)
            candidates = [key for key, v in zip(key_emotions, emotion_values) if v == max_val]
            candidates.sort(key=lambda x: priority[x], reverse=True)
            categorical_label = candidates[0]

        # Use the end time of the interval as representative time
        interval_time = (interval_idx + 1) * interval
        frame_dict = {
            "second": interval_time,
            "faces": interval_faces,                   # List of face images over the interval
            "face_features": interval_face_features,   # List of face feature vectors
            "frame_features": interval_frame_features, # List of frame feature vectors
            "VAD": VAD,
            "categorical_label": categorical_label,
            "audio": 0,   # Placeholder for audio
            "text": 0     # Placeholder for text
        }
        data['frames'].append(frame_dict)

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

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


# async attempt
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
        # Get video information
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])

        # Use the 'select' filter to extract frames at the desired interval
        process = (
            ffmpeg
            .input(video_path)
            .filter('fps', interval)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            # .run(capture_stdout=True, capture_stderr=True, quiet=True)
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


def process_speaker(video_path, annotation_path, output_path, interval=5):
    """
    Process video with an interval of 'interval' seconds:
    - Face detection using the YOLO model.
    - Extracted faces are saved as matrices (numpy arrays) without compression.
    - Results are saved in a pickle file.

    Arguments:
    - video_path (str): Path to the video file.
    - annotation_path (str): Path to the annotation file.
    - output_path (str): Path for saving the results (pickle).
    - interval (int): Interval between extracted frames (in seconds).
    """
    # Determine the device (GPU if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('video_faces_utils/yolov11n-face.pt').to(device)

    # Load annotations
    with open(annotation_path, 'r') as f:
        headers = f.readline().strip().split(',')
        annotations = [line.strip().split(',') for line in f]

    # Prepare data for saving
    data = {
        'speaker_id': os.path.basename(video_path).split('_')[0].upper(),
        'frames': [],
        'annotations': annotations,
        'headers': headers
    }

    # Stream process frames
    frames = extract_frames_ffmpeg(video_path, fps=1 / interval)
    for idx, frame in tqdm(enumerate(frames)):
        # Face detection on the frame
        with torch.no_grad():
            results = model.predict(frame, imgsz=640, device=device, verbose=False)[0]

        # Extract faces as matrices
        faces = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            face = frame[y1:y2, x1:x2]
            faces.append(face)

        # Save data for the frame
        data['frames'].append({
            'second': (idx + 1) * interval,
            'faces': faces,
            'annotation': annotations[idx] if idx < len(annotations) else []
        })

        # Clear memory
        del frame, results
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Save results to a pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

def get_face_feature_extractor(device='cpu'):
    """
    Load a pre-trained ResNet18 model with the final layer removed for feature extraction.

    Args:
    - device (str): 'cuda' or 'cpu'

    Returns:
    - torch.nn.Module: The feature extraction model.
    """
    model = models.resnet18(pretrained=True)
    # Replace the last fully connected layer with an identity function to obtain features
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    return model

# Define the transformation pipeline for face images
face_transform = transforms.Compose([
    transforms.ToPILImage(),             # Convert the input image (numpy array) to a PIL Image.
    transforms.Resize((224, 224)),         # Resize the image to 224x224 pixels.
    transforms.ToTensor(),                 # Convert the image to a PyTorch tensor and scale pixel values to [0, 1].
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean.
                         std=[0.229, 0.224, 0.225])   # Normalize using ImageNet standard deviation.
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
        # Extract faces from the frame
        faces = extract_faces(frame, yolo_model, results)
        # If faces are found, extract features for each face
        if faces:
            features = extract_face_features_from_faces(faces, feature_extractor, device=device)
            face_features_dict[(idx+1)*interval] = features
        else:
            face_features_dict[(idx+1)*interval] = []
        # Free memory
        del frame, results
        if device == 'cuda':
            torch.cuda.empty_cache()
    return face_features_dict

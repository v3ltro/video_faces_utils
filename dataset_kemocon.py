import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np


class KEmoConDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
        self.data = self._load_data()
        self.frame_index_mapping = []
        for sample_idx, sample in enumerate(self.data):
            frames = sample.get("frames", [])
            for frame_idx in range(len(frames)):
                self.frame_index_mapping.append((sample_idx, frame_idx))

    def _load_data(self):
        data = []
        print(f"Data files found: {self.data_files}")
        for file in self.data_files:
            file_path = os.path.join(self.data_dir, file)
            print(f"Loading file: {file_path}")
            try:
                with open(file_path, 'rb') as f:
                    sample = pickle.load(f)
                    data.append(sample)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        return data

    def __len__(self):
        return len(self.frame_index_mapping)

    def __getitem__(self, idx):
        sample_idx, interval_idx = self.frame_index_mapping[idx]
        sample = self.data[sample_idx]
        interval_data = sample["frames"][interval_idx]

        video_features = interval_data.get("frame_features", [])
        face_features = interval_data.get("face_features", [])

        combined_video_features = {
            "frame_features": video_features,
            "face_features": face_features
        }

        vad = interval_data.get("VAD", (0.0, 0.0, 0.0))
        if not isinstance(vad, torch.Tensor):
            vad_label = torch.tensor(vad, dtype=torch.float32)
        else:
            vad_label = vad

        categorical_label = interval_data.get("categorical_label", sample.get("categorical_label", "unknown"))

        return {
            "video": combined_video_features,
            "audio": interval_data.get("audio", 0),
            "text": interval_data.get("text", 0),
            "label": vad_label,
            "categorical_label": categorical_label
        }

    def get_frame_features(self, idx):
        sample_idx, frame_idx = self.frame_index_mapping[idx]
        return self.data[sample_idx]['frames'][frame_idx].get('frame_features')

    def get_face_features(self, idx):
        sample_idx, frame_idx = self.frame_index_mapping[idx]
        return self.data[sample_idx]['frames'][frame_idx].get('face_features')

    def get_annotations(self, idx):
        sample_idx, frame_idx = self.frame_index_mapping[idx]
        return self.data[sample_idx]['frames'][frame_idx].get('annotation')

    def get_sample_info(self, idx):
        sample_idx, _ = self.frame_index_mapping[idx]
        return {
            "file": self.data_files[sample_idx],
            "num_frames": len(self.data[sample_idx]['frames'])
        }

    def get_all_annotations(self):
        return [sample.get('annotations', None) for sample in self.data]

    def get_all_face_features(self):
        return [sample['frames'][0].get('face_features', None) for sample in self.data]

    def get_all_frame_features(self):
        return [sample['frames'][0].get('frame_features', None) for sample in self.data]

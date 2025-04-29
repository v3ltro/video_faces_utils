import os
import csv
import shutil
import pandas as pd
import re

faces_root = r'F:/Users/Ivan/Desktop/faces_by_label'
annotations_root = r'F:/Users/Ivan/Desktop/aggregated_external_annotations'
output_root = r'F:/Users/Ivan/Desktop/vad_dataset'

os.makedirs(os.path.join(output_root, 'images'), exist_ok=True)

csv_path = os.path.join(output_root, 'vad_dataset.csv')
csv_file = open(csv_path, mode='w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['path', 'valence', 'arousal'])

for speaker_folder in os.listdir(faces_root):
    speaker_path = os.path.join(faces_root, speaker_folder)
    if not os.path.isdir(speaker_path):
        continue

    speaker_id = speaker_folder.upper()
    annotation_path = os.path.join(annotations_root, f"{speaker_id}.external.csv")
    if not os.path.exists(annotation_path):
        print(f"No annotation for {speaker_id}, skipping...")
        continue

    annotations = pd.read_csv(annotation_path)

    for emotion_folder in os.listdir(speaker_path):
        emotion_path = os.path.join(speaker_path, emotion_folder)
        if not os.path.isdir(emotion_path):
            continue

        for img_name in os.listdir(emotion_path):
            if not img_name.endswith(('.png', '.jpg', '.jpeg')):
                continue

            match = re.search(r'interval(\d+)', img_name)
            if match:
                frame_num = int(match.group(1))
            else:
                continue

            interval_idx = frame_num - 1

            if interval_idx >= len(annotations):
                continue

            try:
                valence = float(annotations.iloc[interval_idx]['valence'])
                arousal = float(annotations.iloc[interval_idx]['arousal'])
            except Exception:
                continue

            src_img_path = os.path.join(emotion_path, img_name)
            dst_img_name = f"{speaker_id}_{emotion_folder}_{img_name}"
            dst_img_path = os.path.join(output_root, 'images', dst_img_name)
            shutil.copy(src_img_path, dst_img_path)

            relative_path = os.path.join('images', dst_img_name)
            csv_writer.writerow([relative_path, valence, arousal])

csv_file.close()

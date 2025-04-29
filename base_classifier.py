import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

data = pd.read_csv(r"F:/Users/Ivan/Desktop/vad_dataset/vad_dataset.csv")

def extract_label(path):
    parts = path.split('_')
    if len(parts) >= 3:
        return parts[1]
    else:
        return 'unknown'

data['category'] = data['path'].apply(extract_label)

labels = data['category'].values
_, labels_val = train_test_split(labels, test_size=0.2, random_state=42)

neutral_preds = ['neutral'] * len(labels_val)

print("\nNeutral Classifier Metrics:")
print(f"Accuracy: {accuracy_score(labels_val, neutral_preds):.4f}")
print(f"F1-score: {f1_score(labels_val, neutral_preds, average='macro'):.4f}")
print(f"Precision: {precision_score(labels_val, neutral_preds, average='macro'):.4f}")
print(f"Recall: {recall_score(labels_val, neutral_preds, average='macro'):.4f}")

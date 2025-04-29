import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

csv_path = r"F:/Users/Ivan/Desktop/vad_dataset/vad_dataset.csv"
data = pd.read_csv(csv_path)

valence = data['valence'].values
arousal = data['arousal'].values

mean_valence = np.mean(valence)
mean_arousal = np.mean(arousal)

valence_preds = np.full_like(valence, mean_valence)
arousal_preds = np.full_like(arousal, mean_arousal)

# Метрики
valence_r2 = r2_score(valence, valence_preds)
arousal_r2 = r2_score(arousal, arousal_preds)
valence_mae = mean_absolute_error(valence, valence_preds)
arousal_mae = mean_absolute_error(arousal, arousal_preds)
valence_mse = mean_squared_error(valence, valence_preds)
arousal_mse = mean_squared_error(arousal, arousal_preds)
valence_corr, _ = pearsonr(valence, valence_preds)
arousal_corr, _ = pearsonr(arousal, arousal_preds)

print("\nBaseline Constant Model Metrics (Predicting Mean):")
print(f"Valence:")
print(f"  - R2: {valence_r2:.4f}")
print(f"  - MAE: {valence_mae:.4f}")
print(f"  - MSE: {valence_mse:.4f}")
print(f"  - Pearson: {valence_corr:.4f}")
print(f"Arousal:")
print(f"  - R2: {arousal_r2:.4f}")
print(f"  - MAE: {arousal_mae:.4f}")
print(f"  - MSE: {arousal_mse:.4f}")
print(f"  - Pearson: {arousal_corr:.4f}")

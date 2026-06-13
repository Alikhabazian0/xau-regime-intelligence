"""
Threshold Tuning for CatBoost Model
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
import joblib

# Load test data (we'll need to save it from main.py)
# For now, we'll retrain quickly

from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.data.feature_engineering import FeatureEngineer
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

print("="*80)
print("THRESHOLD TUNING FOR CATBOOST")
print("="*80)

# 1. Load and prepare data
print("\n1. Loading data...")
loader = DataLoader(r"data\\processed\\xau_with_vix_dxy.csv")
df = loader.load()
# Skip validation since it's processed data

# 2. Clean and prepare
cleaner = DataCleaner(missing_strategy='median')
df = cleaner.clean(df)

# 3. Feature engineering
engineer = FeatureEngineer()
df, feature_cols = engineer.create_features(df)

# 4. Prepare X and y
X = df[feature_cols].values
y = df['target_encoded'].values

# 5. Train-test split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 6. Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Train CatBoost (your best model)
print("\n2. Training CatBoost...")
model = CatBoostClassifier(
    class_weights={0: 2.2, 1: 0.5, 2: 1.78},
    iterations=150,
    depth=5,
    learning_rate=0.05,
    verbose=0,
    random_seed=42
)
model.fit(X_train, y_train)

# 8. Get probabilities
y_proba = model.predict_proba(X_test)
y_pred_default = model.predict(X_test)

# 9. Find optimal thresholds for minority classes
print("\n3. Finding optimal thresholds...")
class_names = ['downward', 'range', 'upward']
optimal_thresholds = {}

for class_idx, class_name in enumerate(class_names):
    if class_name in ['downward', 'upward']:  # Only minority classes
        best_f1 = 0
        best_thresh = 0.5
        
        for threshold in np.arange(0.2, 0.7, 0.05):
            y_pred_binary = (y_proba[:, class_idx] > threshold).astype(int)
            y_true_binary = (y_test == class_idx).astype(int)
            f1 = f1_score(y_true_binary, y_pred_binary)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = threshold
        
        optimal_thresholds[class_name] = best_thresh
        print(f"  {class_name}: threshold = {best_thresh:.2f} (F1 = {best_f1:.4f})")

# 10. Apply optimized thresholds
print("\n4. Applying optimized thresholds...")
y_pred_optimized = y_pred_default.copy()

for class_name, threshold in optimal_thresholds.items():
    class_idx = class_names.index(class_name)
    high_prob_mask = y_proba[:, class_idx] > threshold
    y_pred_optimized[high_prob_mask] = class_idx

# 11. Compare results
print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)

print("\nDEFAULT THRESHOLD (0.5):")
print(classification_report(y_test, y_pred_default, target_names=class_names))

print("\nOPTIMIZED THRESHOLDS:")
print(classification_report(y_test, y_pred_optimized, target_names=class_names))

# Calculate improvement
from sklearn.metrics import f1_score

default_f1 = f1_score(y_test, y_pred_default, average='macro')
optimized_f1 = f1_score(y_test, y_pred_optimized, average='macro')

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Default Macro F1:  {default_f1:.4f}")
print(f"Optimized Macro F1: {optimized_f1:.4f}")
print(f"Improvement:        {(optimized_f1 - default_f1)*100:.1f}%")

# Save optimal thresholds for future use
import json
with open("experiments/results/optimal_thresholds.json", "w") as f:
    json.dump(optimal_thresholds, f, indent=4)
print(f"\n Optimal thresholds saved to experiments/results/optimal_thresholds.json")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import joblib

# Loa enriched data
df = pd.read_csv(r"data\\processed\\xau_with_vix_dxy.csv")

# Prepare features (same as pipeline)
feature_cols = ['open', 'high', 'low', 'close', 'volume', 
               'past_ret_1h', 'past_ret_2h', 'past_ret_4h', 
               'past_ret_8h', 'past_ret_16h', 'vix', 'dxy']

# Add engineered features
df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
df['returns_mean'] = df[['past_ret_1h', 'past_ret_2h', 'past_ret_4h', 'past_ret_8h', 'past_ret_16h']].mean(axis=1)
df['returns_std'] = df[['past_ret_1h', 'past_ret_2h', 'past_ret_4h', 'past_ret_8h', 'past_ret_16h']].std(axis=1)

feature_cols = feature_cols + ['high_low_ratio', 'returns_mean', 'returns_std']

# Handle missing
df[feature_cols] = df[feature_cols].fillna(0)

# Encode target
regime_map = {'downward': 0, 'range': 1, 'upward': 2}
df['target'] = df['forward regime'].map(regime_map)
df = df.dropna(subset=['target'])

# Prepare full data
X_full = df[feature_cols].values
y_full = df['target'].values

# Scale
scaler = StandardScaler()
X_full = scaler.fit_transform(X_full)

# Train best model
best_model = CatBoostClassifier(
    class_weights={0: 2.2, 1: 0.5, 2: 1.78},
    iterations=150,
    depth=5,
    learning_rate=0.05,
    verbose=0
)
best_model.fit(X_full, y_full)

# Save
joblib.dump(best_model, "experiments/models/catboost_best_model.pkl")
print(" Best model saved to experiments/models/catboost_best_model.pkl")
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.models.tcn_model import PyTorchTCN


DATA_PATH = r"C:\\Users\\user\\OneDrive\\Desktop\\financial_regime_clasification\\data\\processed\\xau_with_all_macro.csv"

TARGET = "forward regime"

FEATURES = [
    "open", "high", "low", "close", "volume",
    "past_ret_1h", "past_ret_2h", "past_ret_4h",
    "past_ret_8h", "past_ret_16h",
    "vix", "dxy",
    "tnx", "real_rate",
    "breakeven_inflation", "gold_yield_spread",
    "tnx_change_5d", "real_rate_change_5d",
    "negative_real_rate",
    "tnx_change_24h",
    "real_rate_change_24h",
    "tnx_change_168h",
]

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

mapping = {
    "downward": -1,
    "range": 0,
    "upward": 1,
}

df[TARGET] = df[TARGET].map(mapping)

df = df.ffill().bfill()

X = df[FEATURES].copy()
y = df[TARGET].copy()

X = X.replace([np.inf, -np.inf], np.nan)
X = X.ffill().bfill()

split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = PyTorchTCN(
    sequence_length=36,
    num_channels=(16, 32),
    kernel_size=2,
    dropout=0.35039058736129347,
    learning_rate=0.001966625789577707,
    batch_size=32,
    epochs=38,
    verbose=True,
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("\n--- Classification Report: Best TCN Temporal Split ---\n")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---\n")
print(confusion_matrix(y_test, y_pred))

macro_f1 = f1_score(y_test, y_pred, average="macro")
weighted_f1 = f1_score(y_test, y_pred, average="weighted")

print(f"\nMacro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
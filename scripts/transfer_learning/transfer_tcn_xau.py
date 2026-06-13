import torch
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.models.tcn_model import TCNClassifier, PyTorchTCN


DATA_PATH = r"data\\processed\\xau_with_all_macro.csv"
PRETRAIN_PATH = r"experiments\\pretraining\\pretrained_tcn.pt"

TARGET = "forward regime"

FEATURES = [
    "open", "high", "low", "close", "volume",
    "past_ret_1h", "past_ret_2h", "past_ret_4h",
    "past_ret_8h", "past_ret_16h",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

X = df[FEATURES].replace([np.inf, -np.inf], np.nan).ffill().bfill()
y = df[TARGET]

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
    dropout=0.25,
    epochs=38,
    batch_size=32,
    learning_rate=0.001966625789577707,
    verbose=True,
)

# First initialize model structure by fitting label encoder manually
model.input_size_ = X_train_scaled.shape[1]
model.label_encoder_.fit(y_train)

num_classes = len(model.label_encoder_.classes_)

model.model = TCNClassifier(
    input_size=model.input_size_,
    num_classes=num_classes,
    num_channels=model.num_channels,
    kernel_size=model.kernel_size,
    dropout=model.dropout,
).to(DEVICE)

checkpoint = torch.load(PRETRAIN_PATH, 
                        map_location=DEVICE,
                        weights_only=False
                        )

pretrained_state = checkpoint["model_state_dict"]
current_state = model.model.state_dict()

compatible_state = {}

# ver 0
#for k, v in pretrained_state.items():
#    if k in current_state and current_state[k].shape == v.shape:
#        compatible_state[k] = v

# app ver 1 - This loads only the TCN feature extractor, not the Silver-trained classifier head
for k, v in pretrained_state.items():
    if k.startswith("linear"):
        continue

    if k in current_state and current_state[k].shape == v.shape:
        compatible_state[k] = v

current_state.update(compatible_state)
model.model.load_state_dict(current_state)

print(f"Loaded {len(compatible_state)} pretrained layers.")

# Fine-tune on XAUUSD
model.fit(X_train_scaled, y_train)

pred = model.predict(X_test_scaled)

print("\n--- Transfer TCN Classification Report ---\n")
print(classification_report(y_test, pred))

print("\n--- Confusion Matrix ---\n")
print(confusion_matrix(y_test, pred))

macro_f1 = f1_score(y_test, pred, average="macro")
weighted_f1 = f1_score(y_test, pred, average="weighted")

print(f"\nTransfer TCN Macro F1: {macro_f1:.4f}")
print(f"Transfer TCN Weighted F1: {weighted_f1:.4f}")
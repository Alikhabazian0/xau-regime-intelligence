import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from src.models.tcn_model import PyTorchTCN

TARGET = "forward regime"

DROP_COLUMNS = [
    "date",
    "unnamed: 0",
    TARGET,
    "probability",
    "0.6 percent prediction",
    "1 percent prediction",
    "1.5 percent prediction"
]

df = pd.read_csv(r"C:\\Users\\user\\OneDrive\\Desktop\\financial_regime_clasification\\data\\processed\\xau_with_all_macro.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

feature_cols = [col for col in df.columns if col not in DROP_COLUMNS]

X = df[feature_cols].copy()
y = df[TARGET].copy()

X = X.replace([np.inf, -np.inf], np.nan)
X = X.ffill().bfill()

split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
X_test  = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = PyTorchTCN(
    sequence_length=48,
    num_channels=(32,32),
    kernel_size=3,
    dropout=0.4,
    epochs=30,
    batch_size=32,
    learning_rate=0.001,
    verbose=True,
)

model.fit(X_train_scaled, y_train)
pred = model.predict(X_test_scaled)

print("classification report")
print(classification_report(y_test, pred))

print("confusion matrixx")
print(confusion_matrix(y_test, pred))
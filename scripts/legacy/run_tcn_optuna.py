import os
import torch
import joblib

import json
import optuna 
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from src.models.tcn_model import PyTorchTCN

TARGET = "forward regime"

DROP_COLUMNS = [
    "date",
    "Unnamed: 0",
    TARGET,
    "probability",
    "0.6 percent prediction",
    "1 percent prediction",
    "1.5 percent prediction",
]

df = pd.read_csv(r"C:\\Users\\user\\OneDrive\\Desktop\\financial_regime_clasification\\data\\processed\\xau_with_all_macro.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

feature_cols = [c for c in df.columns if c not in DROP_COLUMNS]

X = df[feature_cols].copy()
y = df[TARGET].copy()

X = X.replace([np.inf, -np.inf], np.nan)
X = X.ffill().bfill()

split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
X_valid = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_valid = y.iloc[split_idx:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

def objective(trial):
    sequence_length = trial.suggest_categorical(
        "sequence_length",
        [24, 36, 48]
    )

    channel_choice = trial.suggest_categorical(
        "num_channels",
        [
            "16_32",
            "32_32",
            "32_64",
            "32_64_32",
            "64_64",
        ]
    )

    channel_map = {
        "16_32": (16, 32),
        "32_32": (32,32),
        "32_64": (32,64),
        "32_64_32": (32, 64, 32),
        "64_64": (64, 64),
    }

    num_channels = channel_map[channel_choice]

    kernel_size = trial.suggest_categorical(
        "kernel_size",
        [2,3,5]
    )

    dropout = trial.suggest_float(
        "dropout",
        0.25,
        0.60
    )

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-4,
        3e-3,
        log=True
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [16, 32, 64]
    )

    epochs = trial.suggest_int(
        "epochs",
        20,
        50
    )

    model = PyTorchTCN(
        sequence_length=sequence_length,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=False,
    )

    model.fit(X_train_scaled, y_train)

    pred = model.predict(X_valid_scaled)

    macro_f1 = f1_score(
        y_valid,
        pred,
        average="macro"
    )

    return macro_f1

study = optuna.create_study(direction="maximize")

study.optimize(
    objective,
    n_trials=50  # go for 50
)

print("Best Trial:")
print(study.best_trial)

print("best params:")
print(study.best_params)

os.makedirs("experiments/optimization", exist_ok=True)

with open ("experiments/optimization/best_tcn_params.json", "w") as f:
    json.dump(study.best_params, f, indent=4)

results_df = study.trials_dataframe()
results_df.to_csv("experiments/optimization/tcn_optuna_results.csv", index=False)

best_params = study.best_params

channel_map = {
    "16_32": (16, 32),
    "32_32": (32, 32),
    "32_64": (32, 64),
    "32_64_32": (32, 64, 32),
    "64_64": (64, 64),
}

best_model = PyTorchTCN(
    sequence_length=best_params["sequence_length"],
    num_channels=channel_map[best_params["num_channels"]],
    kernel_size=best_params["kernel_size"],
    dropout=best_params["dropout"],
    epochs=best_params["epochs"],
    batch_size=best_params["batch_size"],
    learning_rate=best_params["learning_rate"],
    verbose=True,
)

best_model.fit(X_train_scaled, y_train)

best_pred = best_model.predict(X_valid_scaled)

print("\nBest model classification Report")
print(classification_report(y_valid, best_pred))

print("\nBest model confusion matrix")
print(confusion_matrix(y_valid, best_pred))

torch.save(
    {
        "model_state_dict": best_model.model.state_dict(),
        "input_size": best_model.input_size_,
        "sequence_length": best_model.sequence_length,
        "num_channels": best_model.num_channels,
        "kernel_size": best_model.kernel_size,
        "dropout": best_model.dropout,
        "epochs": best_model.epochs,
        "batch_size": best_model.batch_size,
        "learning_rate": best_model.learning_rate,
        "label_encoder": best_model.label_encoder_,
        "feature_cols": feature_cols,
        "best_params": best_params,
        "best_macro_f1": study.best_value,
    },
    "experiments/optimization/best_tcn_model.pt"
)

joblib.dump(
    scaler,
    "experiments/optimization/best_tcn_scaler.pkl"
)

print("\nModel saved successfully.")
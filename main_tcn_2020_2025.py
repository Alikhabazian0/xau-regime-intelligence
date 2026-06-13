import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)

from src.models.tcn_model import PyTorchTCN


DATA_PATH = r"data\processed\xau_2020_2025_features.csv"

FEATURES = [
    'high_low_ratio',
    'close_open_mometum',
    'typical_price',
    'price_position',
    'returns_mean',
    'returns_std',
    'returns_max',
    'returns_min',
    'retrun_range',
    'returns_trend',
    'rsi',
    'macd',
    'macd_signal',
    'macd_histogram',
    'volatiltiy_10'
]


def main():

    print("=" * 60)
    print("TCN - XAU 2020-2025")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    mapping = {
        "downward": -1,
        "range": 0,
        "upward": 1
    }

    df["target"] = df["forward_regime"].map(mapping)

    X = df[FEATURES]
    y = df["target"]

    split_idx = int(len(df) * 0.8)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    print("\nTrain rows:", len(X_train))
    print("Test rows :", len(X_test))

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = PyTorchTCN(
        sequence_length=36,
        num_channels=(16, 32),
        kernel_size=2,
        dropout=0.35,
        learning_rate=0.001966625789577707,
        batch_size=32,
        epochs=38,
        verbose=True
    )

    print("\nTraining TCN...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print("\n=== Classification Report ===\n")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[-1, 0, 1],
            target_names=[
                "downward",
                "range",
                "upward"
            ]
        )
    )

    print("\n=== Confusion Matrix ===\n")

    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=[-1, 0, 1]
        )
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    print("\nMacro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))


if __name__ == "__main__":
    main()
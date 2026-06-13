import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    classification_report
)

from src.models.tcn_model import PyTorchTCN


DATA_PATH = r"data\\processed\\xau_2020_2025_features.csv"

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


def train_and_evaluate(
    train_df,
    test_df,
    fold_name
):

    mapping = {
        "downward": -1,
        "range": 0,
        "upward": 1
    }

    X_train = train_df[FEATURES]
    X_test = test_df[FEATURES]

    y_train = train_df["forward_regime"].map(mapping)
    y_test = test_df["forward_regime"].map(mapping)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = PyTorchTCN(
        sequence_length=36,
        num_channels=(16, 32),
        kernel_size=2,
        dropout=0.35,
        learning_rate=0.001966625789577707,
        batch_size=32,
        epochs=20,
        verbose=False
    )

    print("\n" + "=" * 60)
    print(fold_name)
    print("=" * 60)

    print(
        f"Train: {train_df['date'].min()} -> "
        f"{train_df['date'].max()}"
    )

    print(
        f"Test : {test_df['date'].min()} -> "
        f"{test_df['date'].max()}"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

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

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[-1, 0, 1]
        )
    )

    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")

    return {
        "fold": fold_name,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }


def main():

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")

    results = []

    #
    # Fold 1
    # Train: 2020-2022
    # Test : 2023
    #

    train = df[df["date"] < "2023-01-01"]
    test = df[
        (df["date"] >= "2023-01-01")
        &
        (df["date"] < "2024-01-01")
    ]

    results.append(
        train_and_evaluate(
            train,
            test,
            "Fold_1_2023"
        )
    )

    #
    # Fold 2
    # Train: 2020-2023
    # Test : 2024
    #

    train = df[df["date"] < "2024-01-01"]
    test = df[
        (df["date"] >= "2024-01-01")
        &
        (df["date"] < "2025-01-01")
    ]

    results.append(
        train_and_evaluate(
            train,
            test,
            "Fold_2_2024"
        )
    )

    #
    # Fold 3
    # Train: 2020-2024
    # Test : 2025
    #

    train = df[df["date"] < "2025-01-01"]
    test = df[
        (df["date"] >= "2025-01-01")
    ]

    results.append(
        train_and_evaluate(
            train,
            test,
            "Fold_3_2025"
        )
    )

    print("\n")
    print("=" * 60)
    print("WALK FORWARD SUMMARY")
    print("=" * 60)

    summary = pd.DataFrame(results)

    print(summary)

    print("\nAverage Macro F1:")
    print(summary["macro_f1"].mean())

    print("\nAverage Weighted F1:")
    print(summary["weighted_f1"].mean())


if __name__ == "__main__":
    main()
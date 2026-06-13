import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)

DATA_PATH = r"data\\processed\\xau_2020_2025_dataset.csv"

FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "past_ret_1h",
    "past_ret_2h",
    "past_ret_4h",
    "past_ret_8h",
    "past_ret_16h",
]


def main():

    print("=" * 60)
    print("RANDOM FOREST - XAU 2020-2025")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    mapping = {
        "downward": -1,
        "range": 0,
        "upward": 1
    }

    df["target"] = df["forward_regime"].map(mapping)

    print("\nDataset")
    print(f"Rows: {len(df)}")

    print("\nClass distribution")
    print(df["forward_regime"].value_counts(normalize=True).round(4))

    X = df[FEATURES]
    y = df["target"]

    # temporal split
    split_idx = int(len(df) * 0.8)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    print("\nTrain rows:", len(X_train))
    print("Test rows :", len(X_test))

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining RF...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

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

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[-1, 0, 1]
    )

    print(cm)

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

    print("\n=== Feature Importance ===\n")

    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    )

    print(importance_df)


if __name__ == "__main__":
    main()
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)

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

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n=== Classification Report ===\n")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[-1,0,1],
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
            labels=[-1,0,1]
        )
    )

    print(
        "\nMacro F1:",
        round(
            f1_score(
                y_test,
                y_pred,
                average="macro"
            ),
            4
        )
    )

    print(
        "Weighted F1:",
        round(
            f1_score(
                y_test,
                y_pred,
                average="weighted"
            ),
            4
        )
    )

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    })

    print("\n=== Feature Importance ===\n")
    print(
        importance.sort_values(
            "importance",
            ascending=False
        )
    )

if __name__ == "__main__":
    main()
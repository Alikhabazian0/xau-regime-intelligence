import os
import pandas as pd
import numpy as np

from src.data.feature_engineering import FeatureEngineer


RAW_PATH = r"data\\raw\\xauusd_1h_2020_2025.csv"
OUT_PATH = r"data\\processed\\xau_2020_2025_dataset.csv"

HORIZON = 8
VOL_WINDOW = 24
K = 1.25


def add_past_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["past_ret_1h"] = df["close"].pct_change(1)
    df["past_ret_2h"] = df["close"].pct_change(2)
    df["past_ret_4h"] = df["close"].pct_change(4)
    df["past_ret_8h"] = df["close"].pct_change(8)
    df["past_ret_16h"] = df["close"].pct_change(16)

    return df


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    df = pd.read_csv(RAW_PATH)

    df.columns = df.columns.str.strip().str.lower()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    required_cols = ["date", "open", "high", "low", "close", "volume"]

    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    print("\nLoaded raw XAUUSD data")
    print(f"Rows: {len(df)}")
    print(f"Start: {df['date'].min()}")
    print(f"End: {df['date'].max()}")

    df = add_past_returns(df)

    fe = FeatureEngineer()

    df = fe.add_volatility_regime_labels(
        df,
        horizon=HORIZON,
        vol_window=VOL_WINDOW,
        k=K,
    )

    final_cols = [
        "date",
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
        "forward_regime",
    ]

    df = df[final_cols].copy()

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna().reset_index(drop=True)

    print("\nFinal dataset")
    print(f"Rows: {len(df)}")
    print(f"Start: {df['date'].min()}")
    print(f"End: {df['date'].max()}")

    print("\nClass distribution")
    print(df["forward_regime"].value_counts())
    print("\nClass distribution percentage")
    print(df["forward_regime"].value_counts(normalize=True).round(4))

    df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
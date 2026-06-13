import pandas as pd
import numpy as np

from src.data.feature_engineering import FeatureEngineer

RAW_PATH = r"data\\raw\\xauusd_1h_2020_2025.csv"
OUT_PATH = r"data\\processed\\xau_2020_2025_features.csv"

HORIZON = 8
VOL_WINDOW = 24
K = 1.25


def add_past_returns(df):

    df["past_ret_1h"] = df["close"].pct_change(1)
    df["past_ret_2h"] = df["close"].pct_change(2)
    df["past_ret_4h"] = df["close"].pct_change(4)
    df["past_ret_8h"] = df["close"].pct_change(8)
    df["past_ret_16h"] = df["close"].pct_change(16)

    return df


def main():

    df = pd.read_csv(RAW_PATH)

    df.columns = df.columns.str.lower()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")

    df = add_past_returns(df)

    fe = FeatureEngineer()

    df = fe.add_volatility_regime_labels(
        df,
        horizon=HORIZON,
        vol_window=VOL_WINDOW,
        k=K
    )

    df, feature_cols = fe.create_features(df)

    df = df.replace([np.inf, -np.inf], np.nan)

    keep_cols = (
        ["date", "forward_regime"] +
        feature_cols
    )

    df = df[keep_cols]

    df = df.dropna().reset_index(drop=True)

    df.to_csv(OUT_PATH, index=False)

    print("\nSaved:")
    print(OUT_PATH)

    print("\nRows:")
    print(len(df))

    print("\nFeatures:")
    print(len(feature_cols))

    print("\nFeature names:")
    print(feature_cols)

    print("\nClass distribution:")
    print(df["forward_regime"].value_counts())


if __name__ == "__main__":
    main()
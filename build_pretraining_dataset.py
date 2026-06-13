""""
build the pretraining dataset from Silver + DXY only inside the XAU train window. That keeps the final XAU test period untouched.
Important: this code creates labels using 0.6% forward return threshold, matching existing 0.6 percent prediction.
"""

import os
import numpy as np
import pandas as pd

from src.data.feature_engineering import FeatureEngineer


fe = FeatureEngineer()


XAU_PATH = r"C:\\Users\\user\\OneDrive\\Desktop\\financial_regime_clasification\\data\\processed\\xau_with_all_macro.csv"
MARKET_DIR = r"C:\\Users\\user\\OneDrive\\Desktop\\financial_regime_clasification\\data\\pretrain\\correlated_markets"
OUT_DIR = r"C:\\Users\\user\\OneDrive\\Desktop\\financial_regime_clasification\\data\\pretrain"

MARKETS = {
    "silver_futures": "silver_futures.csv",
    # "dxy": "dxy.csv", - dropped because of range imbalanced the whole dataset - ver 1
}

FEATURES = [
    "open", 'high', 'low', 'close', 'volume',
    'past_ret_1h', 'past_ret_2h', 'past_ret_4h',
    "past_ret_8h", 'past_ret_16h',
]

def add_features(df):
    df = df.copy()
    df= df.sort_values("date").reset_index(drop=True)

    df['past_ret_1h'] = df['close'].pct_change(1)
    df['past_ret_2h'] = df['close'].pct_change(2)
    df['past_ret_4h'] = df['close'].pct_change(4)
    df['past_ret_8h'] = df['close'].pct_change(8)
    df['past_ret_16h'] = df['close'].pct_change(16)

    return df

def add_regime_labels(df, threshold=0.003, horizon=8):
    df = df.copy()

    future_return = df['close'].shift(-horizon) / df['close'] - 1

    df['forward_regime'] = 'range'
    df.loc[future_return > threshold, 'forward_regime'] = 'upward'
    df.loc[future_return < -threshold, 'forward_regime'] = 'downward'

    return df

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    xau = pd.read_csv(XAU_PATH)
    xau['date'] = pd.to_datetime(xau['date'])
    xau = xau.sort_values('date').reset_index(drop=True)

    split_idx = int(len(xau) * 0.8)
    train_end_date = xau['date'].iloc[split_idx - 1]

    print(f"XAU train end date: {train_end_date}")

    X_all = []
    y_all = []

    for market_name, file_name in MARKETS.items():
        path = os.path.join(MARKET_DIR, file_name)

        df = pd.read_csv(path)

        df.columns = df.columns.str.strip().str.lower()

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        df = df[df['date'] <= train_end_date].copy()

        df= add_features(df)
        
        # ver 2
        #df = add_regime_labels(df, threshold=0.003, horizon=8)

        df = fe.add_volatility_regime_labels(
            df, horizon=8, vol_window=24, k=1.0)

        df = df.replace([np.inf, -np.inf], np.nan)

        print(df.columns.tolist())

        df = df.dropna(subset=FEATURES + ['forward_regime'])

        X = df[FEATURES].values
        y = df['forward_regime'].values

        X_all.append(X)
        y_all.append(y)

        print(f"\n{market_name}")
        print(f"Rows used: {len(df)}")
        print(df['forward_regime'].value_counts())

    X_pretrain = np.vstack(X_all)
    y_pretrain = np.concatenate(y_all)

    np.save(os.path.join(OUT_DIR, "X_pretrain.npy"), X_pretrain)
    np.save(os.path.join(OUT_DIR, "y_pretrain.npy"), y_pretrain)

    print("\nSAved:")
    print(os.path.join(OUT_DIR, "X_pretrain.npy"))
    print(os.path.join(OUT_DIR, "y_pretrain.npy"))

    print("\nFinal pretrainign shape:")
    print("X:", X_pretrain.shape)
    print("y:", y_pretrain.shape)

    print('\nClass distribution:')
    print(pd.Series(y_pretrain).value_counts())

if __name__ == "__main__":
    main()
import os 
import pandas as pd
import numpy as np

from download_correlated_markets import XAU_PATH
MARKET_DIR = r"C:\\Users\\user\\OneDrive\\Desktop\\financial_regime_clasification\\data\\pretrain\\correlated_markets"
OUT_PATH = r"C:\\Users\\user\\OneDrive\\Desktop\\financial_regime_clasification\\data\\pretrain\\correlated_markets\\correlation_summary.csv"


MARKET_FILES = {
    "silver_futures": "silver_futures.csv",
    "dxy": "dxy.csv",
    "gld": "gld.csv",
    "gdx": "gdx.csv",
    "vix": "vix.csv",
}

def load_market(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates('date')
    return df

def add_returns(df, prefix):
    df[f"{prefix}_ret_1h"]  = df['close'].pct_change()
    df[f"{prefix}_ret_2h"]  = df['close'].pct_change(2)
    df[f"{prefix}_ret_4h"]  = df['close'].pct_change(4)
    df[f"{prefix}_ret_8h"]  = df['close'].pct_change(8)        
    df[f"{prefix}_ret_16h"] = df['close'].pct_change(16)
    return df

def safe_corr(a, b):
    temp = pd.concat([a, b], axis=1).dropna()

    if len(temp) < 100:
        return np.nan
    
    return temp.iloc[:, 0].corr(temp.iloc[:, 1])

def main():
    xau = load_market(XAU_PATH)
    xau = xau[['date', 'close']].rename(columns={'close': "xau_close"})
    xau['xau_ret_1h'] = xau['xau_close'].pct_change()

    results = []

    for market_name, file_name in MARKET_FILES.items():
        path = os.path.join(MARKET_DIR, file_name)

        if not os.path.exists(path):
            print(f"missing file: {path}")
            continue

        print(f"\nAnalyzing {market_name}")

        mkt = load_market(path)
        mkt = mkt[['date', 'close', 'volume']].copy()
        mkt = mkt.rename(
            columns = {
                "close": f"{market_name}_close",
                "volume": f"{market_name}_volume",
            }
        )

        mkt[f'{market_name}_ret_1h'] = mkt[f'{market_name}_close'].pct_change()

        merged = pd.merge_asof(
            xau.sort_values("date"),
            mkt.sort_values("date"),
            on="date",
            direction="nearest",
            tolerance=pd.Timedelta("1H")
            
            )
        
        #merged = pd.merge(
        #    xau,
        #    mkt,
        #    on="date",
        #    how='left',
        #)

        matched_rows = merged[f"{market_name}_close"].notna().sum()
        coverage = matched_rows / len(merged)

        corr_same_time = safe_corr(
            merged["xau_ret_1h"],
            merged[f"{market_name}_ret_1h"]
        )

        lag_corrs = {}

        for lag in [1,2,4,8,16,24]:
            lagged_market_ret = merged[f'{market_name}_ret_1h'].shift(lag)

            lag_corrs[f'lag_{lag}h_corr'] = safe_corr(
                merged['xau_ret_1h'],
                lagged_market_ret
            )

        best_lag_col = max(
            lag_corrs,
            key = lambda k: abs(lag_corrs[k]) if not pd.isna(lag_corrs[k]) else -1
        )

        result = {
            "market": market_name,
            "matched_rows": matched_rows,
            "coverage": coverage,
            "same_time_corr": corr_same_time,
            **lag_corrs,
            "best_lag": best_lag_col,
            "best_lag_corr": lag_corrs[best_lag_col],
        }

        results.append(result)

        print(f"matched rows: {matched_rows}")
        print(f"coverage: {coverage:.2%}")
        print(f"same time corr: {corr_same_time:.4f}")
        print(f"best lag: {best_lag_col} = {lag_corrs[best_lag_col]:.4f}")

        results_df = pd.DataFrame(results)

        results_df['abs_same_time_corr'] = results_df['same_time_corr'].abs()
        results_df['abs_best_lag_corr'] = results_df['best_lag_corr'].abs()

        results_df = results_df.sort_values(
            ['abs_best_lag_corr', 'coverage'],
            ascending=False,
        )

        results_df.to_csv(OUT_PATH, index=False)

        print("\n ==== correlation summary ===\n")
        print(results_df)

        print(f"\nSAved to: {OUT_PATH}")

if __name__ == "__main__":
    main()
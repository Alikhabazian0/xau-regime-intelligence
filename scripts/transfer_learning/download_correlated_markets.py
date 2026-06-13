import os
import pandas as pd
import yfinance as yf

XAU_PATH  = r"C:\\Users\\user\\OneDrive\\Desktop\\financial_regime_clasification\\data\\processed\\xau_with_all_macro.csv"
OUT_DIR   = r"C:\\Users\\user\\OneDrive\\Desktop\\financial_regime_clasification\\data\\pretrain\\correlated_markets"

MARKETS = {
    "silver_futures": "SI=F",
    "dxy": "DX-Y.NYB",
    "gld": "GLD",
    "gdx": "GDX",
    "vix": "^VIX",
}

def clean_yfinance_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    date_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={
        date_col: "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

    keep_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    df = df[[c for c in keep_cols if c in df.columns]]

    df['date'] = pd.to_datetime(df['date'])

    if df['date'].dt.tz is not None:
        df['date'] = df['date'].dt.tz_convert(None)

    df['date'] = df['date'].dt.floor('h')

    df = df.drop_duplicates(subset='date')
    df = df.sort_values('date').reset_index(drop=True)

    return df

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    xau = pd.read_csv(XAU_PATH)
    xau['date'] = pd.to_datetime(xau['date'])

    start_date = xau['date'].min().strftime("%Y-%m-%d")
    end_date = (xau['date'].max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"XAU date range: {start_date} to {end_date}")

    summary = []

    for name, ticker in MARKETS.items():
        print(f"\nDonwloading {name}: {ticker}")

        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval='1h',
                auto_adjust=False,
                progress=False,
                prepost=False,
            )

            df = clean_yfinance_df(df)

            if df.empty:
                print(f"warning: NO date returend for {name}")
                summary.append({
                    "market": name,
                    "ticker": ticker,
                    "rows": 0,
                    "start": None,
                    "end": None,
                })

                continue
            
            out_path = os.path.join(OUT_DIR, f"{name}.csv")
            df.to_csv(out_path, index=False)

            print(f"saved: {out_path}")
            print(f"rows: {len(df)}")
            print(f"Start: {df['date'].min()}")
            print(f"end: {df['date'].max()}")

            summary.append({
                "market": name,
                "ticker": ticker,
                "rows": len(df),
                "start": str(df['date'].min()),
                'end': str(df['date'].max()),
            })

        except Exception as e:
            print(f"error downloading {name}: {e}")
            summary.append({
                "market": name,
                "ticker": ticker,
                "rows": 0,
                "start": None,
                "end": None,
                "error": str(e),
                })
            

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(OUT_DIR, "download_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f'\nDownload summary:')
    print(summary_df)
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()

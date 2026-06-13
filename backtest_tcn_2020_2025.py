import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

from src.models.tcn_model import PyTorchTCN


DATA_PATH = r"data\\processed\\xau_2020_2025_features.csv"
OUT_DIR = r"experiments\\backtest"
RAW_PRICE_PATH = r"data\\raw\\xauusd_1h_2020_2025.csv"


HORIZON = 8
INITIAL_CAPITAL = 10_000
COST_PER_TRADE = 0.0002  # 2 bps per trade


FEATURES = [
    "high_low_ratio",
    "close_open_mometum",
    "typical_price",
    "price_position",
    "returns_mean",
    "returns_std",
    "returns_max",
    "returns_min",
    "retrun_range",
    "returns_trend",
    "rsi",
    "macd",
    "macd_signal",
    "macd_histogram",
    "volatiltiy_10",
]


def max_drawdown(equity_curve):
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown.min()


def sharpe_ratio(returns, periods_per_year=252 * 24):
    if returns.std() == 0:
        return 0
    return np.sqrt(periods_per_year) * returns.mean() / returns.std()


def profit_factor(trade_returns):
    gross_profit = trade_returns[trade_returns > 0].sum()
    gross_loss = abs(trade_returns[trade_returns < 0].sum())

    if gross_loss == 0:
        return np.inf

    return gross_profit / gross_loss


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    prices = pd.read_csv(RAW_PRICE_PATH)
    prices.columns = prices.columns.str.strip().str.lower()
    prices["date"] = pd.to_datetime(prices["date"])
    
    prices = prices[["date", "close"]]
    
    df = pd.merge(
        df,
        prices,
        on="date",
        how="left",
        )
    
    df["close"] = df["close"].ffill().bfill()
    
    mapping = {
        "downward": -1,
        "range": 0,
        "upward": 1,
        }

    df["target"] = df["forward_regime"].map(mapping)

    X = df[FEATURES]
    y = df["target"]

    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

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
        verbose=True,
    )

    print("\nTraining best TCN...")
    model.fit(X_train_scaled, y_train)

    pred = model.predict(X_test_scaled)

    print("\n=== Classification Report ===\n")
    print(
        classification_report(
            y_test,
            pred,
            labels=[-1, 0, 1],
            target_names=["downward", "range", "upward"],
        )
    )

    print("\nMacro F1:", round(f1_score(y_test, pred, average="macro"), 4))

    backtest = test_df.copy()
    backtest["prediction"] = pred

    # trading signal
    # +1 long, -1 short, 0 flat
    backtest["signal"] = backtest["prediction"]

    # non-overlapping trades every HORIZON hours
    trade_rows = list(range(0, len(backtest) - HORIZON, HORIZON))

    trades = []

    for i in trade_rows:
        row = backtest.iloc[i]

        signal = row["signal"]

        if signal == 0:
            continue

        entry_price = backtest.iloc[i]["close"]
        exit_price = backtest.iloc[i + HORIZON]["close"]

        raw_return = (exit_price / entry_price) - 1

        strategy_return = signal * raw_return

        strategy_return = strategy_return - COST_PER_TRADE

        trades.append({
            "entry_date": backtest.iloc[i]["date"],
            "exit_date": backtest.iloc[i + HORIZON]["date"],
            "signal": signal,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "raw_return": raw_return,
            "strategy_return": strategy_return,
        })

    trades = pd.DataFrame(trades)

    if trades.empty:
        print("\nNo trades generated.")
        return

    trades["equity"] = INITIAL_CAPITAL * (1 + trades["strategy_return"]).cumprod()

    total_return = trades["equity"].iloc[-1] / INITIAL_CAPITAL - 1

    win_rate = (trades["strategy_return"] > 0).mean()

    avg_trade_return = trades["strategy_return"].mean()

    pf = profit_factor(trades["strategy_return"])

    mdd = max_drawdown(trades["equity"])

    trade_sharpe = sharpe_ratio(
        trades["strategy_return"],
        periods_per_year=252 * 3,  # 8h holding ≈ 3 trades/day
    )

    long_trades = (trades["signal"] == 1).sum()
    short_trades = (trades["signal"] == -1).sum()

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print(f"Initial capital      : {INITIAL_CAPITAL:,.2f}")
    print(f"Final equity         : {trades['equity'].iloc[-1]:,.2f}")
    print(f"Total return         : {total_return:.2%}")
    print(f"Number of trades     : {len(trades)}")
    print(f"Long trades          : {long_trades}")
    print(f"Short trades         : {short_trades}")
    print(f"Win rate             : {win_rate:.2%}")
    print(f"Average trade return : {avg_trade_return:.4%}")
    print(f"Profit factor        : {pf:.4f}")
    print(f"Max drawdown         : {mdd:.2%}")
    print(f"Sharpe ratio         : {trade_sharpe:.4f}")

    trades.to_csv(
        os.path.join(OUT_DIR, "tcn_backtest_trades.csv"),
        index=False,
    )

    backtest.to_csv(
        os.path.join(OUT_DIR, "tcn_backtest_predictions.csv"),
        index=False,
    )

    print("\nSaved:")
    print(os.path.join(OUT_DIR, "tcn_backtest_trades.csv"))
    print(os.path.join(OUT_DIR, "tcn_backtest_predictions.csv"))


if __name__ == "__main__":
    main()
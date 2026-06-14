# Financial Regime Classification of XAUUSD: From Macroeconomic Machine Learning to Deep Learning

## Project Overview

This repository presents a complete quantitative research project investigating financial regime classification in XAUUSD (Gold Spot) using machine learning, macroeconomic variables, feature engineering, and deep learning.

The project evolved through two major research phases:

### Phase I — Classical Machine Learning and Macroeconomic Features

Focus:

* Missing-value analysis
* Class imbalance handling
* Macroeconomic feature integration
* Tree-based machine learning models
* Hyperparameter optimization

Models:

* CatBoost
* XGBoost
* LightGBM
* Random Forest
* Extra Trees
* Logistic Regression

Dataset:

* 2,942 hourly observations

---

### Phase II — Large-Scale Deep Learning and Regime Modeling

Focus:

* Historical dataset expansion
* Volatility-adjusted regime labels
* Temporal Convolutional Networks (TCN)
* Transfer learning
* Walk-forward validation
* Trading backtesting

Models:

* Random Forest
* Temporal Convolutional Networks

Dataset:

* 35,116 hourly observations
* January 2020 – December 2025

---

# Motivation

Financial markets operate under multiple latent regimes characterized by different volatility, momentum, and risk structures.

The ability to identify these regimes is valuable for:

* Quantitative trading
* Portfolio allocation
* Risk management
* Volatility forecasting
* Financial decision-making

Gold is particularly interesting due to its sensitivity to:

* Inflation expectations
* Real interest rates
* Treasury yields
* U.S. Dollar strength
* Market uncertainty

The objective of this project is to classify future market states into:

* Downward
* Range
* Upward

using historical market information and machine learning.

---

# Dataset

## Phase I Dataset

Frequency:

* 1 Hour

Rows:

* 2,942

Features:

* OHLCV
* Lagged returns
* Macroeconomic variables

---

## Phase II Dataset

Source:

* Dukascopy XAUUSD

Raw Frequency:

* 5 Minutes

Aggregation:

* 1 Hour

Period:

* January 2020 – December 2025

Rows:

* 35,116

---

# Macroeconomic Variables

The following external financial variables were incorporated:

* VIX
* DXY
* 10-Year Treasury Yield (TNX)
* TIPS Yield
* Breakeven Inflation

These variables were selected based on established economic theory regarding gold valuation.

---

# Correlation Analysis

Several correlated markets were investigated:

| Asset          | Correlation with Gold |
| -------------- | --------------------: |
| Silver Futures |                0.6047 |
| DXY            |               -0.4104 |
| GLD            |                0.3000 |
| GDX            |                0.3008 |
| VIX            |                0.0594 |

Strongest relationships:

* Silver Futures
* DXY

These assets were later used in transfer-learning experiments.

---

# Regime Labeling

## Phase I

Original labels suffered from severe class imbalance.

Distribution:

* Range ≈ 66%
* Upward ≈ 19%
* Downward ≈ 15%

---

## Phase II

A volatility-adjusted labeling framework was developed.

Parameters:

```python
horizon = 8
vol_window = 24
k = 1.25
```

Classification:

```python
future_return > k * volatility     -> Upward
future_return < -k * volatility    -> Downward
otherwise                          -> Range
```

Final distribution:

| Regime   | Percentage |
| -------- | ---------: |
| Range    |     39.96% |
| Upward   |     32.40% |
| Downward |     27.64% |

This produced a substantially more balanced classification problem.

---

# Feature Engineering

## Price Features

* High-Low Ratio
* Close-Open Momentum
* Typical Price
* Price Position

## Return Features

Using:

* Past Return 1h
* Past Return 2h
* Past Return 4h
* Past Return 8h
* Past Return 16h

Derived:

* Returns Mean
* Returns Std
* Returns Max
* Returns Min
* Return Range
* Returns Trend

## Technical Indicators

### RSI

Window:

14 periods

### MACD

Parameters:

* EMA Fast = 12
* EMA Slow = 26
* Signal = 9

Generated:

* MACD
* MACD Signal
* MACD Histogram

### Volatility

Rolling return volatility.

Window:

10 periods

---

## Final Feature Set

Total engineered features:

```text
15
```

Features:

```text
high_low_ratio
close_open_momentum
typical_price
price_position

returns_mean
returns_std
returns_max
returns_min
return_range
returns_trend

rsi
macd
macd_signal
macd_histogram

volatility_10
```

---

# Machine Learning Models

## Logistic Regression

Result:

* Failed to detect minority regimes

---

## Extra Trees

Macro F1:

```text
0.267
```

---

## Random Forest

### Basic Features

Macro F1:

```text
0.2697
```

### Engineered Features

Macro F1:

```text
0.3246
```

Improvement:

```text
+20%
```

---

## XGBoost

Macro F1:

```text
0.334
```

---

## LightGBM

Macro F1:

```text
0.328
```

---

## CatBoost

Best Classical Model

Minority F1:

```text
0.258
```

Balanced Accuracy:

```text
0.391
```

Macroeconomic variables significantly improved performance.

---

# Hyperparameter Optimization

Optuna was applied to:

* CatBoost
* XGBoost
* LightGBM

Observation:

Aggressive optimization improved overall accuracy but often reduced minority-regime detection.

This highlighted the importance of selecting evaluation metrics carefully in imbalanced financial datasets.

---

# Deep Learning

## Temporal Convolutional Network

Architecture:

```python
sequence_length = 36

num_channels = (16, 32)

kernel_size = 2

dropout = 0.35

batch_size = 32

epochs = 38

learning_rate = 0.001966625789577707
```

Input Features:

```text
15 engineered features
```

---

## Transfer Learning

Pretraining Markets:

* Silver Futures
* DXY

Pretraining Dataset:

```text
4497 samples
```

Technique:

* TCN pretraining
* WeightedRandomSampler
* Class balancing

Result:

Transfer learning did not outperform direct XAUUSD training.

---

# Model Results

## Best Random Forest

Macro F1:

```text
0.3246
```

---

## Best TCN

Single Split:

```text
Macro F1 = 0.3462
```

Weighted F1:

```text
0.3630
```

Best classification result achieved during the project.

---

# Walk-Forward Validation

Expanding-window validation was performed.

### Fold 1

Train:

2020–2022

Test:

2023

Macro F1:

```text
0.3389
```

---

### Fold 2

Train:

2020–2023

Test:

2024

Macro F1:

```text
0.2659
```

---

### Fold 3

Train:

2020–2024

Test:

2025

Macro F1:

```text
0.3384
```

---

## Average Walk-Forward Performance

```text
Macro F1 = 0.3144
Weighted F1 = 0.3243
```

This provides the most realistic estimate of future performance.

---

# Trading Backtest

A simple directional strategy was evaluated.

Rules:

```text
Upward Prediction   -> Long
Downward Prediction -> Short
Range Prediction    -> Flat
```

Holding Period:

```text
8 Hours
```

Initial Capital:

```text
$10,000
```

Transaction Cost:

```text
0.02%
```

---

## Results

Final Equity:

```text
$7,538.86
```

Total Return:

```text
-24.61%
```

Profit Factor:

```text
0.8219
```

Sharpe Ratio:

```text
-1.97
```

Maximum Drawdown:

```text
-29.64%
```

Conclusion:

The classifier achieved moderate predictive performance but did not generate a profitable trading strategy under a simple execution framework.

---

# Repository Structure

```text
financial_regime_classification/

├── data/
│   ├── raw/
│   ├── processed/
│   └── pretrain/
│
├── src/
│   ├── models/
│   │   ├── tcn_model.py
│   │   ├── random_forest.py
│   │   └── feature_eng.py
│   │
│   ├── preprocessing/
│   └── utils/
│
├── experiments/
│   ├── pretraining/
│   ├── walkforward/
│   ├── backtest/
│   └── results/
│
├── build_xau_dataset.py
├── pretrain_tcn.py
├── transfer_tcn_xau.py
├── walkforward_tcn.py
├── backtest_tcn_2020_2025.py
│
└── README.md
```

---

# Key Findings

### Successful

* Historical data expansion improved stability.
* Macroeconomic variables improved minority-regime detection.
* Feature engineering significantly improved performance.
* TCN consistently outperformed Random Forest.
* Walk-forward validation provided realistic performance estimates.
* Volatility-adjusted labels produced balanced classification targets.

### Unsuccessful

* SMOTE destroyed temporal structure.
* Transfer learning from Silver and DXY was ineffective.
* Ensemble voting produced limited gains.
* Classification accuracy did not translate directly into profitable trading.

---

# Technologies

* Python
* Pandas
* NumPy
* Scikit-Learn
* CatBoost
* XGBoost
* LightGBM
* PyTorch
* Optuna
* Matplotlib
* Seaborn

---

# Future Work

Potential extensions include:

* Probability calibration
* Meta-labeling
* Confidence-based signal filtering
* Transformer architectures
* Attention-based temporal models
* Portfolio-level regime allocation
* Multi-asset feature integration
* Reinforcement learning for execution

---

# Final Results

| Model                               |              Metric |
| ----------------------------------- | ------------------: |
| CatBoost + Macro Features           | Minority F1 = 0.258 |
| Random Forest + Engineered Features |   Macro F1 = 0.3246 |
| TCN (Single Split)                  |   Macro F1 = 0.3462 |
| TCN (Walk-Forward Average)          |   Macro F1 = 0.3144 |

This project demonstrates a complete quantitative research workflow spanning data engineering, macroeconomic modeling, machine learning, deep learning, walk-forward validation, and trading-system evaluation for financial regime classification.

# PS Note

The processed feature dataset used for the final TCN experiments is included.
All other datasets can be regenerated using the provided build scripts.

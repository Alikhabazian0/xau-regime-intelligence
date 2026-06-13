import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Create feature for financial regime classification
    optimized for 3000 rows (creates ~20-25 features)
    """

    def __init__(self):
        self.feature_cols = []

    def create_features(self, df):
        """main eature engineering pipeline"""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)

        # start with base feature (excluding target and prediction columns)
        exclude_cols = ['date', 'forward regime', 'target_encoded', 
                        'probability', '0.6 percent prediction',
                        '1 percent prediction', '1.5 percent prediction']
        
        base_features = [col for col in df.columns if col not in exclude_cols]
        print(f"Base features: {len(base_features)}")

        # 1. price-derived features
        df = self._add_price_features(df)

        # 2. volume features
        df = self._add_return_aggregates(df)

        # 3. return aggregates
        df = self._add_return_aggregates(df)

        # 4. technical indicators
        df = self._add_technical_indicators(df)

        # 5. macro features
        df = self._add_macro_features(df)

        # 6. define final feature list
        self.feature_cols = [col for col in df.columns
                             if col not in exclude_cols + base_features]
        
        # handle any remainging missing values
        df[self.feature_cols] = df[self.feature_cols].fillna(df[self.feature_cols].median())

        df[self.feature_cols] = df[self.feature_cols].fillna(0)

        print(f"\nFinal features: {len(self.feature_cols)}")
        print(f"Total feature matrix shape: {df.shape}")   

        return df, self.feature_cols
    
    def _add_price_features(self, df):
        """Basic price-derived features"""
        # range as percentage of price
        df['high_low_ratio'] = (df['high'] - df['low'] / df['close'] + 1e-8)

        # momentum (close vs open)
        df['close_open_mometum'] = (df['close'] - df['open']) / df['open']

        # typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        # price position in candle
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

        return df
    
    def _add_volume_features(self, df):
        """volume-based features"""
        # volume change
        df['volume_change'] = df['volume'].pct_change().fillna(0)

        # volume moving average ratio
        df['volume_ma_ratio'] = df['volume'] / (df['volume'].rolling(10, min_periods=1).mean())

        # price-volume correlation (simplified)
        df['price_volume_corr'] = df['close'].pct_change() * df['volume'].pct_change()

        return df
    
    def _add_return_aggregates(self, df):
        """Aggregate past retruns"""
        ret_cols = [col for col in df.columns if 'past_ret' in col]

        if ret_cols:
            df['returns_mean']  = df[ret_cols].mean(axis=1)
            df['returns_std']   = df[ret_cols].std(axis=1)
            df['returns_max']   = df[ret_cols].max(axis=1)
            df['returns_min']  = df[ret_cols].min(axis=1)
            df['retrun_range'] = df['returns_max'] - df['returns_min']

            # Trend: Are recent return diffrent from older returns?
            if len(ret_cols) >= 2:
                recent = df[ret_cols[:2]].mean(axis=1) # 1h, 2h
                older = df[ret_cols[2:]].mean(axis=1)  # 4h, 8h, 16h
                df['returns_trend'] = recent - older

        return df
        
    def _add_technical_indicators(self, df):
        """Basic technical indicators"""

        # RSI
        delta = df['close'].diff()
        gain  = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss  = (-delta.where(delta < 0 , 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']


        # volatility (rolling std of returns)
        df['volatiltiy_10'] = df['close'].pct_change().rolling(10, min_periods=1).std()

        return df
        
    def _add_macro_features(self, df):
        """Add macro features if they exist"""
        if 'vix' in df.columns:
            # VIX features
            df['vix_change'] = df['vix'].pct_change().fillna(0)
            df['vix_ma_5'] = df['vix'].rolling(5, min_periods=1).mean()
        
        if 'dxy' in df.columns:
            # DXY features (gold inverse correlation)
            df['dxy_change'] = df['dxy'].pct_change().fillna(0)
            df['gold_dxy_ratio'] = df['close'] / (df['dxy'] + 1e-8)
        
        return df
    
    def add_volatility_regime_labels(
            self,
            df,
            horizon=8,
            vol_window=24,
            k=1.0
            ):
        df = df.copy()
        
        returns = df["close"].pct_change()
        rolling_vol = returns.rolling(vol_window).std()
        future_return = df["close"].shift(-horizon) / df["close"] - 1
        
        df["forward_regime"] = "range"
        df.loc[future_return > k * rolling_vol, "forward_regime"] = "upward"
        df.loc[future_return < -k * rolling_vol, "forward_regime"] = "downward"
        
        return df
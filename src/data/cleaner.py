import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

class DataCleaner:
    """
    Handle missing data appropriately for financial time series
    Based on our analysis: 31.8% missing in prediction columns
    """

    def __init__(self, missing_strategy='median'):
        """
        Parameters:
        - missing_strategy: 'median', 'zero', 'forward'
        """
        self.missing_strategy = missing_strategy
        self.imputer = None

    def clean(self, df):
        """Main cleaning pipeline"""
        print("\n" + "="*60)
        print("DATA CLEANING")
        print("="*60)

        # step1: validate target column
        df = self._validate_target(df)

        # step2: handle missing prediction columns
        df = self._handle_missing_predictions(df)

        # step3: encode target variable
        df = self._encode_target(df)

        # step4: remove any infinite values
        df = df.replace([np.inf, -np.inf], np.nan)

        print(f"Cleaning complete. final shape: {df.shape}")
        return df
    
    def _validate_target(self, df, target_col='forward regime'):
        """Ensure target has no missing values"""
        if df[target_col].isnull().any():
            missing = df[target_col].isnull().sum()
            print(f"dropping {missing} rows with missing target")
            df = df.dropna(subset=[target_col])
        else:
            print(f"Target column '{target_col}' has no missing values")
            return df
        
    def _handle_missing_predictions(self, df):
        """Handle missing values in prediction columns"""
        pred_cols = ['probability', '0.6 percent prediction', '1 percent prediction', '1.5 percent prediction']

        # Filter to existing columns
        existing_cols = [col for col in pred_cols if col in df.columns]

        if not existing_cols:
            print("No prediction columns found to clean")
            return df
        
        missing_before = df[existing_cols].isnull().sum().sum()
        print(f"\nMissing values in prediction columns: {missing_before}")

        if self.missing_strategy == 'median':
            # use median imputation (preseves distribution)
            self.imputer = SimpleImputer(strategy='median')
            df[existing_cols] = self.imputer.fit_transform(df[existing_cols])
            print(f"Filled with column medians")

        elif self.missing_strategy == 'zero':
            # fill with 0 (missing = no signal)
            for col in existing_cols:
                if 'probabilty' in col:
                    df[col] = df[col].fillna(0.5)
                else:
                    df[col] = df[col].fillna(0)
            print(f"filled with zeros")

        elif self.missing_strategy == "forward":
            # forward fill time series
            df = df.sort_values('date')
            df[existing_cols] = df[existing_cols].fillna(method='ffill').fillna(method='bfill')
            print(f"filled using forward fill")

        missing_after = df[existing_cols].isnull().sum().sum()
        print(f"missing after cleaning: {missing_after}")

        return df
    
    def _encode_target(self, df, target_col='forward regime'):
        """covert string labels to integers"""
        regime_mapping = {
            'downward': 0,
            'range': 1,
            'upward': 2
        }

        df['target_encoded'] = df[target_col].map(regime_mapping)

        # veify mapping worked
        if df['target_encoded'].isnull().any():
            unknown = df[df['target_encoded'].isnull()][target_col].unique()
            raise ValueError(f"Unknown regime value: {unknown}")
        
        print(f"\nTarget encoding:")
        for regime, code in regime_mapping.items():
            count = (df['target_encoded'] == code).sum()
            print(f" {regime:10} --> {code}: {count} samples ({count/len(df)*100:.1f}%)")

        return df
        

import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    """Handle data loading and basic validation"""

    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = None

    def load(self):
        """Load CSV file"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        print(f'Loading data frmo {self.data_path}...')
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} rows with {len(self.df.columns)} columns")

        return self.df
    
    def validate(self):
        """Basic data validation"""
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'forward regime']
        missing_cols = [col for col in required_cols if col not in self.df.columns]

        if missing_cols:
            raise ValueError(f"missing required columns: {missing_cols}")
        
        # ocnvert date to datetime
        self.df['date'] = pd.to_datetime(self.df['date'])

        # check fpr duplicates
        if self.df.duplicated(subset=['date']).any():
            print(f"warning: found {self.df.duplicated(subset=['date']).sum()} duplicate dates")
            self.df = self.df.drop_duplicates(subset=['date'])

        print("Data Validation passed")
        return self.df
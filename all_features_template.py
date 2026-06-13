"""
Add Macro Economic Features to XAUUSD Data
Features: TNX (10Y Yield), TIPS (Real Rate), Breakeven Inflation
"""

import pandas as pd
import numpy as np
import pandas_datareader as pdr
from datetime import datetime
import os

print("="*80)
print("ADDING MACRO FEATURES TO XAUUSD DATA")
print("="*80)

# 1. Load your existing enriched data
print("\n1. Loading existing data...")
df = pd.read_csv(r"data\processed\xau_with_vix_dxy.csv", parse_dates=['date'])
df.set_index('date', inplace=True)

print(f"Original data: {len(df)} rows")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# 2. Get date range
start_date = df.index.min()
end_date = df.index.max()

print(f"\n2. Downloading macro data from {start_date.date()} to {end_date.date()}...")

# 3. Download macro features
macro_data = pd.DataFrame(index=df.index)

try:
    # TNX: 10-Year Treasury Yield (inverse correlation with gold)
    print(" Downloading TNX (10-Year Treasury Yield)...")
    tnx = pdr.DataReader('DGS10', 'fred', start_date, end_date)
    macro_data['tnx'] = tnx['DGS10'] / 100  # Convert to decimal (e.g., 4.2% -> 0.042)
    print(f" Got {len(tnx)} rows")
    
    # TIPS: 10-Year Real Interest Rate (gold loves negative real rates)
    print(" Downloading TIPS Real Rate (DFII10)...")
    tips = pdr.DataReader('DFII10', 'fred', start_date, end_date)
    macro_data['real_rate'] = tips['DFII10'] / 100  # Convert to decimal
    print(f" Got {len(tips)} rows")
    
    # Breakeven Inflation (TNX - TIPS = market inflation expectation)
    print("   Calculating breakeven inflation...")
    macro_data['breakeven_inflation'] = macro_data['tnx'] - macro_data['real_rate']
    print(" Calculated")
    
except Exception as e:
    print(f" Download error: {e}")
    print("   Trying alternative data sources...")
    
    # Alternative: Use different FRED series if main ones fail
    try:
        tnx = pdr.DataReader('DGS10', 'fred', start_date, end_date)
        macro_data['tnx'] = tnx['DGS10'] / 100
    except:
        print(" TNX unavailable, using forward fill from existing")
        macro_data['tnx'] = np.nan
    
    try:
        # Alternative TIPS series
        tips = pdr.DataReader('FII10', 'fred', start_date, end_date)
        macro_data['real_rate'] = tips['FII10'] / 100
    except:
        print(" TIPS unavailable")
        macro_data['real_rate'] = np.nan

# 4. Add derived features
print("\n3. Creating derived features...")

# Gold vs Yield inverse relationship (strong signal)
macro_data['gold_yield_spread'] = macro_data['tnx'] - macro_data['real_rate']

# Rate of change (momentum signals)
macro_data['tnx_change_5d'] = macro_data['tnx'].diff(5)
macro_data['real_rate_change_5d'] = macro_data['real_rate'].diff(5)

# Negative real rate indicator (binary: 1 if real rate < 0)
macro_data['negative_real_rate'] = (macro_data['real_rate'] < 0).astype(int)

print(" Created derived features")

# 5. Merge with your data
print("\n4. Merging with XAUUSD data...")
combined_df = df.copy()
macro_features = ['tnx', 'real_rate', 'breakeven_inflation', 
                  'gold_yield_spread', 'tnx_change_5d', 
                  'real_rate_change_5d', 'negative_real_rate']

for feature in macro_features:
    if feature in macro_data.columns:
        combined_df[feature] = macro_data[feature]
    else:
        combined_df[feature] = np.nan

# 6. Handle missing values (weekends/holidays)
print("\n5. Handling missing values...")
before_missing = combined_df[macro_features].isnull().sum().sum()
combined_df[macro_features] = combined_df[macro_features].fillna(method='ffill')
combined_df[macro_features] = combined_df[macro_features].fillna(method='bfill')
combined_df[macro_features] = combined_df[macro_features].fillna(0)
after_missing = combined_df[macro_features].isnull().sum().sum()

print(f"   Missing values filled: {before_missing} → {after_missing}")

# 7. Save to new file
print("\n6. Saving enriched data...")
output_path = r"data\processed\xau_with_all_macro.csv"
combined_df.to_csv(output_path)
print(f" Saved to: {output_path}")

# 8. Summary statistics
print("\n" + "="*80)
print("FEATURE SUMMARY")
print("="*80)

print(f"\nNew macro features added:")
for feature in macro_features:
    if feature in combined_df.columns:
        non_null = combined_df[feature].notnull().sum()
        print(f"  • {feature}: {non_null}/{len(combined_df)} rows (mean={combined_df[feature].mean():.4f})")

print(f"\nFinal dataset shape: {combined_df.shape}")
print(f"Columns: {combined_df.columns.tolist()}")

print("\n" + "="*80)
print(" MACRO FEATURES ADDED SUCCESSFULLY!")
print("="*80)

# 9. Quick correlation check with gold
print("\n7. Correlation with Gold Close Price:")
correlations = combined_df[macro_features + ['close']].corr()['close'].sort_values(ascending=False)
print(correlations.to_string())

# Verify key relationships (should be negative for yields)
tnx_corr = combined_df['close'].corr(combined_df['tnx'])
print(f"\n Gold vs TNX correlation: {tnx_corr:.3f} (expected negative)")
if tnx_corr < 0:
    print("Correct inverse relationship!")
else:
    print("Unexpected positive correlation - check data")
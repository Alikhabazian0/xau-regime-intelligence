import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

def compute_class_weights(y):
    """
    Compute balanced class weights for imbalanced data
    Based our premiraliy analysis 4.37:1 imbalance ratio
    """
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    weight_dict = dict(zip(classes, weights))

    print(f"\nClass weights computed:")
    for class_label, weight in weight_dict.items():
        print(f" Class {class_label}: {weight:.4f}")

    return weight_dict

def create_temporal_split(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Create time-based split (no random shuffle for time series
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]

    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]

    print(f"\nTemporal split")
    print(f" Train : {len(X_train)} samples ({train_ratio*100:.0f}%)")
    print(f" Val: {len(X_val)} samples ({val_ratio*100:.0f}%)")
    print(f" Test: {len(X_test)} samples ({100 - (train_ratio + val_ratio)*100:.0f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test

def analyze_results(all_results):
    """
    Compare all models and find the best
    """
    print("\n"+"="*60)
    print("FINAL MODEL COMPARISON")
    print("="*60)

    # create comparison dataframe
    comparison = []
    for model_name, results in all_results.items():
        metrics = results['evaluator'].get_best_model_metric()
        comparison.append({
            'Model': model_name,
            "Avg Minority F1": metrics['avg_minority_f1'],
            'Balanced Accuracy': metrics['balanced_accuracy'],
            'MCC': metrics['mcc']
        })

    df_comparison = pd.DataFrame(comparison)
    df_comparison = df_comparison.sort_values('Avg Minority F1', ascending=False)

    print("\nModel Ranking (by Average Minority Class F1):")
    print(df_comparison.to_string(index=False))

    # find best model
    best_model = df_comparison.iloc[0]['Model']
    best_score = df_comparison.iloc[0]['Avg Minority F1']

    print(f"\n{'='*60}")
    print(f" Best Model: {best_model}")
    print(f" Average Minority F1: {best_score:.4f}")
    print(f" (Focuses on catching downward/upward movements)")
    print(f"{'='*60}")
    
    return df_comparison, best_model
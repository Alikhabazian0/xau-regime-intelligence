"""
Run hyperparameter optimization 2 for all models
finds the best configuration for each model
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# import your modules
from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.data.feature_engineering import FeatureEngineer
from src.training.optimizer_2 import HyperparameterOptimizer_2

def main():
    print("="*80)
    print("HYPERPARAMETER OPTIMIZATION FOR ALL MODELS")
    print('Finding best configurations for RandomForest, CatBoost, etc')
    print('='*80)

    # 1. load data
    print("\n loadiing data...")
    loader = DataLoader(r"C:\\Users\\user\\OneDrive\Desktop\\financial_regime_clasification\\data\\processed\\xau_with_all_macro.csv")
    df = loader.load()

    # 2. clean and prepare
    cleaner = DataCleaner(missing_strategy='median')
    df = cleaner.clean(df)

    # 3. feaure engineering
    engineer = FeatureEngineer()
    df, feature_cols = engineer.create_features(df)

    # 4. prepare X, y
    X = df[feature_cols].values
    y = df['target_encoded'].values

    # 5. Train-Test split
    split_idx = int(len(X)*0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 6. scale
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"\n training data: {X_train.shape}")
    print(f"class distribution: {np.bincount(y_train)}")

    # 7. compute class weights
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    weight_dict = dict(zip(classes, class_weights))

    # 8. define models to optimize
    models_to_optimize = {
        'RandomForest': RandomForestClassifier,
        'ExtraTrees': ExtraTreesClassifier,
        'CatBoost': CatBoostClassifier,
        'LightGBM': lgb.LGBMClassifier,
        'XGBoost': XGBClassifier
    }

    # fixed parameters for each model
    fixed_params = {
        'RandomForest': {
            'class_weight': weight_dict,
            'random_state': 42,
            'n_jobs': -1
        },
        'ExtraTrees': {
            'class_weight': weight_dict,
            'random_state': 42,
            'n_jobs': -1
        },
        'CatBoost': {
            'class_weights': weight_dict,
            'random_state': 42,
            'verbose': 0
        },
        'LightGBM': {
            'is_unbalance': True,
            'random_state': 42,
            'verbose': -1
        },
        'XGBoost': {
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'mlogloss'
        }
    }

    # 9. run optimization
    print("\n" + "="*80)
    print(' STARTING OPTIMIZATION')
    print("="*80)

    optimizer = HyperparameterOptimizer_2(
        n_trials=30,    # number of parameter combination to try
        cv_folds=5,     # 5-fold cross-validation
        random_state=42
    )
    optimized_results = optimizer.optimize_all(
        models_to_optimize,
        X_train, y_train,
        fixed_params
    )

    # 10. evaluate all optimized models
    print("\n" + "="*80)
    print("EVALUATION OPTIMIZED MODELS")
    print("="*80)

    from sklearn.metrics import classification_report, f1_score

    results_summary = []

    for model_name, result in optimized_results.items():
        model = result['model']
        best_params = result['best_params']

        # predict
        y_pred = model.predict(X_test)

        # calculate metrics
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        f1_per_class = f1_score(y_test, y_pred, average=None)
        minority_f1 = (f1_per_class[0] + f1_per_class[2]) / 2

        print(f"\n{'='*50}")
        print(f"{model_name}")
        print(f"{"="*50}")
        print(f"best parameters: {best_params}")
        print(f'\nClassification report:')
        print(classification_report(y_test, y_pred, target_names=['downward', 'range', 'upward']))
        print(f'\nMacro f1: {macro_f1:.4f}')
        print(f"minority f1 (downward+upward): {minority_f1:.4f}")

        results_summary.append({
            'Model': model_name,
            'Macro F1': macro_f1,
            'Minority F1': minority_f1,
            'Best Params': str(best_params)
        })

    # 11. compare all optimized models
    print("\n" + "="*80)
    print('OPTIMIZATION RESULTS COMPARISON')
    print("="*80)

    results_df = pd.DataFrame(results_summary)
    results_df = results_df.sort_values('Macro F1', ascending=False)

    # 12. save results
    import os
    os.makedirs("experiments/optimization", exist_ok=True)
    results_df.to_csv("experiments/optimization/optimization_results.csv", index=False)

    # save best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = optimized_results[best_model_name]['model']

    import joblib
    joblib.dump(best_model, f"experiments/optimization/{best_model_name}_optimized.pkl")

    print(f'\n best model: {best_model_name}')
    print(f'\ results saved to experiments/optimization/')
    print(f'optimized model saved to experiments/optimization/{best_model_name}_optimized.pkl')

    return optimized_results

if __name__ == "__main__":
    results = main()
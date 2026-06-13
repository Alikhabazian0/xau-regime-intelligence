"""
Financial Regime Classification Pipeline
Main orchestrator for the entire project

Strategy: Train all classical ML models -> Fine-tune each -> compare best versions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.data.feature_engineering import FeatureEngineer
from src.models.model_factory import ModelFactory
from src.training.optimizer import HyperparameterOptimizer
from src.training.evaluator import ModelEvaluator
from src.utils.helpers import compute_class_weights, create_temporal_split, analyze_results
from src.utils.visualization import plot_results_comparison

def main():
    print("="*80)
    print("FINANCIAL REGIME CLASSIFICATION PIPELINE")
    print("Strategy: Optimize each model individually, them compare")
    print("="*80)

    # ======== CONFIGURATION ===========
    DATA_PATH = "data\\processed\\xau_with_vix_dxy.csv"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    OPTIMIZATION_TRIALS = 30

    # ====== 1. LOAD DATA ===============
    print("\n" + "="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)

    loader = DataLoader(DATA_PATH)
    df = loader.load()
    df = loader.validate()

    # ====== 2. CLEAN DATA ================
    print("\n" + "="*60)
    print("STEP 2: CLEANING DATA")
    print("="*60)

    cleaner = DataCleaner(missing_strategy='median') # best choice based on our imbalanced dataset
    df = cleaner.clean(df)

    # ======= 3. FEATURE ENGINEERING =======
    print("\n" + "="*60)
    print("STEP 3: FEATURE ENGINEERING")
    print("="*60)

    engineer = FeatureEngineer()
    df, feature_cols = engineer.create_features(df)

    # ====== 4. PREPARE FEATURES AND TARGET ====
    print("\n" + "="*60)
    print("STEP 4: PREPARING DATA FOR MODELING")
    print("="*60)

    X = df[feature_cols].values
    y = df['target_encoded'].values

    print(f"\nFinal feature matrix: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")

    # ======= 5. TRAIN-TEST SPLIT (TEMPORAL) ======
    # use temporal split for time series data
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\nTemporal split:")
    print(f" Trainig: {len(X_train)} samples")
    print(f" Testing: {len(X_test)} samples")

    # ======= 6. SCALE FEATURE ====================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.fit_transform(X_test)

    # ======= 7. HANDLE IMBALANCE =================
    class_weights = compute_class_weights(y_train)

    # ====== 8. GET ALL MODELS ====================
    base_models = ModelFactory.get_all_models(class_weights)
    print(f"\nModels to evaluate: {list(base_models.keys())}")

    # ====== 9. OPTIMIZE EACH MODEL ===============
    print("\n" + "="*60)
    print("STEP 5: HYPERPARAMETER OPTIMIZATION")
    print("="*60)

    optimizer = HyperparameterOptimizer(n_trials=OPTIMIZATION_TRIALS, cv_folds=5)
    optimized_models = {}

    for model_name, base_model in base_models.items():
        print(f"\n{'='*50}")
        print(f"optimizing {model_name}...")
        print(f"{'='}*50")

        try:
            best_model, best_params = optimizer.optimize(
                model_name, base_model, X_train_scaled, y_train
            )
            optimized_models[model_name] = best_model

            # save best params for refrence
            print(f" {model_name} optimized successfully")

        except Exception as e:
            print(f" Optimization Failed for {model_name}: {str(e)}")
            print(f"falling back to default {model_name}")
            # fall back to default model
            base_model.fit(X_train_scaled, y_train)
            optimized_models[model_name] = base_model

    # ======== 10. EVALUATE ALL MODELS =================
    print("\n" +"="*60)
    print("STEP 6: MODEL EVALUATION")
    print("="*60)

    all_results = {}
    class_names = ['downward', 'range', 'upward']

    for model_name, model in optimized_models.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}...")
        print(f"{'='*50}")

        # predict
        y_pred = model.predict(X_test_scaled)

        # get probabilities if available
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test_scaled)

        # Evaluate
        evaluator = ModelEvaluator(model_name, class_names)
        evaluator.evaluate(y_test, y_pred, y_proba)

        # plot confusion matrix
        plot_path = f"experiments/plots/{model_name}_confusion.png"
        evaluator.plot_confuion_matrix(save_path=plot_path)

        # store resutls
        all_results[model_name] = {
            'model': model,
            'evaluator': evaluator,
            'predictions': y_pred,
            'probabilities': y_proba
        }

    # ======== 11. COMPARE AND FIND BEST ============
    print("\n" + "="*60)
    print("STEP 7: MODEL COMPARISON")
    print("="*60)

    # Analyze results
    comparison_df, best_model_name = analyze_results(all_results)

    # visual comparison
    plot_results_comparison(all_results, save_path="experiments\\plots\\model_comparison.png")

    # ========= 12. FINAL SUMMARY ====================
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)

    print(f"\n Total models evaluated: {len(optimized_models)}")
    print(f"Best model: {best_model_name}")
    print(f"Results saved to: experiments/")

    # save results to CSV
    comparison_df.to_csv("experiments\\results\\model_comparison.csv", index=False)
    print(f" Comparison table saved to experiments/results/model_comparison.csv")

    # print recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    best_model = all_results[best_model_name]['model']
    best_metrics = all_results[best_model_name]['evaluator'].get_best_model_metric()

    print(f"\nUse {best_model_name} for production:")
    print(f" - Average minority class F1: {best_metrics['avg_minority_f1']:.3f}")
    print(f" - Balanced accuracy: {best_metrics['balanced_accuracy']:.3f}")

    if best_metrics['avg_minority_f1'] < 0.6:
        print("\n Warining: Minority class performance is low (<0.6)")
        print("  Consider:")
        print("  1. Collecting more data dor upward/downward regimes")
        print("  2. Trying ensemble of top 3 models")
        print("  3. Implementing transfer learning (next phase)")

    print("\n" + "="*80)
    print(" Pipeline execution completed successfully!")
    print("="*80)

    return all_results, best_model_name

if __name__ == "__main__":
    # crete necessary directories
    Path("experiment\\plots").mkdir(parents=True, exist_ok=True)
    Path("experimnets\\results").mkdir(parents=True, exist_ok=True)
    Path("data\\processed").mkdir(parents=True, exist_ok=True)

    # run pipeline
    results, best_model = main()
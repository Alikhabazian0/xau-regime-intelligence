"""
Systematic experimentation for financial regime classification.
Tests techniques sequentially to measure ture impact.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, classification_report
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# import existing modules
from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.data.feature_engineering import FeatureEngineer


class ExperimentRunner:
    """Run sequential experiments to measure improvements"""

    def __init__(self):
        self.results = {}
        self.X_train = None
        self.X_test  = None
        self.y_train = None
        self.y_test  = None
        self.class_names = ['downward', 'range', 'upward']

    def load_data(self):
        """Load and prepare data (reuse your pipeline)"""
        print("="*80)
        print("LOADING DATA")
        print("="*80)
    
        loader = DataLoader(r"data\raw\raw.csv")
        df = loader.load()
        df = loader.validate()
    
        cleaner = DataCleaner(missing_strategy='median')
        df = cleaner.clean(df)
    
        engineer = FeatureEngineer()
        df, feature_cols = engineer.create_features(df)  # Make sure method name matches
    
        print(f"df shape after features: {df.shape}")
        print(f"feature_cols: {feature_cols[:5]}...")  # Show first 5
    
        # Explicitly get X and y
        X = df[feature_cols].values
        y = df['target_encoded'].values
    
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"y unique: {np.unique(y)}")
    
        # Temporal split
        split_idx = int(len(X) * 0.8)
        self.X_train = X[:split_idx]
        self.X_test = X[split_idx:]
        self.y_train = y[:split_idx]
        self.y_test = y[split_idx:]
    
        print(f"After split - X_train: {self.X_train.shape}, y_train: {self.y_train.shape}")
        print(f"self.y_train is None? {self.y_train is None}")
    
        # Scale
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        print(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        print(f"Train class distribution: {np.bincount(self.y_train)}")    
    
    def evaluate_model(self, model, name):
        """Evaluate model and return metrics"""
        y_pred = model.predict(self.X_test)

        macro_f1 = f1_score(self.y_test, y_pred, average='macro')
        f1_per_class = f1_score(self.y_test, y_pred, average=None)
        recall_per_class = recall_score(self.y_test, y_pred, average=None)

        print(f"\n{name} RESULTS:")
        print(f"{'Class':12} {'F1':10} {'Recall':10}")
        print("-"*35)
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:12} {f1_per_class[i]:10.4f} {recall_per_class[i]:10.4f}")
            print(f"\n Macro F1: {macro_f1:.4f}")

            return {
                'macro_f1': macro_f1,
                'f1_downward': f1_per_class[0],
                'f1_upward': f1_per_class[2],
                'recall_downward': recall_per_class[0],
                'recall_upward': recall_per_class[2]
            }
        
    def experiment_1_baseline(self):
        """Current best model with default params"""
        print("\n" + "="*80)
        print("EXPERIMENT 1: BASELINE (current Best model - CatBoost)")
        print("="*80)

        model = CatBoostClassifier(
            class_weights={0: 2.2, 1: 0.5, 2: 1.78},
            iterations = 100,
            depth=5,
            learning_rate = 0.05,
            verbose=0,
            random_seed=42
        )

        model.fit(self.X_train, self.y_train)
        results = self.evaluate_model(model, "CatBoost Baseline")
        self.results['Baseline'] = results
        return results
    
    def experiment_2_smote(self):
        """Apply SMOTE oversampling"""
        print("\n" + "="*80)
        print("EXPERIMENT 2: SMOTE OVERSAMPLING")
        print("="*80)

        # apply SMOTE to balance classes
        smote = SMOTE(sampling_strategy={0:800, 2:800}, random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(self.X_train, self.y_train)
        
        print(f"After SMOTE - Class distribution: {np.bincount(y_train_smote)}")

        model = CatBoostClassifier(
            class_weights={0: 1, 1: 1, 2: 1},  # balanced after SMOTE method
            iterations=100,
            depth=5,
            learning_rate=0.05,
            verbose=0,
            random_state=42
        )

        model.fit(X_train_smote, y_train_smote)
        results  = self.evaluate_model(model, "CatBoost + SMOTE")
        self.results['SMOTE'] = results
        return results
    
    def experiment_3_hyperparamer_tuning(self):
        """Optimized hyperparameters"""
        print("\n" + "="*80)
        print('EXPERIMENT 3: HYPERPARAMETER TUNING')
        print("="*80)

        # better params based on research
        model = CatBoostClassifier(
            class_weights={0: 2.5, 1: 0.4, 2: 2.0}, # more aggresive
            iterations = 300,
            depth=7,
            learning_rate = 0.03,
            l2_leaf_reg=5,
            border_count=128,
            verbose=0,
            random_seed=42            
        )

        model.fit(self.X_train, self.y_train)
        results = self.evaluate_model(model, "CatBoost + Hyperparameter Tuning")
        self.results['Hyperparameter Tuning'] = results
        return results
    
    def experiment_4_ensemble(self):
        """Ensemble of top 3 models from last stage"""
        print("\n" + "="*80)
        print("EXPERIMENT 4: ENSEMBLE (CatBoost + XGBoost + RandomForest)")
        print("="*80)

        # individual models
        catboost = CatBoostClassifier(
            iterations = 150,
            depth=6,
            verbose=0,
            random_seed=42            
        )

        xgb = XGBClassifier(
            scale_pos_weight=2.5,
            max_depth=5,
            learning_rate=0.05,
            n_estimators=150,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )

        rf = RandomForestClassifier(
            class_weight='balanced_subsample',
            n_estimators=150,
            max_depth=10,
            random_state=42
        )

        # ensemble (soft voting = probability averaging)
        ensemble = VotingClassifier(
            estimators=[('cat', catboost), ('xgb', xgb), ('rf', rf)],
            voting='soft'
            )
        
        ensemble.fit(self.X_train, self.y_train)
        results = self.evaluate_model(ensemble, "Ensemble (Soft Voting)")
        self.results['Ensemble'] = results
        return results
    
    def experiment_5_smote_ensemble(self):
        """SMOTE + Ensemble combined"""
        print("\n" + "="*80)
        print("EXPERIMENT 5: SMOTE + ENSEMBLE")
        print("="*80)

        # apply SMOTE
        smote = SMOTE(sampling_strategy={0:800, 2:800}, random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(self.X_train, self.y_train)

        # ensemble on balanced data
        catboost = CatBoostClassifier(
            iterations=50,
            depth=6,
            verbose=0,
            random_state=42
        )

        xgb = XGBClassifier(
            max_depth=5,
            n_estimators=150,
            random_state=42,
            use_label_encoder=False
        )

        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42
        )

        ensemble = VotingClassifier(
            estimators=[('cat', catboost), ('xgb', xgb), ('rf', rf)],
            voting = 'soft'
        )

        ensemble.fit(X_train_smote, y_train_smote)
        results = self.evaluate_model(ensemble, "SMOTE + Ensemble")
        self.results['SMOTE + Ensemble'] = results
        return results
    
    def experiment_6_full_optimization(self):
        """Everything combined: SMOTE + ENSEMBLE + BEST HYPERPARAMETERS"""
        print("\n" + "="*80)
        print("EXPERIMENT 6: FULL OPTIMIZATION (SMOTE + ENSEMBLE + BEST PARAMS)")
        print("="*80)

        # Apply SMOTE
        smote = SMOTE(sampling_strategy={0: 800, 2: 800}, random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(self.X_train, self.y_train)
        
        # Optimized models
        catboost = CatBoostClassifier(
            iterations=300,
            depth=7,
            learning_rate=0.03,
            l2_leaf_reg=5,
            verbose=0,
            random_seed=42
        )
        
        xgb = XGBClassifier(
            max_depth=6,
            learning_rate=0.03,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            random_state=42
        )
        
        ensemble = VotingClassifier(
            estimators=[('cat', catboost), ('xgb', xgb), ('rf', rf)],
            voting='soft'
        )

        ensemble.fit(X_train_smote, y_train_smote)
        results = self.evaluate_model(ensemble, "FULL OPTIMIATION")
        self.results['FULL OPTIMIZATION'] = results
        return results

    def print_summary(self):
        """Print comparison of all experiments"""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY - IMPROVEMENT TRACKING")
        print("="*80)

        summary = []
        for exp_name, metrics in self.results.items():
            summary.append({
                'Experiment': exp_name,
                'Macro F1': metrics['macro_f1'],
                'Upward F1': metrics['f1_upward'],
                'Downward F1': metrics['f1_downward'],
                'Upward Recall': metrics['recall_upward'],
                'Downward Recall': metrics['recall_downward']
            })

        df = pd.DataFrame(summary)
        df = df.sort_values('Macro F1', ascending=False)

        print("\n" + df.to_string(index=False))

        # calculate improvements
        baseline = self.results['Baseline']['macro_f1']
        best = self.results[max(self.results.keys(), key=lambda x: self.results[x]['macro_f1'])]['macro_f1']
        improvement = (best - baseline) / baseline * 100

        print(f"\n{'='*80}")
        print(f"IMPROVEMENT SUMMARY:")
        print(f"Baseline Macro F1: {baseline:.4f}")
        print(f"Best Macro F1: {best:.4f}")
        print(f"Improvement: {improvement:.1f}")
        print(f"{'='*80}")

        # save results
        df.to_csv("experiments\\results\\experiment_comparison.csv", index=False)
        print(f"Full experiment results saved to experiments/results/experiment_comparison.csv")

    def run_all_experiments(self):
        """Run all experiments sequentially"""
        self.load_data()

        self.experiment_1_baseline()
        self.experiment_2_smote()
        self.experiment_3_hyperparamer_tuning()
        self.experiment_4_ensemble()
        self.experiment_5_smote_ensemble()
        self.experiment_6_full_optimization()

        self.print_summary()

        return self.results
    
if __name__ == "__main__":
    runner = ExperimentRunner()
    results = runner.run_all_experiments()
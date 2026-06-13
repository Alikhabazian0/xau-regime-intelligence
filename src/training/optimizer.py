import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import numpy as np

class HyperparameterOptimizer:
    """
    Optimize hyperparameters for each model individually
    Uses Optuna for efficient search
    """

    def __init__(self, n_trials=30, cv_folds=5, random_state=42):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_params = {}

    def optimize(self, model_name, base_model, X_train, y_train):
        """
        Optimize hyperparameters for a specific model

        Returns:
        - best_model: trained model with best parameters
        - best_params: dictionary of best parameters
        """
        print(f"\n{'='*50}")
        print(f"Optimizing {model_name}...")
        print(f"{'='*50}")

        def objective(trial):
            # define hyperparameter search space based on model
            params = self._get_param_space(trial, model_name)

            # update model with trial parameters
            model = base_model.__class__(**params, random_state=self.random_state)

            # cross-validation with stratified folds (important for imbalance)
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                                 random_state=self.random_state)
            
            # use balanced accuracy for imbalanced data
            scores = cross_val_score(model, X_train, y_train,
                                     cv=cv,
                                     scoring='balanced_accuracy',
                                     n_jobs=-1)
            
            return scores.mean()
        
        # create and run study
        study = optuna.create_study(direction='maximize',
                                    study_name=f'{model_name}_optimization')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        # get best parameters
        self.best_params[model_name] = study.best_params
        print(f"\nBest parameters: {study.best_params}")
        print(f"Best CV balanced accuracy: {study.best_value:.4f}")

        # train best model on full training data
        best_model = base_model.__class__(**study.best_params, random_state=self.random_state)
        best_model.fit(X_train, y_train)

        return best_model, study.best_params
    
    def _get_param_space(self, trial, model_name):
        """Define hyperparameter search space for each model"""

        if model_name == 'CatBoost':
            return {
                'depth': trial.suggest_int('depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'l2_lea_reg': trial.suggest_int('l2_leaf_reg', 1, 5),
                'iterations': trial.suggest_int('iterations', 100, 300),
                'border_count': trial.suggest_int('border_count', 32, 255)
            }
        
        elif model_name == 'LightGBM':
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.5),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_btree', 0.6, 0.9)
            }
        
        elif model_name == 'XGBoost':
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.5),
                'n_estimators': trial.suggest_int('n_estimator', 100, 300)
            }
        
        elif model_name in ['RandomForest', 'ExtraTrees']:
            return {
                'n_estimator': trial.suggest_int('n_estimator', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt','log2'])
            }
        
        elif model_name == 'LogisticRegression':
            return {
                'C': trial.suggest_float('C', 0.01, 1.0, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': 'saga' if trial.params.get('penalty') == 'l1' else 'lbfgs'
            }
        
        else:
            return {}
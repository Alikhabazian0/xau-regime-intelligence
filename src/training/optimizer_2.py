"""
Hyperparameter Optimization for all models
strategy: find best params for each model individually
"""

import optuna
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
import warnings
warnings.filterwarnings('ignore')

class HyperparameterOptimizer_2:
    """
    optimize hyperparameters for any model using optuna
    """

    def __init__(self, n_trials=30, cv_folds=5, random_state=42, timeout=None):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.timeout = timeout
        self.best_params = {}
        self.best_models = {}

    def optimize(self, model_name, model_class, X_train, y_train, fixed_params=None):
        """
        optimize hyperparameters for a specific model

        parameters:
        - model_name: str (CatBoost, lightGBM, XGBoost, RandomForest, ExtraTrees)
        - model_class: the model class (not instantiated)
        - X_train, y_train: training data
        - fixed_params: dict of fixed parameters (like class_weights)
        
        Returns:
        - best_model: trained model with best parameters
        - best_params: dict of best parameters
        """

        print(f"\n{'='*60}")
        print(f" OPTIMIZING {model_name}")
        print(f"{'='*60}")

        if fixed_params is None:
            fixed_params = {}

        def objective(trial):
            # get parameter suggestions for this model
            params = self._get_param_space(trial, model_name)

            # merge with fixed parameters
            all_params = {**fixed_params, **params}

            # create model
            model = model_class(**all_params)

            # cross-validatio with stratification (impotant for imbalanced data)
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

            # use macro f1 for optimization (focus on minority claasses)
            scorer = make_scorer(f1_score, average='macro')

            try:
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scorer, n_jobs=-1)
                return scores.mean()
            except Exception as e:
                print(f"trial failed: {e}")
                return 0.0
        
        # create study
        study = optuna.create_study(
            direction = 'maximize',
            study_name = f'{model_name}_optimization',
            load_if_exists = True
        )

        # run optimization
        print(f"running {self.n_trials} trials....")
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=True)

        # get best parameters
        self.best_params[model_name] = study.best_params
        best_value = study.best_value

        print(f"\n Best CV macro F1: {best_value:.4f}")
        print(f"Best Parameters: {study.best_params}")

        # train final model with best parameters
        all_params = {**fixed_params, **study.best_params}
        best_model = model_class(**all_params)
        best_model.fit(X_train, y_train)

        self.best_models[model_name] = best_model

        return best_model, study.best_params
    
    def _get_param_space(self, trial, model_name):
        """
        Define hyperparameter search for each model
        """

        if model_name == 'RandomForest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 4, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
        elif model_name == 'ExtraTrees':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'min_samples_split': trial.suggest_int('min_samples_split',2,20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf',1,10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])            
                }
        elif model_name == 'CatBoost':
            return {
                'iterations': trial.suggest_int('iterations', 100, 300),
                'depth': trial.suggest_int('depth', 4, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255)
            }
        elif model_name == 'LightGBM':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 60),
                'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.1)
            }
        elif model_name == 'XGBoost':
            return {
                'n_estimators': trial.suggest_int('n_estimatos', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.5)
            }
        
        else:
            return {}
        
    def optimize_all(self, models_dict, X_train, y_train, fixed_params_dict=None):
        """
        optimize all models in the dictionary
        
        parameters:
        - models_dict: {'ModelName': ModelClass}
        - X_train, y_train: training data
        - fixed_params_dict: {'ModelName': {'param': value}}
        
        Returns:
        - best_models: dict of optimized models
        """

        if fixed_params_dict is None:
            fixed_params_dict = {}

        results = {}

        for model_name, model_class in models_dict.items():
            fixed_params = fixed_params_dict.get(model_name, {})

            best_model, best_params = self.optimize(
                model_name, model_class, X_train, y_train, fixed_params
            )

            results[model_name] = {
                'model': best_model,
                'best_params': best_params
            }

        return results
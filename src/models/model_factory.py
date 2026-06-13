import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from xgboost import XGBClassifier

class ModelFactory:
    """
    Factory class to create and manage all models
    Based on our analysis
    """

    @staticmethod
    def get_model(model_name, class_weights=None, random_state=42):
        """
        Get a model instance with appropriae parameters for imbalanced data

        Parameters:
        - model_name: str, name of the model
        - class_weights: dict, class weights for imbalance (4.37:1 ratio)
        - random_stare: int, for reproducibility
        """

        if model_name == 'LogisticRegression':
            return LogisticRegression(
                class_weight = class_weights,
                max_iter = 1000,
                C = 0.1,  # Regularization
                random_state = random_state,
                n_jobs=-1
            )
        
        elif model_name == 'RandomForest':
            return RandomForestClassifier(
                class_weight=class_weights,
                n_estimators=150,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=random_state,
                n_jobs=-1
            )
        
        elif model_name == 'ExtraTrees':
            return ExtraTreesClassifier(
                class_weight=class_weights,
                n_estimators=150,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=random_state,
                n_jobs=-1
            )
        
        elif model_name == 'LightGBM':
            return lgb.LGBMClassifier(
                class_weight=class_weights,
                n_estimators=150,
                max_depth=5,
                learning_rat=0.05,
                num_leaves=31,
                reg_alpha=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                verbose=-1,
                n_jobs=-1
            )
        

        elif model_name == 'XGBoost':
            # for XGBoost, use scale_pos_weight instead of class_weights
            if class_weights and len(class_weights) == 3:
                # approximate scale_pos_weight for multiclass
                scale_pos_weight = max(class_weights.values())
                
            else:
                scale_pos_weight = 1
                
            return XGBClassifier(
                scale_pos_weight = scale_pos_weight,
                max_depth = 4,
                learning_rate = 0.05,
                n_estimators = 150,
                subsample = 0.8,
                colsample_bytree = 0.8,
                reg_alpha = 0.1,
                reg_lambda = 0.1,
                random_state = random_state,
                use_label_encoder = False,
                eval_metric = 'mlogloss',
                n_jobs = -1
                )
        
        elif model_name == 'CatBoost':
            return CatBoostClassifier(
                class_weights=class_weights,
                iterations=150,
                depth=5,
                learning_rate=0.05,
                l2_leaf_reg=3,
                verbose=0,
                thread_count=-1
            )
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
    @staticmethod
    def get_all_models(class_weights=None):
        """Return dictionary of all models to compare"""
        return {
            'CatBoost': ModelFactory.get_model('CatBoost', class_weights),
            'LightGBM': ModelFactory.get_model('LightGBM', class_weights),
            'XGBoost': ModelFactory.get_model('XGBoost', class_weights),
            'RandomForest': ModelFactory.get_model('RandomForest', class_weights),
            'ExtraTrees': ModelFactory.get_model('ExtraTrees', class_weights),
            'LogisticRegression': ModelFactory.get_model('LogisticRegression', class_weights)
        }
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score, f1_score,
                             roc_auc_score, matthews_corrcoef)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """
    Comprehensive evaluation for imbalanced classification
    """

    def __init__(self, model_name, class_names=None):
        self.model_name = model_name
        self.class_names = class_names or ['downward', 'range', 'upward']
        self.results = {}

    def evaluate(self, y_true, y_pred, y_proba=None):
        """
        Run all evaluations
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION {self.model_name}")
        print(f"{'='*60}")

        # 1. classification report
        self.results['classification_report'] = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True
        )

        # print formatted report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

        # 2. confusion matrix
        self.results['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        # 3. key metrics for imbalanced data
        self.results['mcc'] = matthews_corrcoef(y_true, y_pred)
        self.results['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

        # 4. per-class f1 (especially for minority classes)
        self.results['per_class_f1'] = f1_score(y_true, y_pred, average=None)

        # 5. ROC AUC (if probabilities available)
        if y_proba is not None:
            try:
                self.results['roc_auc'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovr',
                    average='weighted'
                )
                print(f"\nWeighted ROC AUC: {self.results['roc_auc']:.4f}")
            except:
                print("warning: could not compute ROC AUC")

        # print summary
        self._print_summary()

        return self.results
    
    def _print_summary(self):
        """print focused summary for imbalanced data"""
        print("\n" + "-"*40)
        print("IMBALANCED-FOCUSED METRICS")
        print("-"*40)

        # focus on minority classes (downward and upward)
        for i, class_name in enumerate(self.class_names):
            if class_name in ['downward', 'upward']:
                f1 = self.results['per_class_f1'][i]
                recall = self.results['classification_report'][class_name]['recall']
                precision = self.results['classification_report'][class_name]['precision']
                print(f"\n{class_name.upper()}:")
                print(f" Precision: {precision:.3f}")
                print(f" Recall: {recall:.3f}")
                print(f" F1-score: {f1:.3f}")
        
        print(f"\nBalanced Accuracy: {self.results['balanced_accuracy']:.3f}")
        print(f"Matthews CC: {self.results['mcc']:.3f}")

    def plot_confuion_matrix(self, save_path=None):
        """plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(8,6))

        sns.heatmap(self.results['confusion_matrix'],
                    annot=True, fmt='d', cmap='Blues',
                    xticklabels = self.class_names,
                    yticklabels=self.class_names,
                    ax=ax)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {self.model_name}')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def get_best_model_metric(self):
        """
        Retrun the main metric for model comparison
        """
        # use average minority class F1 as primary metric
        minority_indices = [i for i, name in enumerate(self.class_names)
                            if name in ['downward', 'upward']]
        avg_minority_f1 = np.mean([self.results['per_class_f1'][i] for i in minority_indices])

        return {
            'avg_minority_f1': avg_minority_f1,
            'balanced_accuracy': self.results['balanced_accuracy'],
            'mcc': self.results['mcc']
        }
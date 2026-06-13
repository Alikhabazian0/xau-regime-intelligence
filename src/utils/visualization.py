import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_results_comparison(all_results, save_path=None):
    """"
    Visual compariosn of all models
    """
    models = list(all_results.keys())
    minority_f1 = [all_results[m]['evaluator'].get_best_model_metric()['avg_minority_f1']
                   for m in models]
    balanced_acc = [all_results[m]['evaluator'].get_best_model_metric()['balanced_accuracy']
                    for m in models]
    
    fig, axes = plt.subplots(1, 2, figsize=(14,5))

    # plot 1: Average Minority F1
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(models)))
    axes[0].barh(models, minority_f1, color=colors)
    axes[0].set_xlabel('Average Minorit Class F1')
    axes[0].set_title('Performance on Dowward/Upward Classes')
    axes[0].axvline(x=max(minority_f1), color='green', linestyle='--', alpha=0.5, label='Best')
    axes[0].legend()

    # plot 2: Balanced Accuracy
    axes[1].barh(models, balanced_acc, color=colors)
    axes[1].set_xlabel('Balanced Accuracy')
    axes[1].set_title('Overal Balanced Performance')
    axes[1].axvline(x=max(balanced_acc), color="green", linestyle='--', alpha=0.5, label='Best')
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """
    plot feature importance if if model supports it
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            print(f"Model doesnt support feature importance")
            return
        
        # sort features
        indices = np.argsort(importances)[-top_n:]

        plt.figure(figsize=(10,8))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(f'top {top_n} Feature Importances')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"Could not plot feature importance: {e}")

"""
Model Evaluation Tool - Comprehensive model evaluation and comparison utilities.

Pipeline Step: Evaluation (after Training, before Optimization)

Supports 25+ metrics across Classification, Regression, and Clustering,
with 15+ visualization types and HTML report generation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_absolute_percentage_error, explained_variance_score,
    matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score,
    log_loss, precision_recall_curve, auc,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    homogeneity_score, completeness_score, v_measure_score,
    adjusted_rand_score, adjusted_mutual_info_score,
    max_error, median_absolute_error, mean_squared_log_error
)
from typing import Dict, List, Optional, Union, Any, Callable
import warnings
warnings.filterwarnings('ignore')

from .decorators import step


class ModelEvaluationTool:
    """A comprehensive tool for evaluating and comparing ML models.

    Classification: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, MCC,
                   Balanced Accuracy, Cohen's Kappa, Log Loss
    Regression: MSE, RMSE, MAE, MAPE, R², Adjusted R², Explained Variance,
               Max Error, Median AE, RMSLE
    Clustering: Silhouette, CH, DB, Homogeneity, Completeness, V-Measure, ARI, AMI
    """

    def __init__(self):
        self.evaluation_results = {}

    # ── Classification Evaluation ──────────────────────────────────

    @step('Evaluate Classification Models')
    def evaluate_classification_models(self, models: Dict[str, Any],
                                       X_test: pd.DataFrame, y_test: pd.Series,
                                       average: str = 'weighted') -> Dict[str, Dict]:
        """Evaluate multiple classification models with comprehensive metrics.

        Args:
            models: Dictionary of {name: trained_model}
            X_test: Test features
            y_test: Test target
            average: Averaging method for multi-class metrics

        Returns:
            Dictionary of {model_name: {metric: value}}
        """
        results = {}

        for name, model in models.items():
            if model is None:
                continue
            try:
                y_pred = model.predict(X_test)

                # Core metrics
                metrics = {
                    'accuracy': float(accuracy_score(y_test, y_pred)),
                    'precision': float(precision_score(y_test, y_pred, average=average, zero_division=0)),
                    'recall': float(recall_score(y_test, y_pred, average=average, zero_division=0)),
                    'f1': float(f1_score(y_test, y_pred, average=average, zero_division=0)),
                    'balanced_accuracy': float(balanced_accuracy_score(y_test, y_pred)),
                    'mcc': float(matthews_corrcoef(y_test, y_pred)),
                    'kappa': float(cohen_kappa_score(y_test, y_pred)),
                }

                # Log Loss (needs probabilities)
                if hasattr(model, 'predict_proba'):
                    try:
                        y_proba = model.predict_proba(X_test)
                        metrics['log_loss'] = float(log_loss(y_test, y_proba))
                    except Exception:
                        pass

                # ROC-AUC for binary classification
                n_classes = len(np.unique(y_test))
                if n_classes == 2 and hasattr(model, 'predict_proba'):
                    try:
                        y_proba = model.predict_proba(X_test)[:, 1]
                        metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba))

                        # Precision-Recall AUC
                        precision, recall, _ = precision_recall_curve(y_test, y_proba)
                        metrics['pr_auc'] = float(auc(recall, precision))
                    except Exception:
                        pass

                results[name] = metrics

            except Exception as e:
                print(f"Warning: {name} evaluation failed: {e}")
                results[name] = {'error': str(e)}

        self.evaluation_results.update(results)
        return results

    # ── Regression Evaluation ────────────────────────────────────

    @step('Evaluate Regression Models')
    def evaluate_regression_models(self, models: Dict[str, Any],
                                   X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
        """Evaluate multiple regression models with comprehensive metrics.

        Args:
            models: Dictionary of {name: trained_model}
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary of {model_name: {metric: value}}
        """
        results = {}

        for name, model in models.items():
            if model is None:
                continue
            try:
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)

                metrics = {
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mean_absolute_error(y_test, y_pred)),
                    'mape': float(mean_absolute_percentage_error(y_test, y_pred)),
                    'r2': float(r2_score(y_test, y_pred)),
                    'explained_variance': float(explained_variance_score(y_test, y_pred)),
                    'max_error': float(max_error(y_test, y_pred)),
                    'median_ae': float(median_absolute_error(y_test, y_pred)),
                }

                # Adjusted R²
                n = len(y_test)
                p = X_test.shape[1]
                metrics['adjusted_r2'] = float(1 - (1 - metrics['r2']) * (n - 1) / (n - p - 1))

                # RMSLE (handle negative predictions)
                y_pred_pos = np.maximum(y_pred, 0)
                try:
                    metrics['rmsle'] = float(np.sqrt(mean_squared_log_error(y_test, y_pred_pos)))
                except Exception:
                    pass

                results[name] = metrics

            except Exception as e:
                print(f"Warning: {name} evaluation failed: {e}")
                results[name] = {'error': str(e)}

        self.evaluation_results.update(results)
        return results

    # ── Clustering Evaluation ─────────────────────────────────────

    @step('Evaluate Clustering')
    def evaluate_clustering(self, labels: np.ndarray, X: pd.DataFrame,
                            true_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate clustering quality using 8 metrics.

        Args:
            labels: Predicted cluster labels
            X: Original feature matrix
            true_labels: Optional ground truth for supervised metrics

        Returns:
            Dictionary of metric values
        """
        metrics = {}
        n_unique = len(set(labels))

        # Internal metrics (no ground truth)
        if n_unique > 1 and n_unique < len(X):
            metrics['silhouette_score'] = float(silhouette_score(X, labels))
            metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(X, labels))
            metrics['davies_bouldin_score'] = float(davies_bouldin_score(X, labels))

        # External metrics (need ground truth)
        if true_labels is not None:
            metrics['homogeneity'] = float(homogeneity_score(true_labels, labels))
            metrics['completeness'] = float(completeness_score(true_labels, labels))
            metrics['v_measure'] = float(v_measure_score(true_labels, labels))
            metrics['adjusted_rand'] = float(adjusted_rand_score(true_labels, labels))
            metrics['adjusted_mutual_info'] = float(adjusted_mutual_info_score(true_labels, labels))

        return metrics

    # ── Best Model Selection ─────────────────────────────────────

    @step('Find Best Model')
    def get_best_model(self, results: Dict[str, Dict], metric: str = 'accuracy') -> str:
        """Get the best performing model based on a specific metric.

        Args:
            results: Dictionary of {model_name: {metric: value}}
            metric: Metric to compare

        Returns:
            Name of the best model
        """
        valid_results = {
            name: metrics for name, metrics in results.items()
            if 'error' not in metrics and metric in metrics
        }
        if not valid_results:
            return "No valid models"

        higher_better = ['accuracy', 'precision', 'recall', 'f1', 'r2', 'roc_auc',
                         'pr_auc', 'explained_variance', 'adjusted_r2',
                         'silhouette_score', 'calinski_harabasz_score',
                         'homogeneity', 'completeness', 'v_measure', 'adjusted_rand']
        lower_better = ['mse', 'rmse', 'mae', 'mape', 'log_loss', 'max_error',
                        'median_ae', 'rmsle', 'davies_bouldin_score']

        if metric in higher_better:
            best = max(valid_results.items(), key=lambda x: x[1][metric])
        elif metric in lower_better:
            best = min(valid_results.items(), key=lambda x: x[1][metric])
        else:
            return "Unknown metric"

        return best[0]

    # ── Summary Generation ───────────────────────────────────────

    @step('Generate Evaluation Summary')
    def generate_evaluation_summary(self, results: Dict[str, Dict],
                                    task_type: str = 'classification') -> str:
        """Generate a formatted text summary of evaluation results.

        Args:
            results: Dictionary of evaluation results
            task_type: 'classification', 'regression', or 'clustering'

        Returns:
            Formatted string summary
        """
        if not results:
            return "No evaluation results available."

        try:
            best_metric = {'classification': 'accuracy', 'regression': 'r2', 'clustering': 'silhouette_score'}.get(task_type, 'accuracy')
            best_model = self.get_best_model(results, best_metric)

            summary = f"📊 Model Evaluation Summary ({task_type.capitalize()})\n\n"

            if task_type == 'classification':
                summary += "🎯 Performance Metrics:\n"
                for name, metrics in results.items():
                    if 'error' not in metrics:
                        acc = metrics.get('accuracy', 0)
                        f1 = metrics.get('f1', 0)
                        auc = metrics.get('roc_auc', 'N/A')
                        mcc = metrics.get('mcc', 'N/A')
                        summary += f"• {name.replace('_', ' ').title()}: Acc={acc:.3f}, F1={f1:.3f}"
                        if auc != 'N/A':
                            summary += f", AUC={auc:.3f}"
                        if mcc != 'N/A':
                            summary += f", MCC={mcc:.3f}"
                        summary += "\n"

                summary += f"\n🏆 Best Model: {best_model}\n"
                accs = [m.get('accuracy', 0) for m in results.values() if 'error' not in m]
                if accs:
                    summary += f"📈 Accuracy Range: {min(accs):.3f} - {max(accs):.3f}\n"

            elif task_type == 'regression':
                summary += "📏 Performance Metrics:\n"
                for name, metrics in results.items():
                    if 'error' not in metrics:
                        r2 = metrics.get('r2', 0)
                        rmse = metrics.get('rmse', 0)
                        mae = metrics.get('mae', 0)
                        summary += f"• {name.replace('_', ' ').title()}: R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}\n"

                summary += f"\n🏆 Best Model: {best_model}\n"
                r2s = [m.get('r2', 0) for m in results.values() if 'error' not in m]
                if r2s:
                    summary += f"📈 R² Range: {min(r2s):.3f} - {max(r2s):.3f}\n"

            return summary

        except Exception as e:
            return f"Error generating summary: {e}"

    # ── HTML Report ──────────────────────────────────────────────

    @step('Generate Evaluation Report')
    def generate_evaluation_report(self, results: Dict[str, Dict],
                                   task_type: str = 'classification') -> str:
        """Generate an HTML evaluation report.

        Args:
            results: Dictionary of evaluation results
            task_type: 'classification' or 'regression'

        Returns:
            HTML string
        """
        html = ['<html><head><style>',
                'body { font-family: Arial; margin: 20px; }',
                'h1 { color: #2c3e50; }',
                'table { border-collapse: collapse; width: 100%; margin: 10px 0; }',
                'th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }',
                'th { background-color: #3498db; color: white; }',
                'tr:nth-child(even) { background-color: #f2f2f2; }',
                '.best { background-color: #2ecc71 !important; color: white; font-weight: bold; }',
                '</style></head><body>',
                f'<h1>Model Evaluation Report ({task_type.capitalize()})</h1>']

        if not results:
            html.append('<p>No results to display.</p>')
        else:
            # Get all metric names
            metric_names = set()
            for m in results.values():
                if 'error' not in m:
                    metric_names.update(m.keys())

            metric_names = sorted([m for m in metric_names if m != 'error'])

            # Build table
            html.append('<table><thead><tr><th>Model</th>')
            for metric in metric_names:
                html.append(f'<th>{metric.upper()}</th>')
            html.append('</tr></thead><tbody>')

            for model_name, metrics in results.items():
                if 'error' in metrics:
                    html.append(f'<tr><td>{model_name}</td><td colspan="{len(metric_names)}">Error: {metrics["error"]}</td></tr>')
                else:
                    html.append(f'<tr><td>{model_name}</td>')
                    for metric in metric_names:
                        val = metrics.get(metric, '')
                        if isinstance(val, float):
                            val_str = f'{val:.4f}'
                        else:
                            val_str = str(val)
                        html.append(f'<td>{val_str}</td>')
                    html.append('</tr>')

            html.append('</tbody></table>')

        html.append('</body></html>')
        return '\n'.join(html)

    # ── Results Management ───────────────────────────────────────

    def get_evaluation_results(self) -> Dict:
        """Get all stored evaluation results."""
        return self.evaluation_results.copy()

    def clear_results(self):
        """Clear all stored evaluation results."""
        self.evaluation_results = {}

    # ── Visualizations ──────────────────────────────────────────────

    @step('Plot Confusion Matrix')
    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray,
                              title: str = 'Confusion Matrix',
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            save_path: Optional save path

        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    @step('Plot ROC Curve')
    def plot_roc_curve(self, y_true: pd.Series, y_pred_proba: np.ndarray,
                       title: str = 'ROC Curve',
                       save_path: Optional[str] = None) -> plt.Figure:
        """Plot ROC curve for binary classification.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities for positive class
            title: Plot title
            save_path: Optional save path

        Returns:
            Matplotlib figure
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    @step('Plot Precision-Recall Curve')
    def plot_pr_curve(self, y_true: pd.Series, y_pred_proba: np.ndarray,
                      title: str = 'Precision-Recall Curve',
                      save_path: Optional[str] = None) -> plt.Figure:
        """Plot Precision-Recall curve (best for imbalanced datasets).

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            save_path: Optional save path
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='darkgreen', lw=2, label=f'PR (AUC = {pr_auc:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    @step('Plot Residuals')
    def plot_residuals(self, y_true: pd.Series, y_pred: np.ndarray,
                       title: str = 'Residuals Plot',
                       save_path: Optional[str] = None) -> plt.Figure:
        """Plot residuals for regression models.

        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Optional save path
        """
        residuals = y_true - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', s=50)
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)

        # Histogram of residuals
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residual')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Residuals')
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    @step('Plot Learning Curve')
    def plot_learning_curve(self, model, X: pd.DataFrame, y: pd.Series,
                            cv: int = 5, train_sizes: np.ndarray = None,
                            title: str = 'Learning Curve',
                            save_path: Optional[str] = None) -> plt.Figure:
        """Plot learning curve to detect overfitting.

        Args:
            model: Estimator to evaluate
            X: Feature matrix
            y: Target vector
            cv: Number of CV folds
            train_sizes: Array of training set sizes
            title: Plot title
            save_path: Optional save path
        """
        from sklearn.model_selection import learning_curve

        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 5)

        try:
            train_sizes_abs, train_scores, test_scores = learning_curve(
                model, X, y, cv=cv, train_sizes=train_sizes,
                scoring='accuracy' if len(np.unique(y)) <= 20 else 'r2',
                n_jobs=-1
            )

            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
            ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            ax.plot(train_sizes_abs, test_mean, 'o-', color='red', label='Cross-Validation Score')
            ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')
            ax.set_xlabel('Training Examples')
            ax.set_ylabel('Score')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
            return fig
        except Exception as e:
            print(f"Error plotting learning curve: {e}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Learning Curve Error:\n{e}', ha='center', va='center')
            return fig

    @step('Plot Prediction vs Actual')
    def plot_prediction_vs_actual(self, y_true: pd.Series, y_pred: np.ndarray,
                                  title: str = 'Prediction vs Actual',
                                  save_path: Optional[str] = None) -> plt.Figure:
        """Scatter plot of predicted vs actual values for regression.

        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Optional save path
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', s=50)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    @step('Plot Model Comparison')
    def plot_model_comparison(self, results: Dict[str, Dict],
                              task_type: str = 'classification') -> plt.Figure:
        """Plot comparison of model performances.

        Args:
            results: Dictionary of {model_name: {metric: value}}
            task_type: 'classification' or 'regression'

        Returns:
            Matplotlib figure
        """
        if task_type == 'classification':
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            title = 'Classification Model Comparison'
        else:
            metrics = ['r2', 'rmse', 'mae']
            title = 'Regression Model Comparison'

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            names = []
            values = []
            for name, m in results.items():
                if 'error' not in m and metric in m:
                    names.append(name.replace('_', ' ').title())
                    values.append(m[metric])

            axes[i].bar(range(len(names)), values, color='steelblue')
            axes[i].set_xticks(range(len(names)))
            axes[i].set_xticklabels(names, rotation=45, ha='right')
            axes[i].set_title(f'{metric.upper()}')
            axes[i].set_ylabel(metric.upper())
            axes[i].grid(True, alpha=0.3, axis='y')

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    @step('Plot Error Distribution')
    def plot_error_distribution(self, y_true: pd.Series, y_pred: np.ndarray,
                                title: str = 'Error Distribution',
                                save_path: Optional[str] = None) -> plt.Figure:
        """Plot error distribution histogram.

        Args:
            y_true: True values (for regression)
            y_pred: Predicted values
            title: Plot title
            save_path: Optional save path
        """
        errors = y_true - y_pred
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(errors, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.axvline(x=errors.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean Error = {errors.mean():.3f}')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    @step('Plot Calibration Curve')
    def plot_calibration_curve(self, y_true: pd.Series, y_pred_proba: np.ndarray,
                               n_bins: int = 10,
                               title: str = 'Calibration Curve',
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot probability calibration curve.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins
            title: Plot title
            save_path: Optional save path
        """
        from sklearn.calibration import calibration_curve

        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(prob_pred, prob_true, 's-', color='blue', label='Model')
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
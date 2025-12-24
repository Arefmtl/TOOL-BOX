"""
Model Evaluation Tool - Comprehensive model evaluation and comparison utilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_absolute_percentage_error, explained_variance_score
)
from typing import Dict, List, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluationTool:
    """A comprehensive tool for evaluating and comparing machine learning models."""

    def __init__(self):
        self.evaluation_results = {}

    def evaluate_classification_models(self, models: Dict[str, Any],
                                     X_test: pd.DataFrame, y_test: pd.Series,
                                     average: str = 'weighted') -> Dict[str, Dict]:
        """
        Evaluate multiple classification models.

        Args:
            models: Dictionary of trained models {name: model}
            X_test: Test features
            y_test: Test target
            average: Averaging method for multiclass ('macro', 'micro', 'weighted')

        Returns:
            Dictionary of evaluation metrics for each model
        """
        results = {}

        for name, model in models.items():
            try:
                y_pred = model.predict(X_test)

                # Basic metrics
                results[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average=average, zero_division=0),
                    'recall': recall_score(y_test, y_pred, average=average, zero_division=0),
                    'f1': f1_score(y_test, y_pred, average=average, zero_division=0)
                }

                # ROC-AUC for binary classification
                if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
                    try:
                        y_proba = model.predict_proba(X_test)[:, 1]
                        results[name]['roc_auc'] = roc_auc_score(y_test, y_proba)
                    except:
                        pass

            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
                results[name] = {'error': str(e)}

        self.evaluation_results.update(results)
        return results

    def evaluate_regression_models(self, models: Dict[str, Any],
                                 X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
        """
        Evaluate multiple regression models.

        Args:
            models: Dictionary of trained models {name: model}
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary of evaluation metrics for each model
        """
        results = {}

        for name, model in models.items():
            try:
                y_pred = model.predict(X_test)

                results[name] = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'mape': mean_absolute_percentage_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred),
                    'explained_variance': explained_variance_score(y_test, y_pred)
                }

            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
                results[name] = {'error': str(e)}

        self.evaluation_results.update(results)
        return results

    def plot_confusion_matrix(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                            model_name: str = 'Model', normalize: str = None):
        """
        Plot confusion matrix for a classification model.

        Args:
            model: Trained classification model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model for plot title
            normalize: Normalization method ('true', 'pred', 'all', or None)
        """
        try:
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred, normalize=normalize)

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()

        except Exception as e:
            print(f"Error plotting confusion matrix: {str(e)}")

    def plot_roc_curve(self, models: Dict[str, Any], X_test: pd.DataFrame,
                      y_test: pd.Series):
        """
        Plot ROC curves for binary classification models.

        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test target
        """
        try:
            plt.figure(figsize=(10, 8))

            for name, model in models.items():
                if hasattr(model, 'predict_proba') and len(np.unique(y_test)) == 2:
                    try:
                        y_proba = model.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                        auc_score = roc_auc_score(y_test, y_proba)

                        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
                    except:
                        continue

            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        except Exception as e:
            print(f"Error plotting ROC curves: {str(e)}")

    def plot_model_comparison(self, results: Dict[str, Dict], task_type: str = 'classification'):
        """
        Plot model comparison results.

        Args:
            results: Evaluation results dictionary
            task_type: 'classification' or 'regression'
        """
        try:
            if task_type == 'classification':
                metrics = ['accuracy', 'precision', 'recall', 'f1']
            else:
                metrics = ['r2', 'rmse', 'mae']

            fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
            if len(metrics) == 1:
                axes = [axes]

            for i, metric in enumerate(metrics):
                model_names = []
                scores = []

                for model_name, metrics_dict in results.items():
                    if metric in metrics_dict and 'error' not in metrics_dict:
                        model_names.append(model_name.replace('_', ' ').title())
                        scores.append(metrics_dict[metric])

                if scores:
                    bars = axes[i].bar(model_names, scores, color=plt.cm.Set3(np.linspace(0, 1, len(model_names))))
                    axes[i].set_title(f'{metric.upper()} Comparison')
                    axes[i].set_ylabel(metric.upper())
                    axes[i].tick_params(axis='x', rotation=45)

                    # Add value labels on bars
                    for bar, score in zip(bars, scores):
                        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                   f'{score:.3f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error plotting model comparison: {str(e)}")

    def plot_residuals(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                      model_name: str = 'Model'):
        """
        Plot residuals for a regression model.

        Args:
            model: Trained regression model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model for plot title
        """
        try:
            y_pred = model.predict(X_test)
            residuals = y_test - y_pred

            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # Residuals vs Predicted
            axes[0].scatter(y_pred, residuals, alpha=0.6)
            axes[0].axhline(y=0, color='r', linestyle='--')
            axes[0].set_xlabel('Predicted Values')
            axes[0].set_ylabel('Residuals')
            axes[0].set_title(f'Residuals vs Predicted - {model_name}')
            axes[0].grid(True, alpha=0.3)

            # Residuals distribution
            axes[1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            axes[1].axvline(x=0, color='r', linestyle='--')
            axes[1].set_xlabel('Residuals')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title(f'Residuals Distribution - {model_name}')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error plotting residuals: {str(e)}")

    def generate_evaluation_report(self, results: Dict[str, Dict],
                                 task_type: str = 'classification',
                                 output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive evaluation report.

        Args:
            results: Evaluation results dictionary
            task_type: 'classification' or 'regression'
            output_path: Path to save HTML report (optional)

        Returns:
            HTML report as string
        """
        try:
            html_report = f"""
            <html>
            <head>
                <title>Model Evaluation Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    h2 {{ color: #666; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .metric {{ font-weight: bold; color: #2e7d32; }}
                    .best {{ background-color: #e8f5e8; }}
                    .worst {{ background-color: #ffebee; }}
                </style>
            </head>
            <body>
                <h1>Model Evaluation Report</h1>
                <h2>Task Type: {task_type.capitalize()}</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        {"".join([f"<th>{metric.upper()}</th>" for result in results.values()
                                 for metric in result.keys() if metric != 'error'])}
                    </tr>
                    {"".join([f"<tr><td>{model_name}</td>" +
                             "".join([f"<td>{metrics.get(metric, 'N/A'):.4f}</td>"
                                     for metric in list(results.values())[0].keys()
                                     if metric != 'error']) + "</tr>"
                             for model_name, metrics in results.items()
                             if 'error' not in metrics])}
                </table>
                {"".join([f"<p><strong>Error in {model_name}:</strong> {metrics['error']}</p>"
                         for model_name, metrics in results.items()
                         if 'error' in metrics])}
            </body>
            </html>
            """

            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_report)

            return html_report

        except Exception as e:
            return f"<p>Error generating report: {str(e)}</p>"

    def get_best_model(self, results: Dict[str, Dict], metric: str = 'accuracy') -> str:
        """
        Get the best performing model based on a specific metric.

        Args:
            results: Evaluation results dictionary
            metric: Metric to use for comparison

        Returns:
            Name of the best model
        """
        try:
            # Filter out models with errors
            valid_results = {name: metrics for name, metrics in results.items()
                           if 'error' not in metrics and metric in metrics}

            if not valid_results:
                return "No valid models to compare"

            # For metrics where higher is better (accuracy, precision, recall, f1, r2)
            higher_better = ['accuracy', 'precision', 'recall', 'f1', 'r2', 'roc_auc', 'explained_variance']

            if metric in higher_better:
                best_model = max(valid_results.items(), key=lambda x: x[1][metric])
            else:  # For metrics where lower is better (mse, rmse, mae, mape)
                best_model = min(valid_results.items(), key=lambda x: x[1][metric])

            return best_model[0]

        except Exception as e:
            return f"Error finding best model: {str(e)}"

    def get_evaluation_results(self) -> Dict:
        """Get all evaluation results."""
        return self.evaluation_results.copy()

    def clear_results(self):
        """Clear all evaluation results."""
        self.evaluation_results = {}

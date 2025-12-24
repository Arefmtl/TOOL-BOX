"""
Feature Importance Tool - Analyze and visualize feature importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class FeatureImportanceTool:
    """A comprehensive tool for analyzing feature importance and selection."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.importance_results = {}
        self.selected_features = {}

    def calculate_tree_importance(self, model, feature_names: List[str], method: str = 'feature_importance') -> pd.DataFrame:
        """
        Calculate feature importance for tree-based models.

        Args:
            model: Trained tree-based model
            feature_names: List of feature names
            method: 'feature_importance' or 'permutation'

        Returns:
            DataFrame with feature importances
        """
        if method == 'feature_importance':
            if hasattr(model, 'feature_importances_'):
                importance_values = model.feature_importances_
            else:
                raise ValueError("Model does not have feature_importances_ attribute")

        elif method == 'permutation':
            # This would require X and y data, so we'll skip for now
            raise ValueError("Permutation importance requires training data")

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        return importance_df

    def calculate_linear_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """
        Calculate feature importance for linear models using coefficients.

        Args:
            model: Trained linear model
            feature_names: List of feature names

        Returns:
            DataFrame with feature importances
        """
        if hasattr(model, 'coef_'):
            # Handle multi-class case
            coef = model.coef_
            if coef.ndim > 1:
                # Take absolute mean across classes
                importance_values = np.abs(coef).mean(axis=0)
            else:
                importance_values = np.abs(coef)
        else:
            raise ValueError("Model does not have coef_ attribute")

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        return importance_df

    def calculate_permutation_importance(self, model, X: pd.DataFrame, y: pd.Series,
                                       n_repeats: int = 10, random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate permutation feature importance.

        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            n_repeats: Number of permutation repeats
            random_state: Random state

        Returns:
            DataFrame with permutation importances
        """
        if random_state is None:
            random_state = self.random_state

        try:
            perm_importance = permutation_importance(
                model, X, y, n_repeats=n_repeats, random_state=random_state
            )

            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': perm_importance.importances_mean,
                'std': perm_importance.importances_std
            }).sort_values('importance', ascending=False).reset_index(drop=True)

            return importance_df

        except Exception as e:
            print(f"Error calculating permutation importance: {str(e)}")
            return pd.DataFrame()

    def univariate_feature_selection(self, X: pd.DataFrame, y: pd.Series,
                                   method: str = 'f_classif', k: Optional[int] = None) -> Dict:
        """
        Perform univariate feature selection.

        Args:
            X: Feature matrix
            y: Target vector
            method: Selection method ('f_classif', 'f_regression', 'mutual_info_classif', 'mutual_info_regression')
            k: Number of top features to select (None for all)

        Returns:
            Dictionary with selection results
        """
        # Choose scoring function
        if method == 'f_classif':
            score_func = f_classif
        elif method == 'f_regression':
            score_func = f_regression
        elif method == 'mutual_info_classif':
            score_func = mutual_info_classif
        elif method == 'mutual_info_regression':
            score_func = mutual_info_regression
        else:
            raise ValueError(f"Unsupported method: {method}")

        # Perform selection
        if k is None:
            k = 'all'

        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)

        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()

        # Get scores
        scores = selector.scores_
        p_values = getattr(selector, 'pvalues_', None)

        results = {
            'selected_features': selected_features,
            'scores': dict(zip(X.columns, scores)),
            'selected_mask': selected_mask
        }

        if p_values is not None:
            results['p_values'] = dict(zip(X.columns, p_values))

        self.selected_features[method] = selected_features
        return results

    def analyze_feature_importance(self, model, X: pd.DataFrame, y: pd.Series,
                                 feature_names: Optional[List[str]] = None,
                                 methods: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Comprehensive feature importance analysis using multiple methods.

        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            methods: List of methods to use

        Returns:
            Dictionary with importance results from different methods
        """
        if feature_names is None:
            feature_names = X.columns.tolist()

        if methods is None:
            methods = ['model', 'permutation']

        results = {}

        # Model-based importance
        if 'model' in methods:
            try:
                if hasattr(model, 'feature_importances_'):
                    results['tree_importance'] = self.calculate_tree_importance(model, feature_names)
                elif hasattr(model, 'coef_'):
                    results['linear_importance'] = self.calculate_linear_importance(model, feature_names)
            except Exception as e:
                print(f"Error calculating model importance: {str(e)}")

        # Permutation importance
        if 'permutation' in methods:
            try:
                results['permutation_importance'] = self.calculate_permutation_importance(model, X, y)
            except Exception as e:
                print(f"Error calculating permutation importance: {str(e)}")

        # Univariate selection
        if 'univariate' in methods:
            try:
                task_type = 'classification' if len(y.unique()) < 10 else 'regression'
                method = 'f_classif' if task_type == 'classification' else 'f_regression'
                results['univariate_selection'] = self.univariate_feature_selection(X, y, method=method)
            except Exception as e:
                print(f"Error in univariate selection: {str(e)}")

        self.importance_results.update(results)
        return results

    def plot_feature_importance(self, importance_df: pd.DataFrame,
                              title: str = 'Feature Importance', top_n: Optional[int] = None,
                              plot_type: str = 'bar') -> None:
        """
        Plot feature importance.

        Args:
            importance_df: DataFrame with feature importance data
            title: Plot title
            top_n: Number of top features to show
            plot_type: 'bar' or 'horizontal'
        """
        try:
            # Limit to top_n features if specified
            if top_n and len(importance_df) > top_n:
                plot_df = importance_df.head(top_n).copy()
            else:
                plot_df = importance_df.copy()

            plt.figure(figsize=(12, 8))

            if plot_type == 'bar':
                bars = plt.bar(range(len(plot_df)), plot_df['importance'],
                             color=plt.cm.Set3(np.linspace(0, 1, len(plot_df))))
                plt.xticks(range(len(plot_df)), plot_df['feature'], rotation=45, ha='right')
                plt.ylabel('Importance')
            elif plot_type == 'horizontal':
                bars = plt.barh(range(len(plot_df)), plot_df['importance'],
                              color=plt.cm.Set3(np.linspace(0, 1, len(plot_df))))
                plt.yticks(range(len(plot_df)), plot_df['feature'])
                plt.xlabel('Importance')

            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error plotting feature importance: {str(e)}")

    def plot_importance_comparison(self, importance_results: Dict[str, pd.DataFrame],
                                 top_n: int = 10) -> None:
        """
        Plot comparison of different importance methods.

        Args:
            importance_results: Dictionary of importance DataFrames
            top_n: Number of top features to show
        """
        try:
            fig, axes = plt.subplots(1, len(importance_results), figsize=(6*len(importance_results), 8))
            if len(importance_results) == 1:
                axes = [axes]

            for i, (method_name, importance_df) in enumerate(importance_results.items()):
                if isinstance(importance_df, pd.DataFrame) and 'importance' in importance_df.columns:
                    plot_df = importance_df.head(top_n).copy()
                    bars = axes[i].barh(range(len(plot_df)), plot_df['importance'])
                    axes[i].set_yticks(range(len(plot_df)))
                    axes[i].set_yticklabels(plot_df['feature'])
                    axes[i].set_xlabel('Importance')
                    axes[i].set_title(f'{method_name.replace("_", " ").title()}')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error plotting importance comparison: {str(e)}")

    def get_top_features(self, importance_df: pd.DataFrame, top_n: int = 10,
                        threshold: Optional[float] = None) -> List[str]:
        """
        Get top important features.

        Args:
            importance_df: DataFrame with feature importance
            top_n: Number of top features
            threshold: Minimum importance threshold

        Returns:
            List of top feature names
        """
        try:
            # Filter by threshold if specified
            if threshold is not None:
                filtered_df = importance_df[importance_df['importance'] >= threshold]
            else:
                filtered_df = importance_df

            # Get top_n features
            top_features = filtered_df.head(top_n)['feature'].tolist()
            return top_features

        except Exception as e:
            print(f"Error getting top features: {str(e)}")
            return []

    def generate_importance_report(self, importance_results: Dict,
                                 output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive feature importance report.

        Args:
            importance_results: Dictionary of importance analysis results
            output_path: Path to save HTML report (optional)

        Returns:
            HTML report as string
        """
        try:
            html_report = f"""
            <html>
            <head>
                <title>Feature Importance Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    h2 {{ color: #666; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .important {{ background-color: #e8f5e8; }}
                </style>
            </head>
            <body>
                <h1>Feature Importance Analysis Report</h1>
                <h2>Analysis Methods Used</h2>
                <ul>
                    {"".join([f"<li>{method.replace('_', ' ').title()}</li>" for method in importance_results.keys()])}
                </ul>
            """

            # Add tables for each method
            for method_name, result in importance_results.items():
                if isinstance(result, pd.DataFrame):
                    html_report += f"""
                    <h2>{method_name.replace('_', ' ').title()}</h2>
                    <table>
                        <tr><th>Rank</th><th>Feature</th><th>Importance</th></tr>
                        {"".join([f"<tr><td>{i+1}</td><td>{row['feature']}</td><td>{row['importance']:.4f}</td></tr>"
                                 for i, row in result.head(20).iterrows()])}
                    </table>
                    """

            html_report += "</body></html>"

            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_report)

            return html_report

        except Exception as e:
            return f"<p>Error generating report: {str(e)}</p>"

    def get_importance_results(self) -> Dict:
        """Get all importance analysis results."""
        return self.importance_results.copy()

    def get_selected_features(self) -> Dict:
        """Get selected features from different methods."""
        return self.selected_features.copy()

    def clear_results(self):
        """Clear all results."""
        self.importance_results = {}
        self.selected_features = {}

"""
Feature Selector - Analyze feature importance, select best features, and interpret models.

Pipeline Step: Feature Engineering & Selection (after preprocessing, before training)

Supports: Tree importance, linear coefficients, permutation importance,
univariate selection, RFE, SelectFromModel, and SHAP values.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression,
    mutual_info_classif, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from .decorators import step

# ── Optional SHAP import ─────────────────────────────────────────

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class FeatureSelector:
    """A comprehensive tool for analyzing feature importance and selection.

    Supports tree-based importance, linear coefficients, permutation importance,
    univariate selection, RFE, SelectFromModel, and SHAP values.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.importance_results = {}
        self.selected_features = {}

    # ── Importance calculation ──────────────────────────────────────

    @step('Tree-based Importance')
    def calculate_tree_importance(self, model, feature_names: List[str],
                                  method: str = 'feature_importance') -> pd.DataFrame:
        """Calculate feature importance for tree-based models.

        Args:
            model: Trained tree-based model (RF, GBT, XGBoost, etc.)
            feature_names: List of feature names
            method: 'feature_importance' or 'permutation'

        Returns:
            DataFrame with feature importances sorted
        """
        if method == 'feature_importance':
            if hasattr(model, 'feature_importances_'):
                importance_values = model.feature_importances_
            else:
                raise ValueError("Model does not have feature_importances_ attribute")
        else:
            raise ValueError("Permutation importance requires training data")

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        importance_df['importance_pct'] = (importance_df['importance'] / importance_df['importance'].sum() * 100).round(2)

        return importance_df

    @step('Linear Model Importance')
    def calculate_linear_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """Calculate feature importance for linear models using coefficients.

        Args:
            model: Trained linear model with coef_ attribute
            feature_names: List of feature names

        Returns:
            DataFrame with feature importances
        """
        if hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim > 1:
                importance_values = np.abs(coef).mean(axis=0)
            else:
                importance_values = np.abs(coef)
        else:
            raise ValueError("Model does not have coef_ attribute")

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        importance_df['importance_pct'] = (importance_df['importance'] / importance_df['importance'].sum() * 100).round(2)

        return importance_df

    @step('Permutation Importance')
    def calculate_permutation_importance(self, model, X: pd.DataFrame, y: pd.Series,
                                         n_repeats: int = 10,
                                         sample_size: Optional[int] = None) -> pd.DataFrame:
        """Calculate permutation feature importance.

        Uses sampling for large datasets to improve speed.

        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            n_repeats: Number of permutation repeats
            sample_size: Optional sample size for large datasets (None = all data)

        Returns:
            DataFrame with permutation importances
        """
        if sample_size is not None and sample_size < len(X):
            idx = np.random.RandomState(self.random_state).choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[idx]
            y_sample = y.iloc[idx]
        else:
            X_sample = X
            y_sample = y

        try:
            perm_importance = permutation_importance(
                model, X_sample, y_sample, n_repeats=n_repeats,
                random_state=self.random_state
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

    # ── SHAP Values ────────────────────────────────────────────────

    @step('SHAP Values')
    def calculate_shap_values(self, model, X: pd.DataFrame,
                              n_samples: int = 100) -> Optional[pd.DataFrame]:
        """Calculate SHAP feature importance with background sampling for speed.

        Args:
            model: Trained model (tree-based recommended)
            X: Feature matrix
            n_samples: Number of background samples (default: 100)

        Returns:
            DataFrame with SHAP values (mean absolute SHAP per feature)
        """
        if not SHAP_AVAILABLE:
            print("Warning: SHAP not available. Install with: pip install shap")
            return None

        try:
            # Sample background data for speed
            background = X.sample(min(n_samples, len(X)), random_state=self.random_state)

            # Use TreeExplainer for tree-based models, KernelExplainer for others
            if hasattr(model, 'feature_importances_'):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict_proba if hasattr(model, 'predict_proba') else model.predict, background)

            shap_values = explainer.shap_values(background)

            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values).mean(axis=0)

            # Mean absolute SHAP value per feature
            mean_shap = np.abs(shap_values).mean(axis=0)

            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': mean_shap
            }).sort_values('importance', ascending=False).reset_index(drop=True)

            importance_df['importance_pct'] = (importance_df['importance'] / importance_df['importance'].sum() * 100).round(2)

            self.importance_results['shap'] = importance_df
            return importance_df

        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            return None

    # ── Feature Selection methods ───────────────────────────────────

    @step('Univariate Feature Selection')
    def univariate_feature_selection(self, X: pd.DataFrame, y: pd.Series,
                                     method: str = 'f_classif',
                                     k: Optional[int] = 10) -> Dict:
        """Perform univariate feature selection using statistical tests.

        Args:
            X: Feature matrix
            y: Target vector
            method: 'f_classif', 'f_regression', 'mutual_info_classif', 'mutual_info_regression'
            k: Number of top features to select (None for auto)

        Returns:
            Dictionary with selected features, scores, and p-values
        """
        score_func_map = {
            'f_classif': f_classif,
            'f_regression': f_regression,
            'mutual_info_classif': mutual_info_classif,
            'mutual_info_regression': mutual_info_regression,
        }

        if method not in score_func_map:
            raise ValueError(f"Unsupported method: {method}. Choose from {list(score_func_map.keys())}")

        score_func = score_func_map[method]

        if k is None:
            k = 'all'

        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)

        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()

        scores = selector.scores_
        p_values = getattr(selector, 'pvalues_', None)

        results = {
            'method': method,
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'scores': dict(zip(X.columns, scores)),
            'selected_mask': selected_mask
        }

        if p_values is not None:
            results['p_values'] = dict(zip(X.columns, p_values))

        self.selected_features[method] = selected_features
        return results

    @step('Recursive Feature Elimination')
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series,
                                      n_features: Optional[int] = None,
                                      step: int = 1) -> Dict:
        """Perform Recursive Feature Elimination (RFE).

        Args:
            X: Feature matrix
            y: Target vector
            n_features: Number of features to select (None = half)
            step: Number of features to remove at each iteration

        Returns:
            Dictionary with selected features and ranking
        """
        if n_features is None:
            n_features = X.shape[1] // 2

        # Auto-detect task type
        if y.dtype in ['int64', 'int32', 'bool'] or len(y.unique()) <= 20:
            estimator = RandomForestClassifier(
                n_estimators=50, random_state=self.random_state, n_jobs=-1
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=50, random_state=self.random_state, n_jobs=-1
            )

        try:
            rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=step)
            rfe.fit(X, y)

            selected_features = X.columns[rfe.get_support()].tolist()
            ranking = pd.DataFrame({
                'feature': X.columns,
                'rank': rfe.ranking_
            }).sort_values('rank')

            results = {
                'method': 'rfe',
                'selected_features': selected_features,
                'n_selected': len(selected_features),
                'ranking': ranking,
                'estimator': rfe
            }

            self.selected_features['rfe'] = selected_features
            return results

        except Exception as e:
            return {'error': str(e)}

    @step('Select From Model')
    def select_from_model(self, X: pd.DataFrame, y: pd.Series,
                          threshold: Optional[str] = 'mean',
                          max_features: Optional[int] = None) -> Dict:
        """Select features based on importance weights from a trained model.

        Args:
            X: Feature matrix
            y: Target vector
            threshold: Feature selection threshold ('mean', 'median', or float)
            max_features: Maximum number of features to select

        Returns:
            Dictionary with selected features
        """
        # Auto-detect task type
        if y.dtype in ['int64', 'int32', 'bool'] or len(y.unique()) <= 20:
            estimator = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )

        try:
            selector = SelectFromModel(
                estimator=estimator,
                threshold=threshold,
                max_features=max_features,
                prefit=False
            )
            selector.fit(X, y)

            selected_features = X.columns[selector.get_support()].tolist()

            # Get importance values
            importances = selector.estimator_.feature_importances_
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': importances,
                'selected': selector.get_support()
            }).sort_values('importance', ascending=False)

            results = {
                'method': 'select_from_model',
                'selected_features': selected_features,
                'n_selected': len(selected_features),
                'importance_df': importance_df,
                'selector': selector
            }

            self.selected_features['select_from_model'] = selected_features
            return results

        except Exception as e:
            return {'error': str(e)}

    # ── Composite analysis ──────────────────────────────────────────

    @step('Analyze Feature Importance')
    def analyze_feature_importance(self, model, X: pd.DataFrame, y: pd.Series,
                                   feature_names: Optional[List[str]] = None,
                                   methods: Optional[List[str]] = None,
                                   use_shap: bool = False) -> Dict[str, pd.DataFrame]:
        """Comprehensive feature importance analysis using multiple methods.

        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            methods: List of methods to use (default: ['model', 'permutation'])
            use_shap: Whether to include SHAP analysis (slower)

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
                print(f"Model importance failed: {e}")

        # Permutation importance
        if 'permutation' in methods:
            try:
                results['permutation_importance'] = self.calculate_permutation_importance(model, X, y)
            except Exception as e:
                print(f"Permutation importance failed: {e}")

        # SHAP
        if use_shap and SHAP_AVAILABLE:
            try:
                shap_result = self.calculate_shap_values(model, X)
                if shap_result is not None:
                    results['shap'] = shap_result
            except Exception as e:
                print(f"SHAP analysis failed: {e}")

        self.importance_results = results
        return results

    @step('Auto Feature Selection')
    def auto_feature_selection(self, X: pd.DataFrame, y: pd.Series,
                               method: str = 'auto',
                               n_features: Optional[int] = None) -> Dict:
        """Automatically select best features using the chosen method.

        Args:
            X: Feature matrix
            y: Target vector
            method: 'auto' (tries multiple, returns best), 'univariate', 'rfe', 'model'
            n_features: Number of features to select

        Returns:
            Dictionary with selection results
        """
        if n_features is None:
            n_features = max(5, X.shape[1] // 2)

        if method == 'auto':
            # Try multiple methods and combine results
            results = {}

            # 1. Univariate
            uni_method = 'f_classif' if len(y.unique()) <= 20 else 'f_regression'
            uni_result = self.univariate_feature_selection(X, y, method=uni_method, k=n_features)
            results['univariate'] = uni_result

            # 2. RFE
            rfe_result = self.recursive_feature_elimination(X, y, n_features=n_features)
            results['rfe'] = rfe_result

            # 3. SelectFromModel
            sfm_result = self.select_from_model(X, y, max_features=n_features)
            results['model'] = sfm_result

            # Combine: features selected by at least 2 methods
            all_selected = []
            for method_name, res in results.items():
                if 'selected_features' in res:
                    all_selected.extend(res['selected_features'])

            feature_counts = pd.Series(all_selected).value_counts()
            consensus = feature_counts[feature_counts >= 2].index.tolist()

            return {
                'method': 'auto',
                'consensus_features': consensus,
                'n_consensus': len(consensus),
                'individual_results': results
            }

        elif method == 'univariate':
            uni_method = 'f_classif' if len(y.unique()) <= 20 else 'f_regression'
            return self.univariate_feature_selection(X, y, method=uni_method, k=n_features)

        elif method == 'rfe':
            return self.recursive_feature_elimination(X, y, n_features=n_features)

        elif method == 'model':
            return self.select_from_model(X, y, max_features=n_features)

        else:
            raise ValueError(f"Unknown method: {method}")

    # ── Utilities ──────────────────────────────────────────────────

    def get_top_features(self, importance_df: pd.DataFrame, n: int = 10) -> List[str]:
        """Get the top N most important features.

        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            n: Number of top features

        Returns:
            List of top feature names
        """
        return importance_df.head(n)['feature'].tolist()

    @step('Generate Importance Report')
    def generate_importance_report(self, importance_dict: Dict[str, pd.DataFrame]) -> str:
        """Generate an HTML report summarizing feature importance.

        Args:
            importance_dict: Dictionary of {method_name: importance_df}

        Returns:
            HTML string
        """
        html = ['<html><head><style>',
                'body { font-family: Arial; margin: 20px; }',
                'table { border-collapse: collapse; width: 100%; margin: 10px 0; }',
                'th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }',
                'th { background-color: #4CAF50; color: white; }',
                'tr:nth-child(even) { background-color: #f2f2f2; }',
                '</style></head><body>',
                '<h1>Feature Importance Report</h1>']

        for method_name, df in importance_dict.items():
            html.append(f'<h2>{method_name.replace("_", " ").title()}</h2>')
            html.append(df.head(10).to_html(index=False))

        html.append('</body></html>')
        return '\n'.join(html)

    # ── Plotting ──────────────────────────────────────────────────

    def plot_feature_importance(self, importance_df: pd.DataFrame,
                                title: str = 'Feature Importance',
                                top_n: int = 15,
                                plot_type: str = 'bar') -> plt.Figure:
        """Plot feature importance.

        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            title: Plot title
            top_n: Number of top features to show
            plot_type: 'bar' or 'horizontal'

        Returns:
            Matplotlib figure
        """
        df = importance_df.head(top_n).copy()

        fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.4)))

        if plot_type == 'horizontal':
            ax.barh(range(len(df)), df['importance'].values, color='steelblue')
            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(df['feature'].values)
            ax.invert_yaxis()
        else:
            ax.bar(range(len(df)), df['importance'].values, color='steelblue')
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df['feature'].values, rotation=45, ha='right')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        plt.tight_layout()
        return fig

    def plot_importance_comparison(self, importance_dict: Dict[str, pd.DataFrame],
                                   top_n: int = 10) -> plt.Figure:
        """Compare feature importance across multiple methods.

        Args:
            importance_dict: Dictionary of {method_name: importance_df}
            top_n: Number of top features

        Returns:
            Matplotlib figure
        """
        n_methods = len(importance_dict)
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 6))

        if n_methods == 1:
            axes = [axes]

        for ax, (method, df) in zip(axes, importance_dict.items()):
            df_top = df.head(top_n)
            ax.barh(range(len(df_top)), df_top['importance'].values, color='steelblue')
            ax.set_yticks(range(len(df_top)))
            ax.set_yticklabels(df_top['feature'].values)
            ax.invert_yaxis()
            ax.set_title(method.replace('_', ' ').title())
            ax.set_xlabel('Importance')

        plt.tight_layout()
        return fig
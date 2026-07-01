"""
Model Interpreter - Model interpretability with SHAP, LIME, and Partial Dependence.

Pipeline Step: Interpretability (after Evaluation/Optimization, before Export)

Provides SHAP summary/dependence plots, LIME local explanations,
Partial Dependence Plots, and HTML interpretation reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from .decorators import step

# ── Optional imports ─────────────────────────────────────────────

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    from sklearn.inspection import partial_dependence, PartialDependenceDisplay
    PDP_AVAILABLE = True
except ImportError:
    PDP_AVAILABLE = False


class ModelInterpreter:
    """Model interpretability tool with SHAP, LIME, and Partial Dependence plots.

    Provides global (SHAP, PDP) and local (LIME) explanations for any trained model.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.shap_explainer = None

    # ── SHAP ───────────────────────────────────────────────────────

    @step('SHAP Summary Plot')
    def plot_shap_summary(self, model, X: pd.DataFrame,
                          n_samples: int = 100,
                          title: str = 'SHAP Summary Plot',
                          save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Generate SHAP summary bar/beeswarm plot.

        Args:
            model: Trained model (tree-based preferred)
            X: Feature matrix
            n_samples: Background samples (default: 100 for speed)
            title: Plot title
            save_path: Optional save path

        Returns:
            Matplotlib figure or None
        """
        if not SHAP_AVAILABLE:
            print("Warning: SHAP not available. Install with: pip install shap")
            return None

        try:
            background = X.sample(min(n_samples, len(X)), random_state=self.random_state)

            if hasattr(model, 'feature_importances_'):
                self.shap_explainer = shap.TreeExplainer(model)
            else:
                self.shap_explainer = shap.KernelExplainer(
                    model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                    background
                )

            shap_values = self.shap_explainer.shap_values(background)

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Summary bar plot
            shap.summary_plot(shap_values, background, plot_type='bar', show=False, ax=axes[0])
            axes[0].set_title('Feature Importance (Mean |SHAP|)')

            # Summary dot plot
            shap.summary_plot(shap_values, background, show=False, ax=axes[1])
            axes[1].set_title('SHAP Value Distribution')

            fig.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')

            return fig

        except Exception as e:
            print(f"SHAP summary error: {e}")
            return None

    @step('SHAP Dependence Plot')
    def plot_shap_dependence(self, model, X: pd.DataFrame,
                             feature: str,
                             interaction_feature: Optional[str] = None,
                             n_samples: int = 100,
                             save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Generate SHAP dependence plot for a single feature.

        Args:
            model: Trained model
            X: Feature matrix
            feature: Feature name to plot dependence for
            interaction_feature: Optional feature for color interaction
            n_samples: Background samples
            save_path: Optional save path

        Returns:
            Matplotlib figure or None
        """
        if not SHAP_AVAILABLE:
            return None

        try:
            background = X.sample(min(n_samples, len(X)), random_state=self.random_state)

            if self.shap_explainer is None:
                if hasattr(model, 'feature_importances_'):
                    self.shap_explainer = shap.TreeExplainer(model)
                else:
                    return None

            shap_values = self.shap_explainer.shap_values(background)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.dependence_plot(feature, shap_values, background,
                                interaction_index=interaction_feature,
                                show=False, ax=ax)
            ax.set_title(f'SHAP Dependence: {feature}', fontsize=14, fontweight='bold')
            plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')

            return fig

        except Exception as e:
            print(f"SHAP dependence error: {e}")
            return None

    # ── LIME ───────────────────────────────────────────────────────

    @step('LIME Explanation')
    def explain_instance_lime(self, model, X: pd.DataFrame,
                              instance_idx: int = 0,
                              num_features: int = 10,
                              save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Generate LIME explanation for a single prediction.

        Args:
            model: Trained model
            X: Feature matrix
            instance_idx: Index of instance to explain
            num_features: Number of features to show
            save_path: Optional save path

        Returns:
            Matplotlib figure or None
        """
        if not LIME_AVAILABLE:
            print("Warning: LIME not available. Install with: pip install lime")
            return None

        try:
            # Determine mode: classification or regression
            mode = 'classification' if hasattr(model, 'predict_proba') else 'regression'

            explainer = lime_tabular.LimeTabularExplainer(
                X.values,
                feature_names=X.columns.tolist(),
                mode=mode,
                random_state=self.random_state
            )

            instance = X.iloc[instance_idx].values

            if mode == 'classification':
                exp = explainer.explain_instance(instance, model.predict_proba, num_features=num_features)
            else:
                exp = explainer.explain_instance(instance, model.predict, num_features=num_features)

            fig, ax = plt.subplots(figsize=(10, 6))
            exp.as_pyplot_figure(ax=ax)
            ax.set_title(f'LIME Explanation (Instance {instance_idx})', fontsize=14, fontweight='bold')
            plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')

            return fig

        except Exception as e:
            print(f"LIME error: {e}")
            return None

    # ── Partial Dependence ─────────────────────────────────────────

    @step('Partial Dependence Plot')
    def plot_partial_dependence(self, model, X: pd.DataFrame,
                                features: List[str],
                                n_cols: int = 3,
                                save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Generate Partial Dependence Plots for specified features.

        Args:
            model: Trained model
            X: Feature matrix
            features: List of feature names to plot
            n_cols: Number of columns in subplot grid
            save_path: Optional save path

        Returns:
            Matplotlib figure or None
        """
        if not PDP_AVAILABLE:
            print("Warning: partial_dependence not available in this sklearn version.")
            return None

        try:
            feature_indices = [list(X.columns).index(f) for f in features]

            n_features = len(features)
            n_rows = (n_features + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            axes = axes.flatten()

            for i, (feat, idx) in enumerate(zip(features, feature_indices)):
                pd_results = partial_dependence(model, X, [idx])
                ax = axes[i]
                ax.plot(pd_results['values'][0], pd_results['average'][0], 'b-', linewidth=2)
                ax.set_xlabel(feat)
                ax.set_ylabel('Partial Dependence')
                ax.set_title(f'PDP: {feat}')
                ax.grid(True, alpha=0.3)

            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)

            fig.suptitle('Partial Dependence Plots', fontsize=14, fontweight='bold')
            plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')

            return fig

        except Exception as e:
            print(f"PDP error: {e}")
            return None

    # ── Feature Ranking ────────────────────────────────────────────

    @step('Feature Ranking')
    def feature_ranking(self, model, X: pd.DataFrame,
                        methods: Optional[List[str]] = None) -> pd.DataFrame:
        """Compute feature importance ranking using multiple methods.

        Args:
            model: Trained model
            X: Feature matrix
            methods: Importance methods to use

        Returns:
            DataFrame with combined feature rankings
        """
        if methods is None:
            methods = ['coef', 'importance', 'shap']

        rankings = pd.DataFrame({'feature': X.columns})

        # Tree importance
        if 'importance' in methods and hasattr(model, 'feature_importances_'):
            rankings['tree_importance'] = model.feature_importances_

        # Coefficients
        if 'coef' in methods and hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim > 1:
                rankings['coefficient'] = np.abs(coef).mean(axis=0)
            else:
                rankings['coefficient'] = np.abs(coef)

        # SHAP
        if 'shap' in methods and SHAP_AVAILABLE:
            try:
                background = X.sample(min(100, len(X)), random_state=self.random_state)
                explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else None
                if explainer:
                    shap_values = explainer.shap_values(background)
                    if isinstance(shap_values, list):
                        shap_values = np.array(shap_values).mean(axis=0)
                    rankings['shap_importance'] = np.abs(shap_values).mean(axis=0)
            except Exception:
                pass

        return rankings

    @step('Generate Interpretation Report')
    def generate_interpretation_report(self, model, X: pd.DataFrame,
                                       y: Optional[pd.Series] = None) -> str:
        """Generate an HTML interpretation report.

        Args:
            model: Trained model
            X: Feature matrix
            y: Optional target vector for additional context

        Returns:
            HTML string
        """
        html = ['<html><head><style>',
                'body { font-family: Arial; margin: 20px; }',
                'h1, h2 { color: #2c3e50; }',
                'table { border-collapse: collapse; width: 100%; margin: 10px 0; }',
                'th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }',
                'th { background-color: #3498db; color: white; }',
                'tr:nth-child(even) { background-color: #f2f2f2; }',
                '.section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }',
                '</style></head><body>',
                '<h1>Model Interpretation Report</h1>']

        # Model info
        html.append('<div class="section">')
        html.append('<h2>Model Information</h2>')
        html.append(f'<p>Type: {type(model).__name__}</p>')
        html.append(f'<p>Features: {list(X.columns)}</p>')
        html.append(f'<p>Number of samples: {len(X)}</p>')
        html.append('</div>')

        # Feature ranking table
        html.append('<div class="section">')
        html.append('<h2>Feature Importance Ranking</h2>')
        try:
            ranking = self.feature_ranking(model, X)
            html.append(ranking.to_html(index=False))
        except Exception as e:
            html.append(f'<p>Error computing ranking: {e}</p>')
        html.append('</div>')

        # Model parameters
        html.append('<div class="section">')
        html.append('<h2>Model Parameters</h2>')
        try:
            params = model.get_params()
            html.append('<table><tr><th>Parameter</th><th>Value</th></tr>')
            for key, val in sorted(params.items()):
                html.append(f'<tr><td>{key}</td><td>{val}</td></tr>')
            html.append('</table>')
        except Exception:
            pass
        html.append('</div>')

        html.append('</body></html>')
        return '\n'.join(html)
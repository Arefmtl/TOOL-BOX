"""
Regression Tool - Multiple regression algorithms for machine learning.

Pipeline Step: Training (Regression branch)

Supports 16+ algorithms with parallel training, graceful fallbacks, and structured logging.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, QuantileRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import warnings
import os
import joblib
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')

from .decorators import step

# ── Optional imports with fallbacks ──────────────────────────────

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from sklearn.linear_model import PoissonRegressor, GammaRegressor
    POISSON_GAMMA_AVAILABLE = True
except ImportError:
    POISSON_GAMMA_AVAILABLE = False


class RegressionTool:
    """A comprehensive tool for training multiple regression algorithms.

    Supports 16 algorithms: Linear, Ridge, Lasso, ElasticNet, Huber, Quantile,
    Poisson, Gamma, RandomForest, SVR, GradientBoosting, KNN, DecisionTree,
    XGBoost, LightGBM, CatBoost, ExtraTrees, MLP, HistGradientBoosting.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.trained_models = {}

    # ── Individual training methods ─────────────────────────────────

    @step('Train Linear Regression')
    def train_linear_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                                **kwargs) -> LinearRegression:
        """Train Linear Regression model."""
        model = LinearRegression(n_jobs=-1, **kwargs)
        model.fit(X_train, y_train)
        self.trained_models['linear_regression'] = model
        return model

    @step('Train Ridge Regression')
    def train_ridge(self, X_train: pd.DataFrame, y_train: pd.Series,
                    alpha: float = 1.0, **kwargs) -> Ridge:
        """Train Ridge Regression model."""
        model = Ridge(alpha=alpha, random_state=self.random_state, **kwargs)
        model.fit(X_train, y_train)
        self.trained_models['ridge'] = model
        return model

    @step('Train Lasso Regression')
    def train_lasso(self, X_train: pd.DataFrame, y_train: pd.Series,
                    alpha: float = 1.0, **kwargs) -> Lasso:
        """Train Lasso Regression model."""
        model = Lasso(alpha=alpha, random_state=self.random_state, **kwargs)
        model.fit(X_train, y_train)
        self.trained_models['lasso'] = model
        return model

    @step('Train ElasticNet')
    def train_elastic_net(self, X_train: pd.DataFrame, y_train: pd.Series,
                          alpha: float = 1.0, l1_ratio: float = 0.5,
                          **kwargs) -> ElasticNet:
        """Train ElasticNet Regression (Ridge + Lasso hybrid).

        Args:
            alpha: Constant that multiplies the penalty terms
            l1_ratio: Mix ratio (0 = Ridge, 1 = Lasso)
        """
        model = ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio,
            random_state=self.random_state, **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['elastic_net'] = model
        return model

    @step('Train Huber Regressor')
    def train_huber(self, X_train: pd.DataFrame, y_train: pd.Series,
                    epsilon: float = 1.35, max_iter: int = 100,
                    **kwargs) -> HuberRegressor:
        """Train Huber Regressor (robust to outliers).

        Args:
            epsilon: Threshold for outlier classification
        """
        model = HuberRegressor(epsilon=epsilon, max_iter=max_iter, **kwargs)
        model.fit(X_train, y_train)
        self.trained_models['huber'] = model
        return model

    @step('Train Quantile Regressor')
    def train_quantile(self, X_train: pd.DataFrame, y_train: pd.Series,
                       alpha: float = 0.5, **kwargs) -> QuantileRegressor:
        """Train Quantile Regressor (predicts conditional quantiles).

        Args:
            alpha: Quantile to predict (0.5 = median)
        """
        model = QuantileRegressor(alpha=alpha, **kwargs)
        model.fit(X_train, y_train)
        self.trained_models['quantile'] = model
        return model

    @step('Train Poisson Regressor')
    def train_poisson(self, X_train: pd.DataFrame, y_train: pd.Series,
                      **kwargs) -> Optional[Any]:
        """Train Poisson Regressor (for count data)."""
        if not POISSON_GAMMA_AVAILABLE:
            print("Warning: PoissonRegressor not available in this sklearn version.")
            return None
        model = PoissonRegressor(**kwargs)
        model.fit(X_train, y_train)
        self.trained_models['poisson'] = model
        return model

    @step('Train Gamma Regressor')
    def train_gamma(self, X_train: pd.DataFrame, y_train: pd.Series,
                    **kwargs) -> Optional[Any]:
        """Train Gamma Regressor (for right-skewed positive data)."""
        if not POISSON_GAMMA_AVAILABLE:
            print("Warning: GammaRegressor not available in this sklearn version.")
            return None
        model = GammaRegressor(**kwargs)
        model.fit(X_train, y_train)
        self.trained_models['gamma'] = model
        return model

    @step('Train Random Forest Regressor')
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                            n_estimators: int = 100, max_depth: Optional[int] = None,
                            **kwargs) -> RandomForestRegressor:
        """Train Random Forest Regressor."""
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=self.random_state, n_jobs=-1, **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['random_forest'] = model
        return model

    @step('Train SVR')
    def train_svm(self, X_train: pd.DataFrame, y_train: pd.Series,
                  kernel: str = 'rbf', C: float = 1.0, **kwargs) -> SVR:
        """Train Support Vector Regressor."""
        model = SVR(kernel=kernel, C=C, **kwargs)
        model.fit(X_train, y_train)
        self.trained_models['svr'] = model
        return model

    @step('Train Gradient Boosting Regressor')
    def train_gradient_boosting(self, X_train: pd.DataFrame, y_train: pd.Series,
                                n_estimators: int = 100, learning_rate: float = 0.1,
                                max_depth: int = 3, **kwargs) -> GradientBoostingRegressor:
        """Train Gradient Boosting Regressor."""
        model = GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=self.random_state, **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['gradient_boosting'] = model
        return model

    @step('Train HistGradientBoosting')
    def train_hist_gradient_boosting(self, X_train: pd.DataFrame, y_train: pd.Series,
                                     max_iter: int = 100, learning_rate: float = 0.1,
                                     **kwargs) -> HistGradientBoostingRegressor:
        """Train Histogram-based Gradient Boosting (fastest sklearn GB, ~10x faster).

        Args:
            max_iter: Number of boosting iterations
            learning_rate: Learning rate
        """
        model = HistGradientBoostingRegressor(
            max_iter=max_iter, learning_rate=learning_rate,
            random_state=self.random_state, **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['hist_gradient_boosting'] = model
        return model

    @step('Train KNN Regressor')
    def train_knn(self, X_train: pd.DataFrame, y_train: pd.Series,
                  n_neighbors: int = 5, **kwargs) -> KNeighborsRegressor:
        """Train K-Nearest Neighbors Regressor."""
        model = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1, **kwargs)
        model.fit(X_train, y_train)
        self.trained_models['knn'] = model
        return model

    @step('Train Decision Tree Regressor')
    def train_decision_tree(self, X_train: pd.DataFrame, y_train: pd.Series,
                            max_depth: Optional[int] = None, **kwargs) -> DecisionTreeRegressor:
        """Train Decision Tree Regressor."""
        model = DecisionTreeRegressor(
            max_depth=max_depth, random_state=self.random_state, **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['decision_tree'] = model
        return model

    @step('Train Extra Trees Regressor')
    def train_extra_trees(self, X_train: pd.DataFrame, y_train: pd.Series,
                          n_estimators: int = 100, max_depth: Optional[int] = None,
                          **kwargs) -> ExtraTreesRegressor:
        """Train Extra Trees Regressor."""
        model = ExtraTreesRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=self.random_state, n_jobs=-1, **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['extra_trees'] = model
        return model

    @step('Train MLP Regressor')
    def train_mlp(self, X_train: pd.DataFrame, y_train: pd.Series,
                  hidden_layer_sizes: Tuple = (100,), max_iter: int = 300,
                  **kwargs) -> MLPRegressor:
        """Train MLP Neural Network Regressor."""
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter,
            random_state=self.random_state, early_stopping=True, **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['mlp'] = model
        return model

    # ── XGBoost ───────────────────────────────────────────────────

    @step('Train XGBoost Regressor')
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                      n_estimators: int = 100, max_depth: int = 6,
                      learning_rate: float = 0.1, **kwargs) -> Optional[Any]:
        """Train XGBoost Regressor."""
        if not XGBOOST_AVAILABLE:
            print("Warning: XGBoost not available. Skipping.")
            return None
        model = xgb.XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=self.random_state,
            n_jobs=-1, **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['xgboost'] = model
        return model

    # ── LightGBM ──────────────────────────────────────────────────

    @step('Train LightGBM Regressor')
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                       n_estimators: int = 100, learning_rate: float = 0.1,
                       num_leaves: int = 31, **kwargs) -> Optional[Any]:
        """Train LightGBM Regressor."""
        if not LIGHTGBM_AVAILABLE:
            print("Warning: LightGBM not available. Skipping.")
            return None
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, random_state=self.random_state,
            n_jobs=-1, verbose=-1, **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['lightgbm'] = model
        return model

    # ── CatBoost ──────────────────────────────────────────────────

    @step('Train CatBoost Regressor')
    def train_catboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                       iterations: int = 100, learning_rate: float = 0.1,
                       depth: int = 6, **kwargs) -> Optional[Any]:
        """Train CatBoost Regressor (best for categorical data)."""
        if not CATBOOST_AVAILABLE:
            print("Warning: CatBoost not available. Install with: pip install catboost")
            return None
        model = CatBoostRegressor(
            iterations=iterations, learning_rate=learning_rate,
            depth=depth, random_seed=self.random_state,
            verbose=False, **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['catboost'] = model
        return model

    # ── Batch / Parallel training ──────────────────────────────────

    @step('Train Multiple Models')
    def train_multiple_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                              models: Optional[List[str]] = None,
                              n_jobs: int = -1) -> Dict[str, Any]:
        """Train multiple regression models in parallel.

        Args:
            X_train: Training features
            y_train: Training target
            models: List of model names to train (None = all available)
            n_jobs: Number of parallel jobs (-1 = all CPUs)

        Returns:
            Dictionary of {model_name: trained_model}
        """
        if models is None:
            models = [
                'linear_regression', 'ridge', 'lasso', 'elastic_net', 'huber',
                'random_forest', 'svr', 'gradient_boosting', 'hist_gradient_boosting',
                'knn', 'decision_tree', 'extra_trees', 'mlp',
                'xgboost', 'lightgbm', 'catboost'
            ]

        configs = {
            'linear_regression': (self.train_linear_regression, {}),
            'ridge': (self.train_ridge, {}),
            'lasso': (self.train_lasso, {}),
            'elastic_net': (self.train_elastic_net, {}),
            'huber': (self.train_huber, {}),
            'quantile': (self.train_quantile, {}),
            'random_forest': (self.train_random_forest, {}),
            'svr': (self.train_svm, {}),
            'gradient_boosting': (self.train_gradient_boosting, {}),
            'hist_gradient_boosting': (self.train_hist_gradient_boosting, {}),
            'knn': (self.train_knn, {}),
            'decision_tree': (self.train_decision_tree, {}),
            'extra_trees': (self.train_extra_trees, {}),
            'mlp': (self.train_mlp, {}),
            'xgboost': (self.train_xgboost, {}),
            'lightgbm': (self.train_lightgbm, {}),
            'catboost': (self.train_catboost, {}),
        }

        to_train = [m for m in models if m in configs]

        def _train_single(name: str) -> Tuple[str, Any]:
            func, params = configs[name]
            try:
                model = func(X_train, y_train, **params)
                return name, model
            except Exception as e:
                print(f"Warning: {name} failed: {e}")
                return name, None

        results = Parallel(n_jobs=n_jobs)(
            delayed(_train_single)(name) for name in to_train
        )

        trained = {k: v for k, v in dict(results).items() if v is not None}
        self.trained_models.update(trained)
        return trained

    @step('Train All Regressors')
    def train_all_regressors(self, X_train: pd.DataFrame, y_train: pd.Series,
                             n_jobs: int = -1) -> Dict[str, Any]:
        """Convenience method: train all available regressors.

        Args:
            X_train: Training features
            y_train: Training target
            n_jobs: Number of parallel jobs

        Returns:
            Dictionary of all trained models
        """
        return self.train_multiple_models(X_train, y_train, n_jobs=n_jobs)

    # ── Prediction ─────────────────────────────────────────────────

    @step('Predict')
    def predict(self, model_name: str, X_test: pd.DataFrame) -> Optional[np.ndarray]:
        """Make predictions with a trained model.

        Args:
            model_name: Name of the trained model
            X_test: Test features

        Returns:
            Array of predictions, or None if model not found
        """
        model = self.trained_models.get(model_name)
        if model is None:
            print(f"Model '{model_name}' not found. Train it first.")
            return None
        return model.predict(X_test)

    # ── Persistence ────────────────────────────────────────────────

    @step('Save Model')
    def save_model(self, model_name: str, path: str) -> bool:
        """Save a trained model to disk."""
        if model_name not in self.trained_models:
            print(f"Model '{model_name}' not found.")
            return False
        try:
            joblib.dump(self.trained_models[model_name], path)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    @step('Load Model')
    def load_model(self, path: str, model_name: Optional[str] = None) -> Any:
        """Load a trained model from disk."""
        import warnings
        warnings.warn(
            f"Loading model from '{path}'. Only load models from trusted sources — "
            "joblib deserialization can execute arbitrary code.",
            UserWarning, stacklevel=2
        )
        model = joblib.load(path)
        if model_name:
            self.trained_models[model_name] = model
        return model

    def save_all_models(self, directory: str) -> bool:
        """Save all trained models to a directory."""
        os.makedirs(directory, exist_ok=True)
        success = True
        for name, model in self.trained_models.items():
            path = os.path.join(directory, f"{name}.joblib")
            try:
                joblib.dump(model, path)
            except Exception as e:
                print(f"Error saving {name}: {e}")
                success = False
        return success
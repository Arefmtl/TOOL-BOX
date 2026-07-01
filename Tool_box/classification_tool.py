"""
Classification Tool - Multiple classification algorithms for machine learning.

Pipeline Step: Training (Classification branch)

Supports 17+ algorithms with parallel training, graceful fallbacks, and structured logging.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
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
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class ClassificationTool:
    """A comprehensive tool for training multiple classification algorithms.

    Supports 17 algorithms: LogisticRegression, RandomForest, SVM, GradientBoosting,
    KNN, NaiveBayes, DecisionTree, XGBoost, LightGBM, AdaBoost, ExtraTrees, MLP,
    CatBoost, QDA, RidgeClassifier, VotingClassifier, StackingClassifier.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.trained_models = {}

    # ── Individual training methods ─────────────────────────────────

    @step('Train Logistic Regression')
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                                   max_iter: int = 1000, **kwargs) -> LogisticRegression:
        """Train Logistic Regression model."""
        model = LogisticRegression(
            random_state=self.random_state, max_iter=max_iter, n_jobs=-1, **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['logistic_regression'] = model
        return model

    @step('Train Random Forest')
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                             n_estimators: int = 100, max_depth: Optional[int] = None,
                             **kwargs) -> RandomForestClassifier:
        """Train Random Forest Classifier."""
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=self.random_state, n_jobs=-1, **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['random_forest'] = model
        return model

    @step('Train SVM')
    def train_svm(self, X_train: pd.DataFrame, y_train: pd.Series,
                  kernel: str = 'rbf', C: float = 1.0, **kwargs) -> SVC:
        """Train Support Vector Machine classifier."""
        model = SVC(
            kernel=kernel, C=C, probability=True,
            random_state=self.random_state, **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['svm'] = model
        return model

    @step('Train Gradient Boosting')
    def train_gradient_boosting(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 n_estimators: int = 100, learning_rate: float = 0.1,
                                 max_depth: int = 3, **kwargs) -> GradientBoostingClassifier:
        """Train Gradient Boosting Classifier."""
        model = GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=self.random_state, **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['gradient_boosting'] = model
        return model

    @step('Train KNN')
    def train_knn(self, X_train: pd.DataFrame, y_train: pd.Series,
                  n_neighbors: int = 5, **kwargs) -> KNeighborsClassifier:
        """Train K-Nearest Neighbors classifier."""
        model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1, **kwargs)
        model.fit(X_train, y_train)
        self.trained_models['knn'] = model
        return model

    @step('Train Naive Bayes')
    def train_naive_bayes(self, X_train: pd.DataFrame, y_train: pd.Series,
                          **kwargs) -> GaussianNB:
        """Train Naive Bayes classifier."""
        model = GaussianNB(**kwargs)
        model.fit(X_train, y_train)
        self.trained_models['naive_bayes'] = model
        return model

    @step('Train Decision Tree')
    def train_decision_tree(self, X_train: pd.DataFrame, y_train: pd.Series,
                            max_depth: Optional[int] = None, **kwargs) -> DecisionTreeClassifier:
        """Train Decision Tree Classifier."""
        model = DecisionTreeClassifier(
            max_depth=max_depth, random_state=self.random_state, **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['decision_tree'] = model
        return model

    @step('Train AdaBoost')
    def train_adaboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                       n_estimators: int = 50, learning_rate: float = 1.0,
                       **kwargs) -> Optional[AdaBoostClassifier]:
        """Train AdaBoost Classifier."""
        try:
            model = AdaBoostClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate,
                random_state=self.random_state, **kwargs
            )
            model.fit(X_train, y_train)
            self.trained_models['adaboost'] = model
            return model
        except Exception as e:
            print(f"Warning: AdaBoost failed: {e}")
            return None

    @step('Train Extra Trees')
    def train_extra_trees(self, X_train: pd.DataFrame, y_train: pd.Series,
                          n_estimators: int = 100, max_depth: Optional[int] = None,
                          **kwargs) -> ExtraTreesClassifier:
        """Train Extra Trees Classifier."""
        model = ExtraTreesClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=self.random_state, n_jobs=-1, **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['extra_trees'] = model
        return model

    @step('Train MLP Neural Network')
    def train_mlp(self, X_train: pd.DataFrame, y_train: pd.Series,
                  hidden_layer_sizes: Tuple = (100,), max_iter: int = 300,
                  **kwargs) -> MLPClassifier:
        """Train MLP Neural Network classifier."""
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter,
            random_state=self.random_state, early_stopping=True, **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['mlp'] = model
        return model

    # ── XGBoost ───────────────────────────────────────────────────

    @step('Train XGBoost Classifier')
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                      n_estimators: int = 100, max_depth: int = 6,
                      learning_rate: float = 0.1, **kwargs) -> Optional[Any]:
        """Train XGBoost Classifier."""
        if not XGBOOST_AVAILABLE:
            print("Warning: XGBoost not available. Skipping.")
            return None
        model = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=self.random_state,
            n_jobs=-1, **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['xgboost'] = model
        return model

    # ── LightGBM ──────────────────────────────────────────────────

    @step('Train LightGBM Classifier')
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                       n_estimators: int = 100, learning_rate: float = 0.1,
                       num_leaves: int = 31, **kwargs) -> Optional[Any]:
        """Train LightGBM Classifier."""
        if not LIGHTGBM_AVAILABLE:
            print("Warning: LightGBM not available. Skipping.")
            return None
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, random_state=self.random_state,
            n_jobs=-1, verbose=-1, **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['lightgbm'] = model
        return model

    # ── CatBoost ──────────────────────────────────────────────────

    @step('Train CatBoost Classifier')
    def train_catboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                       iterations: int = 100, learning_rate: float = 0.1,
                       depth: int = 6, **kwargs) -> Optional[Any]:
        """Train CatBoost Classifier (best for categorical data)."""
        if not CATBOOST_AVAILABLE:
            print("Warning: CatBoost not available. Install with: pip install catboost")
            return None
        model = CatBoostClassifier(
            iterations=iterations, learning_rate=learning_rate,
            depth=depth, random_seed=self.random_state,
            verbose=False, **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['catboost'] = model
        return model

    # ── QDA ──────────────────────────────────────────────────────

    @step('Train QDA')
    def train_qda(self, X_train: pd.DataFrame, y_train: pd.Series,
                  **kwargs) -> QuadraticDiscriminantAnalysis:
        """Train Quadratic Discriminant Analysis classifier.

        QDA works well for normally distributed data with separate covariance per class.
        """
        model = QuadraticDiscriminantAnalysis(**kwargs)
        model.fit(X_train, y_train)
        self.trained_models['qda'] = model
        return model

    # ── RidgeClassifier ──────────────────────────────────────────

    @step('Train Ridge Classifier')
    def train_ridge_classifier(self, X_train: pd.DataFrame, y_train: pd.Series,
                               alpha: float = 1.0, **kwargs) -> RidgeClassifier:
        """Train Ridge Classifier.

        Efficient alternative to LogisticRegression, especially for multicollinear data.
        """
        model = RidgeClassifier(alpha=alpha, random_state=self.random_state, **kwargs)
        model.fit(X_train, y_train)
        self.trained_models['ridge'] = model
        return model

    # ── VotingClassifier ─────────────────────────────────────────

    @step('Train Voting Ensemble')
    def train_voting_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                              estimators: Optional[List[Tuple[str, Any]]] = None,
                              voting: str = 'soft', **kwargs) -> VotingClassifier:
        """Train a Voting ensemble of multiple classifiers.

        Args:
            X_train: Training features
            y_train: Training target
            estimators: List of (name, model) tuples. If None, uses default 5-model ensemble
            voting: 'soft' (probability-weighted) or 'hard' (majority vote)

        Returns:
            Trained VotingClassifier
        """
        if estimators is None:
            estimators = [
                ('lr', LogisticRegression(random_state=self.random_state, max_iter=1000, n_jobs=-1)),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)),
                ('gb', GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)),
                ('svm', SVC(kernel='rbf', probability=True, random_state=self.random_state)),
                ('xgb', xgb.XGBClassifier(n_jobs=-1, random_state=self.random_state)) if XGBOOST_AVAILABLE else ('dt', DecisionTreeClassifier(random_state=self.random_state))
            ]

        model = VotingClassifier(estimators=estimators, voting=voting, **kwargs)
        model.fit(X_train, y_train)
        self.trained_models['voting_ensemble'] = model
        return model

    # ── StackingClassifier ───────────────────────────────────────

    @step('Train Stacking Ensemble')
    def train_stacking_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                                estimators: Optional[List[Tuple[str, Any]]] = None,
                                final_estimator: Optional[Any] = None,
                                **kwargs) -> StackingClassifier:
        """Train a Stacking ensemble (meta-learner on top of base models).

        Args:
            X_train: Training features
            y_train: Training target
            estimators: List of (name, model) tuples
            final_estimator: Meta-learner model (default: LogisticRegression)

        Returns:
            Trained StackingClassifier
        """
        if estimators is None:
            estimators = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)),
                ('gb', GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)),
                ('svm', SVC(kernel='rbf', probability=True, random_state=self.random_state)),
            ]

        if final_estimator is None:
            final_estimator = LogisticRegression(random_state=self.random_state, max_iter=1000, n_jobs=-1)

        model = StackingClassifier(
            estimators=estimators, final_estimator=final_estimator,
            **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['stacking_ensemble'] = model
        return model

    # ── Batch / Parallel training ──────────────────────────────────

    @step('Train Multiple Models')
    def train_multiple_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                              models: Optional[List[str]] = None,
                              n_jobs: int = -1) -> Dict[str, Any]:
        """Train multiple classification models in parallel.

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
                'logistic_regression', 'random_forest', 'svm', 'gradient_boosting',
                'knn', 'naive_bayes', 'decision_tree', 'adaboost', 'extra_trees', 'mlp',
                'xgboost', 'lightgbm', 'catboost', 'qda', 'ridge'
            ]

        configs = {
            'logistic_regression': (self.train_logistic_regression, {}),
            'random_forest': (self.train_random_forest, {}),
            'svm': (self.train_svm, {}),
            'gradient_boosting': (self.train_gradient_boosting, {}),
            'knn': (self.train_knn, {}),
            'naive_bayes': (self.train_naive_bayes, {}),
            'decision_tree': (self.train_decision_tree, {}),
            'adaboost': (self.train_adaboost, {}),
            'extra_trees': (self.train_extra_trees, {}),
            'mlp': (self.train_mlp, {}),
            'xgboost': (self.train_xgboost, {}),
            'lightgbm': (self.train_lightgbm, {}),
            'catboost': (self.train_catboost, {}),
            'qda': (self.train_qda, {}),
            'ridge': (self.train_ridge_classifier, {}),
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

    @step('Train All Classifiers')
    def train_all_classifiers(self, X_train: pd.DataFrame, y_train: pd.Series,
                              n_jobs: int = -1) -> Dict[str, Any]:
        """Convenience method: train all available classifiers including ensembles.

        Args:
            X_train: Training features
            y_train: Training target
            n_jobs: Number of parallel jobs

        Returns:
            Dictionary of all trained models
        """
        base_models = self.train_multiple_models(X_train, y_train, n_jobs=n_jobs)

        # Train ensembles separately (need base models first)
        try:
            self.train_voting_ensemble(X_train, y_train)
            self.train_stacking_ensemble(X_train, y_train)
        except Exception as e:
            print(f"Warning: Ensemble training failed: {e}")

        return self.trained_models

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
        if model_name not in self.trained_models:
            print(f"Model '{model_name}' not found. Train it first.")
            return None
        return self.trained_models[model_name].predict(X_test)

    @step('Predict Probabilities')
    def predict_proba(self, model_name: str, X_test: pd.DataFrame) -> Optional[np.ndarray]:
        """Get probability predictions from a trained model.

        Args:
            model_name: Name of the trained model
            X_test: Test features

        Returns:
            Array of probability predictions
        """
        model = self.trained_models.get(model_name)
        if model is None:
            print(f"Model '{model_name}' not found.")
            return None
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X_test)
        print(f"Model '{model_name}' does not support predict_proba.")
        return None

    # ── Persistence ────────────────────────────────────────────────

    @step('Save Model')
    def save_model(self, model_name: str, path: str) -> bool:
        """Save a trained model to disk.

        Args:
            model_name: Name of the trained model
            path: Output file path

        Returns:
            True if successful
        """
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
        """Load a trained model from disk.

        Args:
            path: Path to saved model file
            model_name: Optional name to store in self.trained_models

        Returns:
            Loaded model
        """
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
        """Save all trained models to a directory.

        Args:
            directory: Output directory

        Returns:
            True if all saved successfully
        """
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
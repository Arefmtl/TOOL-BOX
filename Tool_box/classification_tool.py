"""
Classification Tool - Multiple classification algorithms for machine learning
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost with fallback
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

class ClassificationTool:
    """A comprehensive tool for training multiple classification algorithms."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.trained_models = {}

    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                                max_iter: int = 1000, **kwargs) -> LogisticRegression:
        """
        Train Logistic Regression model.

        Args:
            X_train: Training features
            y_train: Training target
            max_iter: Maximum iterations
            **kwargs: Additional parameters

        Returns:
            Trained LogisticRegression model
        """
        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=max_iter,
            **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['logistic_regression'] = model
        return model

    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                          n_estimators: int = 100, max_depth: Optional[int] = None,
                          **kwargs) -> RandomForestClassifier:
        """
        Train Random Forest Classifier.

        Args:
            X_train: Training features
            y_train: Training target
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            **kwargs: Additional parameters

        Returns:
            Trained RandomForestClassifier
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['random_forest'] = model
        return model

    def train_svm(self, X_train: pd.DataFrame, y_train: pd.Series,
                kernel: str = 'rbf', C: float = 1.0, **kwargs) -> SVC:
        """
        Train Support Vector Machine classifier.

        Args:
            X_train: Training features
            y_train: Training target
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            C: Regularization parameter
            **kwargs: Additional parameters

        Returns:
            Trained SVC model
        """
        model = SVC(
            kernel=kernel,
            C=C,
            random_state=self.random_state,
            **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['svm'] = model
        return model

    def train_gradient_boosting(self, X_train: pd.DataFrame, y_train: pd.Series,
                              n_estimators: int = 100, learning_rate: float = 0.1,
                              max_depth: int = 3, **kwargs) -> GradientBoostingClassifier:
        """
        Train Gradient Boosting Classifier.

        Args:
            X_train: Training features
            y_train: Training target
            n_estimators: Number of boosting stages
            learning_rate: Learning rate
            max_depth: Maximum depth of individual trees
            **kwargs: Additional parameters

        Returns:
            Trained GradientBoostingClassifier
        """
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=self.random_state,
            **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['gradient_boosting'] = model
        return model

    def train_knn(self, X_train: pd.DataFrame, y_train: pd.Series,
                n_neighbors: int = 5, **kwargs) -> KNeighborsClassifier:
        """
        Train K-Nearest Neighbors classifier.

        Args:
            X_train: Training features
            y_train: Training target
            n_neighbors: Number of neighbors
            **kwargs: Additional parameters

        Returns:
            Trained KNeighborsClassifier
        """
        model = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)
        model.fit(X_train, y_train)
        self.trained_models['knn'] = model
        return model

    def train_naive_bayes(self, X_train: pd.DataFrame, y_train: pd.Series,
                        **kwargs) -> GaussianNB:
        """
        Train Naive Bayes classifier.

        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional parameters

        Returns:
            Trained GaussianNB model
        """
        model = GaussianNB(**kwargs)
        model.fit(X_train, y_train)
        self.trained_models['naive_bayes'] = model
        return model

    def train_decision_tree(self, X_train: pd.DataFrame, y_train: pd.Series,
                          max_depth: Optional[int] = None, **kwargs) -> DecisionTreeClassifier:
        """
        Train Decision Tree Classifier.

        Args:
            X_train: Training features
            y_train: Training target
            max_depth: Maximum depth of tree
            **kwargs: Additional parameters

        Returns:
            Trained DecisionTreeClassifier
        """
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=self.random_state,
            **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['decision_tree'] = model
        return model

    def train_xgboost_classifier(self, X_train: pd.DataFrame, y_train: pd.Series,
                               n_estimators: int = 100, max_depth: int = 6,
                               learning_rate: float = 0.1, subsample: float = 1.0,
                               colsample_bytree: float = 1.0, reg_alpha: float = 0,
                               reg_lambda: float = 1, early_stopping_rounds: Optional[int] = None,
                               eval_set: Optional[List[tuple]] = None, **kwargs) -> Any:
        """
        Train XGBoost Classifier.

        Args:
            X_train: Training features
            y_train: Training target
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Learning rate
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            early_stopping_rounds: Number of rounds for early stopping
            eval_set: Evaluation set for early stopping
            **kwargs: Additional XGBoost parameters

        Returns:
            Trained XGBoost classifier
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Install with: pip install xgboost")

        try:
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=self.random_state,
                **kwargs
            )

            # Handle early stopping if eval_set is provided
            if early_stopping_rounds and eval_set:
                model.fit(X_train, y_train, eval_set=eval_set, 
                         early_stopping_rounds=early_stopping_rounds, verbose=False)
            else:
                model.fit(X_train, y_train)

            self.trained_models['xgboost'] = model
            return model

        except Exception as e:
            print(f"Error training XGBoost: {str(e)}")
            return None

    def train_multiple_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                            model_list: Optional[List[str]] = None) -> Dict:
        """
        Train multiple classification models at once.

        Args:
            X_train: Training features
            y_train: Training target
            model_list: List of models to train (default: all available)

        Returns:
            Dictionary of trained models
        """
        if model_list is None:
            model_list = ['logistic', 'random_forest', 'svm', 'gradient_boosting',
                         'knn', 'naive_bayes', 'decision_tree', 'xgboost']

        models = {}

        for model_name in model_list:
            try:
                if model_name == 'logistic':
                    models[model_name] = self.train_logistic_regression(X_train, y_train)
                elif model_name == 'random_forest':
                    models[model_name] = self.train_random_forest(X_train, y_train)
                elif model_name == 'svm':
                    models[model_name] = self.train_svm(X_train, y_train)
                elif model_name == 'gradient_boosting':
                    models[model_name] = self.train_gradient_boosting(X_train, y_train)
                elif model_name == 'knn':
                    models[model_name] = self.train_knn(X_train, y_train)
                elif model_name == 'naive_bayes':
                    models[model_name] = self.train_naive_bayes(X_train, y_train)
                elif model_name == 'decision_tree':
                    models[model_name] = self.train_decision_tree(X_train, y_train)
                elif model_name == 'xgboost':
                    models[model_name] = self.train_xgboost_classifier(X_train, y_train)
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                models[model_name] = None

        return models

    def predict_single_model(self, model_name: str, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using a trained model.

        Args:
            model_name: Name of the trained model
            X_test: Test features

        Returns:
            Predictions array
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found. Train it first.")
        return self.trained_models[model_name].predict(X_test)

    def predict_proba_single_model(self, model_name: str, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions using a trained model.

        Args:
            model_name: Name of the trained model
            X_test: Test features

        Returns:
            Probability predictions array
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found. Train it first.")

        model = self.trained_models[model_name]
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X_test)
        else:
            raise ValueError(f"Model '{model_name}' does not support probability predictions.")

    def get_trained_models(self) -> Dict:
        """Get dictionary of all trained models."""
        return self.trained_models.copy()

    def clear_models(self):
        """Clear all trained models."""
        self.trained_models = {}

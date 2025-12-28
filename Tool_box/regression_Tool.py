"""
Regression Tool - Multiple regression algorithms for machine learning
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
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

class RegressionTool:
    """A comprehensive tool for training multiple regression algorithms."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.trained_models = {}

    def train_linear_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                              **kwargs) -> LinearRegression:
        """
        Train Linear Regression model.

        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional parameters

        Returns:
            Trained LinearRegression model
        """
        model = LinearRegression(**kwargs)
        model.fit(X_train, y_train)
        self.trained_models['linear_regression'] = model
        return model

    def train_ridge_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                             alpha: float = 1.0, **kwargs) -> Ridge:
        """
        Train Ridge Regression model.

        Args:
            X_train: Training features
            y_train: Training target
            alpha: Regularization strength
            **kwargs: Additional parameters

        Returns:
            Trained Ridge model
        """
        model = Ridge(alpha=alpha, random_state=self.random_state, **kwargs)
        model.fit(X_train, y_train)
        self.trained_models['ridge_regression'] = model
        return model

    def train_lasso_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                             alpha: float = 1.0, **kwargs) -> Lasso:
        """
        Train Lasso Regression model.

        Args:
            X_train: Training features
            y_train: Training target
            alpha: Regularization strength
            **kwargs: Additional parameters

        Returns:
            Trained Lasso model
        """
        model = Lasso(alpha=alpha, random_state=self.random_state, **kwargs)
        model.fit(X_train, y_train)
        self.trained_models['lasso_regression'] = model
        return model

    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                          n_estimators: int = 100, max_depth: Optional[int] = None,
                          **kwargs) -> RandomForestRegressor:
        """
        Train Random Forest Regressor.

        Args:
            X_train: Training features
            y_train: Training target
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            **kwargs: Additional parameters

        Returns:
            Trained RandomForestRegressor
        """
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['random_forest'] = model
        return model

    def train_svm(self, X_train: pd.DataFrame, y_train: pd.Series,
                kernel: str = 'rbf', C: float = 1.0, **kwargs) -> SVR:
        """
        Train Support Vector Regressor.

        Args:
            X_train: Training features
            y_train: Training target
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            C: Regularization parameter
            **kwargs: Additional parameters

        Returns:
            Trained SVR model
        """
        model = SVR(
            kernel=kernel,
            C=C,
            **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['svm'] = model
        return model

    def train_gradient_boosting(self, X_train: pd.DataFrame, y_train: pd.Series,
                              n_estimators: int = 100, learning_rate: float = 0.1,
                              max_depth: int = 3, **kwargs) -> GradientBoostingRegressor:
        """
        Train Gradient Boosting Regressor.

        Args:
            X_train: Training features
            y_train: Training target
            n_estimators: Number of boosting stages
            learning_rate: Learning rate
            max_depth: Maximum depth of individual trees
            **kwargs: Additional parameters

        Returns:
            Trained GradientBoostingRegressor
        """
        model = GradientBoostingRegressor(
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
                n_neighbors: int = 5, **kwargs) -> KNeighborsRegressor:
        """
        Train K-Nearest Neighbors regressor.

        Args:
            X_train: Training features
            y_train: Training target
            n_neighbors: Number of neighbors
            **kwargs: Additional parameters

        Returns:
            Trained KNeighborsRegressor
        """
        model = KNeighborsRegressor(n_neighbors=n_neighbors, **kwargs)
        model.fit(X_train, y_train)
        self.trained_models['knn'] = model
        return model

    def train_decision_tree(self, X_train: pd.DataFrame, y_train: pd.Series,
                          max_depth: Optional[int] = None, **kwargs) -> DecisionTreeRegressor:
        """
        Train Decision Tree Regressor.

        Args:
            X_train: Training features
            y_train: Training target
            max_depth: Maximum depth of tree
            **kwargs: Additional parameters

        Returns:
            Trained DecisionTreeRegressor
        """
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=self.random_state,
            **kwargs
        )
        model.fit(X_train, y_train)
        self.trained_models['decision_tree'] = model
        return model

    def train_xgboost_regressor(self, X_train: pd.DataFrame, y_train: pd.Series,
                              n_estimators: int = 100, max_depth: int = 6,
                              learning_rate: float = 0.1, subsample: float = 1.0,
                              colsample_bytree: float = 1.0, reg_alpha: float = 0,
                              reg_lambda: float = 1, early_stopping_rounds: Optional[int] = None,
                              eval_set: Optional[List[tuple]] = None, **kwargs) -> Any:
        """
        Train XGBoost Regressor.

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
            Trained XGBoost regressor
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Install with: pip install xgboost")

        try:
            model = xgb.XGBRegressor(
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
        Train multiple regression models at once.

        Args:
            X_train: Training features
            y_train: Training target
            model_list: List of models to train (default: all available)

        Returns:
            Dictionary of trained models
        """
        if model_list is None:
            model_list = ['linear', 'ridge', 'lasso', 'random_forest', 'svm',
                         'gradient_boosting', 'knn', 'decision_tree', 'xgboost']

        models = {}

        for model_name in model_list:
            try:
                if model_name == 'linear':
                    models[model_name] = self.train_linear_regression(X_train, y_train)
                elif model_name == 'ridge':
                    models[model_name] = self.train_ridge_regression(X_train, y_train)
                elif model_name == 'lasso':
                    models[model_name] = self.train_lasso_regression(X_train, y_train)
                elif model_name == 'random_forest':
                    models[model_name] = self.train_random_forest(X_train, y_train)
                elif model_name == 'svm':
                    models[model_name] = self.train_svm(X_train, y_train)
                elif model_name == 'gradient_boosting':
                    models[model_name] = self.train_gradient_boosting(X_train, y_train)
                elif model_name == 'knn':
                    models[model_name] = self.train_knn(X_train, y_train)
                elif model_name == 'decision_tree':
                    models[model_name] = self.train_decision_tree(X_train, y_train)
                elif model_name == 'xgboost':
                    models[model_name] = self.train_xgboost_regressor(X_train, y_train)
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

    def get_trained_models(self) -> Dict:
        """Get dictionary of all trained models."""
        return self.trained_models.copy()

    def clear_models(self):
        """Clear all trained models."""
        self.trained_models = {}

"""
Hyperparameter Tuning Tool - Comprehensive hyperparameter optimization utilities
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, f1_score, r2_score, mean_squared_error
from typing import Dict, List, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

class HyperparameterTuningTool:
    """A comprehensive tool for hyperparameter tuning and optimization."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.tuning_results = {}
        self.best_models = {}

    def grid_search_cv(self, model, param_grid: Dict, X_train: pd.DataFrame, y_train: pd.Series,
                      cv: int = 5, scoring: Optional[str] = None, n_jobs: int = -1) -> Dict:
        """
        Perform Grid Search Cross-Validation.

        Args:
            model: ML model to tune
            param_grid: Parameter grid dictionary
            X_train: Training features
            y_train: Training target
            cv: Cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs

        Returns:
            Grid search results
        """
        if scoring is None:
            # Auto-detect scoring based on task type
            if hasattr(model, 'predict_proba') or str(type(model)).find('Classifier') != -1:
                scoring = 'accuracy'
            else:
                scoring = 'neg_mean_squared_error'

        try:
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs,
                return_train_score=True, verbose=1
            )

            grid_search.fit(X_train, y_train)

            results = {
                'method': 'grid_search',
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'best_estimator': grid_search.best_estimator_,
                'cv_results': {
                    'mean_test_score': grid_search.cv_results_['mean_test_score'],
                    'std_test_score': grid_search.cv_results_['std_test_score'],
                    'mean_train_score': grid_search.cv_results_['mean_train_score'],
                    'params': grid_search.cv_results_['params']
                },
                'scoring': scoring,
                'n_candidates': len(grid_search.cv_results_['params'])
            }

            self.tuning_results['grid_search'] = results
            self.best_models['grid_search'] = grid_search.best_estimator_

            return results

        except Exception as e:
            return {'error': str(e)}

    def random_search_cv(self, model, param_distributions: Dict, X_train: pd.DataFrame, y_train: pd.Series,
                        n_iter: int = 100, cv: int = 5, scoring: Optional[str] = None,
                        n_jobs: int = -1, random_state: Optional[int] = None) -> Dict:
        """
        Perform Randomized Search Cross-Validation.

        Args:
            model: ML model to tune
            param_distributions: Parameter distributions dictionary
            X_train: Training features
            y_train: Training target
            n_iter: Number of parameter combinations to try
            cv: Cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            random_state: Random state for reproducibility

        Returns:
            Random search results
        """
        if scoring is None:
            if hasattr(model, 'predict_proba') or str(type(model)).find('Classifier') != -1:
                scoring = 'accuracy'
            else:
                scoring = 'neg_mean_squared_error'

        if random_state is None:
            random_state = self.random_state

        try:
            random_search = RandomizedSearchCV(
                model, param_distributions, n_iter=n_iter, cv=cv, scoring=scoring,
                n_jobs=n_jobs, random_state=random_state, return_train_score=True, verbose=1
            )

            random_search.fit(X_train, y_train)

            results = {
                'method': 'random_search',
                'best_params': random_search.best_params_,
                'best_score': random_search.best_score_,
                'best_estimator': random_search.best_estimator_,
                'cv_results': {
                    'mean_test_score': random_search.cv_results_['mean_test_score'],
                    'std_test_score': random_search.cv_results_['std_test_score'],
                    'mean_train_score': random_search.cv_results_['mean_train_score'],
                    'params': random_search.cv_results_['params']
                },
                'scoring': scoring,
                'n_candidates': len(random_search.cv_results_['params'])
            }

            self.tuning_results['random_search'] = results
            self.best_models['random_search'] = random_search.best_estimator_

            return results

        except Exception as e:
            return {'error': str(e)}

    def tune_classification_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                                method: str = 'grid', cv: int = 5) -> Dict:
        """
        Tune hyperparameters for common classification models.

        Args:
            model_name: Name of the model ('logistic', 'random_forest', 'svm', 'xgboost')
            X_train: Training features
            y_train: Training target
            method: Tuning method ('grid' or 'random')
            cv: Cross-validation folds

        Returns:
            Tuning results
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC

        # Define parameter grids for common models
        param_grids = {
            'logistic': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
        }

        try:
            # Create model instance
            if model_name == 'logistic':
                model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            elif model_name == 'random_forest':
                model = RandomForestClassifier(random_state=self.random_state)
            elif model_name == 'svm':
                model = SVC(random_state=self.random_state)
            else:
                return {'error': f'Unsupported model: {model_name}'}

            if method == 'grid':
                return self.grid_search_cv(model, param_grids[model_name], X_train, y_train, cv=cv)
            elif method == 'random':
                return self.random_search_cv(model, param_grids[model_name], X_train, y_train, cv=cv)
            else:
                return {'error': f'Unsupported method: {method}'}

        except Exception as e:
            return {'error': str(e)}

    def tune_regression_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                            method: str = 'grid', cv: int = 5) -> Dict:
        """
        Tune hyperparameters for common regression models.

        Args:
            model_name: Name of the model ('linear', 'ridge', 'lasso', 'random_forest', 'svm')
            X_train: Training features
            y_train: Training target
            method: Tuning method ('grid' or 'random')
            cv: Cross-validation folds

        Returns:
            Tuning results
        """
        from sklearn.linear_model import Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR

        # Define parameter grids for common models
        param_grids = {
            'ridge': {
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr']
            },
            'lasso': {
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                'max_iter': [1000, 2000, 5000]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
        }

        try:
            # Create model instance
            if model_name == 'ridge':
                model = Ridge(random_state=self.random_state)
            elif model_name == 'lasso':
                model = Lasso(random_state=self.random_state)
            elif model_name == 'random_forest':
                model = RandomForestRegressor(random_state=self.random_state)
            elif model_name == 'svm':
                model = SVR()
            else:
                return {'error': f'Unsupported model: {model_name}'}

            if method == 'grid':
                return self.grid_search_cv(model, param_grids[model_name], X_train, y_train, cv=cv)
            elif method == 'random':
                return self.random_search_cv(model, param_grids[model_name], X_train, y_train, cv=cv)
            else:
                return {'error': f'Unsupported method: {method}'}

        except Exception as e:
            return {'error': str(e)}

    def compare_tuning_methods(self, model, param_grid: Dict, X_train: pd.DataFrame, y_train: pd.Series,
                             cv: int = 5, n_iter: int = 50) -> Dict:
        """
        Compare grid search vs random search performance.

        Args:
            model: ML model to tune
            param_grid: Parameter grid/distributions
            X_train: Training features
            y_train: Training target
            cv: Cross-validation folds
            n_iter: Number of iterations for random search

        Returns:
            Comparison results
        """
        comparison = {}

        # Grid search
        grid_results = self.grid_search_cv(model, param_grid, X_train, y_train, cv=cv)
        if 'error' not in grid_results:
            comparison['grid_search'] = {
                'best_score': grid_results['best_score'],
                'n_candidates': grid_results['n_candidates'],
                'time_complexity': 'exponential'
            }

        # Random search
        random_results = self.random_search_cv(model, param_grid, X_train, y_train,
                                             n_iter=n_iter, cv=cv)
        if 'error' not in random_results:
            comparison['random_search'] = {
                'best_score': random_results['best_score'],
                'n_candidates': random_results['n_candidates'],
                'time_complexity': 'linear'
            }

        return comparison

    def get_best_model(self, method: str = 'grid_search') -> Any:
        """
        Get the best tuned model.

        Args:
            method: Tuning method ('grid_search' or 'random_search')

        Returns:
            Best tuned model
        """
        return self.best_models.get(method)

    def get_tuning_results(self) -> Dict:
        """Get all tuning results."""
        return self.tuning_results.copy()

    def clear_results(self):
        """Clear all tuning results and models."""
        self.tuning_results = {}
        self.best_models = {}

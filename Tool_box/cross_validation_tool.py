"""
Cross Validation Tool - Comprehensive cross-validation utilities.

Pipeline Step: Evaluation (used with Training and Hyperparameter Tuning)

Supports: KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold,
RepeatedKFold, LeaveOneOut, ShuffleSplit, Nested CV, and parallel processing.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    cross_val_score, cross_validate,
    KFold, StratifiedKFold, LeaveOneOut,
    RepeatedKFold, RepeatedStratifiedKFold, ShuffleSplit,
    GroupKFold, StratifiedGroupKFold, TimeSeriesSplit
)
from sklearn.metrics import make_scorer, accuracy_score, f1_score, r2_score, mean_squared_error
from joblib import Parallel, delayed
from typing import Dict, List, Optional, Union, Any, Callable
import warnings
warnings.filterwarnings('ignore')

from .decorators import step


class CrossValidationTool:
    """A comprehensive tool for cross-validation techniques.

    Supports 8 CV methods with parallel processing, auto-scoring,
    and Nested CV for unbiased hyperparameter evaluation.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.cv_results = {}

    def _auto_scoring(self, y: pd.Series) -> List[str]:
        """Auto-detect scoring metrics based on target type."""
        if y.dtype in ['int64', 'int32', 'bool'] and len(y.unique()) <= 20:
            return ['accuracy', 'f1_weighted']
        else:
            return ['neg_mean_squared_error', 'r2']

    def _get_cv_splitter(self, method: str, n_splits: int = 5,
                         shuffle: bool = True, **kwargs):
        """Get CV splitter by method name."""
        splitters = {
            'kfold': KFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_state),
            'stratified': StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_state),
            'timeseries': TimeSeriesSplit(n_splits=n_splits),
            'repeated': RepeatedKFold(n_splits=n_splits, n_repeats=kwargs.get('n_repeats', 3),
                                       random_state=self.random_state),
            'repeated_stratified': RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=kwargs.get('n_repeats', 3),
                random_state=self.random_state),
            'loo': LeaveOneOut(),
            'shuffle': ShuffleSplit(n_splits=n_splits, test_size=kwargs.get('test_size', 0.2),
                                     random_state=self.random_state),
            'group': GroupKFold(n_splits=n_splits),
        }
        return splitters.get(method, KFold(n_splits=n_splits))

    @step('K-Fold Cross Validation')
    def k_fold_cross_validation(self, model, X: pd.DataFrame, y: pd.Series,
                                n_splits: int = 5, shuffle: bool = True,
                                scoring: Optional[Union[str, List[str]]] = None,
                                n_jobs: int = -1) -> Dict:
        """Perform K-Fold Cross-Validation with parallel processing.

        Args:
            model: ML model to validate
            X: Features
            y: Target
            n_splits: Number of folds
            shuffle: Whether to shuffle data
            scoring: Scoring metric(s) (None = auto-detect)
            n_jobs: Number of parallel jobs (-1 = all CPUs)

        Returns:
            Dictionary with scores, mean, std per metric
        """
        if scoring is None:
            scoring = self._auto_scoring(y)

        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_state)

        try:
            cv_results = cross_validate(model, X, y, cv=kf, scoring=scoring, n_jobs=n_jobs)

            results = {}
            for metric in scoring:
                scores = cv_results[f'test_{metric}']
                metric_name = metric.replace('neg_', '')
                results[metric_name] = {
                    'scores': scores.tolist(),
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'min': float(scores.min()),
                    'max': float(scores.max())
                }

            results['fit_times'] = {
                'mean': float(cv_results['fit_time'].mean()),
                'std': float(cv_results['fit_time'].std())
            }

            self.cv_results['kfold'] = results
            return results

        except Exception as e:
            return {'error': str(e)}

    @step('Stratified K-Fold CV')
    def stratified_k_fold_cv(self, model, X: pd.DataFrame, y: pd.Series,
                             n_splits: int = 5, scoring: Optional[List[str]] = None,
                             n_jobs: int = -1) -> Dict:
        """Perform Stratified K-Fold CV (preserves class proportions)."""
        if scoring is None:
            scoring = self._auto_scoring(y)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        try:
            cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=n_jobs)
            results = {}
            for metric in scoring:
                scores = cv_results[f'test_{metric}']
                metric_name = metric.replace('neg_', '')
                results[metric_name] = {
                    'scores': scores.tolist(),
                    'mean': float(scores.mean()),
                    'std': float(scores.std())
                }
            self.cv_results['stratified'] = results
            return results
        except Exception as e:
            return {'error': str(e)}

    @step('Time Series CV')
    def time_series_cv(self, model, X: pd.DataFrame, y: pd.Series,
                       n_splits: int = 5, scoring: Optional[List[str]] = None) -> Dict:
        """Perform Time Series Split CV (for temporal data, no shuffling)."""
        if scoring is None:
            scoring = self._auto_scoring(y)

        tscv = TimeSeriesSplit(n_splits=n_splits)

        try:
            cv_results = cross_validate(model, X, y, cv=tscv, scoring=scoring)
            results = {}
            for metric in scoring:
                scores = cv_results[f'test_{metric}']
                metric_name = metric.replace('neg_', '')
                results[metric_name] = {
                    'scores': scores.tolist(),
                    'mean': float(scores.mean()),
                    'std': float(scores.std())
                }
            self.cv_results['timeseries'] = results
            return results
        except Exception as e:
            return {'error': str(e)}

    @step('Group K-Fold CV')
    def group_kfold_cv(self, model, X: pd.DataFrame, y: pd.Series,
                       groups: pd.Series, n_splits: int = 5,
                       scoring: Optional[List[str]] = None,
                       n_jobs: int = -1) -> Dict:
        """Perform Group K-Fold CV (no data leakage across groups).

        Args:
            model: ML model
            X: Features
            y: Target
            groups: Group labels for each sample (e.g., patient IDs)
            n_splits: Number of folds
            scoring: Scoring metrics
            n_jobs: Parallel jobs

        Returns:
            Dictionary with scores per fold
        """
        if scoring is None:
            scoring = self._auto_scoring(y)

        gkf = GroupKFold(n_splits=n_splits)

        try:
            cv_results = cross_validate(model, X, y, groups=groups, cv=gkf, scoring=scoring, n_jobs=n_jobs)
            results = {}
            for metric in scoring:
                scores = cv_results[f'test_{metric}']
                metric_name = metric.replace('neg_', '')
                results[metric_name] = {
                    'scores': scores.tolist(),
                    'mean': float(scores.mean()),
                    'std': float(scores.std())
                }
            self.cv_results['group'] = results
            return results
        except Exception as e:
            return {'error': str(e)}

    @step('Nested Cross Validation')
    def nested_cross_validation(self, model, param_grid: Dict, X: pd.DataFrame,
                                 y: pd.Series, outer_splits: int = 5,
                                 inner_splits: int = 3,
                                 scoring: Optional[str] = None) -> Dict:
        """Perform Nested Cross-Validation for unbiased evaluation.

        Outer CV evaluates model performance; Inner CV tunes hyperparameters.
        This prevents data leakage between tuning and evaluation.

        Args:
            model: Untrained model instance
            param_grid: Dictionary of hyperparameters to search
            X: Features
            y: Target
            outer_splits: Number of outer folds
            inner_splits: Number of inner CV folds
            scoring: Scoring metric

        Returns:
            Dictionary with outer scores and best params per fold
        """
        if scoring is None:
            scoring = 'accuracy' if len(np.unique(y)) <= 20 else 'r2'

        from sklearn.model_selection import GridSearchCV

        outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=self.random_state)
        inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=self.random_state)

        outer_scores = []
        best_params_list = []

        for train_idx, test_idx in outer_cv.split(X, y):
            X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
            y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]

            # Inner CV: hyperparameter tuning
            gs = GridSearchCV(model, param_grid, cv=inner_cv, scoring=scoring, n_jobs=-1)
            gs.fit(X_train_outer, y_train_outer)

            # Outer CV: evaluate best model
            from sklearn.metrics import accuracy_score, r2_score
            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test_outer)

            if scoring in ['accuracy', 'f1']:
                outer_scores.append(accuracy_score(y_test_outer, y_pred))
            else:
                outer_scores.append(r2_score(y_test_outer, y_pred))

            best_params_list.append(gs.best_params_)

        results = {
            'outer_scores': outer_scores,
            'mean_score': float(np.mean(outer_scores)),
            'std_score': float(np.std(outer_scores)),
            'best_params_per_fold': best_params_list
        }

        self.cv_results['nested'] = results
        return results

    @step('Compare CV Methods')
    def compare_cv_methods(self, model, X: pd.DataFrame, y: pd.Series,
                           methods: Optional[List[str]] = None,
                           n_splits: int = 5, scoring: Optional[str] = None,
                           n_jobs: int = -1) -> Dict:
        """Compare multiple CV methods on the same model.

        Args:
            model: ML model
            X: Features
            y: Target
            methods: CV methods to compare
            n_splits: Number of folds
            scoring: Scoring metric
            n_jobs: Parallel jobs

        Returns:
            Dictionary with results for each CV method
        """
        if methods is None:
            methods = ['kfold', 'stratified', 'timeseries']

        if scoring is None:
            scoring = self._auto_scoring(y)[0]

        results = {}
        for method in methods:
            try:
                cv = self._get_cv_splitter(method, n_splits=n_splits)
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
                results[method] = {
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'scores': scores.tolist()
                }
            except Exception as e:
                results[method] = {'error': str(e)}

        self.cv_results['comparison'] = results
        return results

    def get_cv_results(self) -> Dict:
        """Get all stored CV results."""
        return self.cv_results.copy()

    def clear_results(self):
        """Clear all stored CV results."""
        self.cv_results = {}
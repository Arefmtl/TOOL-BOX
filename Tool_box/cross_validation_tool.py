"""
Cross Validation Tool - Comprehensive cross-validation utilities
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    cross_val_score, cross_val_predict, cross_validate,
    KFold, StratifiedKFold, LeaveOneOut, LeavePOut,
    RepeatedKFold, RepeatedStratifiedKFold, ShuffleSplit,
    GroupKFold, TimeSeriesSplit
)
from sklearn.metrics import make_scorer, accuracy_score, f1_score, r2_score, mean_squared_error
from typing import Dict, List, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

class CrossValidationTool:
    """A comprehensive tool for cross-validation techniques."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.cv_results = {}

    def k_fold_cross_validation(self, model, X: pd.DataFrame, y: pd.Series,
                              n_splits: int = 5, shuffle: bool = True,
                              scoring: Optional[Union[str, List[str]]] = None) -> Dict:
        """
        Perform K-Fold Cross-Validation.

        Args:
            model: ML model to validate
            X: Features
            y: Target
            n_splits: Number of folds
            shuffle: Whether to shuffle data
            scoring: Scoring metric(s)

        Returns:
            Cross-validation results
        """
        if scoring is None:
            # Auto-detect scoring based on target type
            if y.dtype in ['int64', 'int32'] and len(y.unique()) > 10:
                scoring = ['neg_mean_squared_error', 'r2']
            else:
                scoring = ['accuracy', 'f1_weighted']

        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_state)

        try:
            cv_results = cross_validate(
                model, X, y, cv=kf, scoring=scoring, return_train_score=True
            )

            results = {
                'method': 'k_fold',
                'n_splits': n_splits,
                'test_scores': {metric: cv_results[f'test_{metric}'] for metric in scoring},
                'train_scores': {metric: cv_results[f'train_{metric}'] for metric in scoring},
                'fit_times': cv_results['fit_time'],
                'score_times': cv_results['score_time'],
                'mean_test_scores': {metric: np.mean(cv_results[f'test_{metric}']) for metric in scoring},
                'std_test_scores': {metric: np.std(cv_results[f'test_{metric}']) for metric in scoring}
            }

            self.cv_results['k_fold'] = results
            return results

        except Exception as e:
            return {'error': str(e)}

    def stratified_k_fold_cross_validation(self, model, X: pd.DataFrame, y: pd.Series,
                                         n_splits: int = 5, shuffle: bool = True,
                                         scoring: Optional[List[str]] = None) -> Dict:
        """
        Perform Stratified K-Fold Cross-Validation.

        Args:
            model: ML model to validate
            X: Features
            y: Target
            n_splits: Number of folds
            shuffle: Whether to shuffle data
            scoring: Scoring metrics

        Returns:
            Cross-validation results
        """
        if scoring is None:
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_state)

        try:
            cv_results = cross_validate(
                model, X, y, cv=skf, scoring=scoring, return_train_score=True
            )

            results = {
                'method': 'stratified_k_fold',
                'n_splits': n_splits,
                'test_scores': {metric: cv_results[f'test_{metric}'] for metric in scoring},
                'train_scores': {metric: cv_results[f'train_{metric}'] for metric in scoring},
                'fit_times': cv_results['fit_time'],
                'score_times': cv_results['score_time'],
                'mean_test_scores': {metric: np.mean(cv_results[f'test_{metric}']) for metric in scoring},
                'std_test_scores': {metric: np.std(cv_results[f'test_{metric}']) for metric in scoring}
            }

            self.cv_results['stratified_k_fold'] = results
            return results

        except Exception as e:
            return {'error': str(e)}

    def time_series_split(self, model, X: pd.DataFrame, y: pd.Series,
                         n_splits: int = 5, max_train_size: Optional[int] = None,
                         scoring: Optional[str] = None) -> Dict:
        """
        Perform Time Series Cross-Validation.

        Args:
            model: ML model to validate
            X: Features (time-ordered)
            y: Target (time-ordered)
            n_splits: Number of splits
            max_train_size: Maximum training set size
            scoring: Scoring metric

        Returns:
            Cross-validation results
        """
        if scoring is None:
            scoring = 'neg_mean_squared_error' if y.dtype in ['float64', 'int64'] else 'accuracy'

        tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size)

        try:
            scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring)

            results = {
                'method': 'time_series_split',
                'n_splits': n_splits,
                'scores': scores.tolist(),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores)
            }

            self.cv_results['time_series'] = results
            return results

        except Exception as e:
            return {'error': str(e)}

    def repeated_cross_validation(self, model, X: pd.DataFrame, y: pd.Series,
                                n_splits: int = 5, n_repeats: int = 10,
                                stratified: bool = True, scoring: Optional[str] = None) -> Dict:
        """
        Perform Repeated Cross-Validation.

        Args:
            model: ML model to validate
            X: Features
            y: Target
            n_splits: Number of folds
            n_repeats: Number of repetitions
            stratified: Whether to use stratified folds
            scoring: Scoring metric

        Returns:
            Cross-validation results
        """
        if scoring is None:
            scoring = 'accuracy' if stratified else 'neg_mean_squared_error'

        if stratified:
            cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=self.random_state)
        else:
            cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=self.random_state)

        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

            results = {
                'method': 'repeated_cv',
                'n_splits': n_splits,
                'n_repeats': n_repeats,
                'stratified': stratified,
                'scores': scores.tolist(),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'confidence_interval': [
                    np.mean(scores) - 1.96 * np.std(scores),
                    np.mean(scores) + 1.96 * np.std(scores)
                ]
            }

            self.cv_results['repeated_cv'] = results
            return results

        except Exception as e:
            return {'error': str(e)}

    def leave_one_out_cross_validation(self, model, X: pd.DataFrame, y: pd.Series,
                                    scoring: Optional[str] = None) -> Dict:
        """
        Perform Leave-One-Out Cross-Validation.

        Args:
            model: ML model to validate
            X: Features
            y: Target
            scoring: Scoring metric

        Returns:
            Cross-validation results
        """
        if scoring is None:
            scoring = 'accuracy' if y.dtype in ['object', 'category'] else 'neg_mean_squared_error'

        loo = LeaveOneOut()

        try:
            scores = cross_val_score(model, X, y, cv=loo, scoring=scoring)

            results = {
                'method': 'leave_one_out',
                'n_splits': len(X),
                'scores': scores.tolist(),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'accuracy': np.mean(scores > 0) if 'neg_' in scoring else np.mean(scores)
            }

            self.cv_results['leave_one_out'] = results
            return results

        except Exception as e:
            return {'error': str(e)}

    def shuffle_split_cross_validation(self, model, X: pd.DataFrame, y: pd.Series,
                                     n_splits: int = 10, test_size: float = 0.2,
                                     scoring: Optional[str] = None) -> Dict:
        """
        Perform Shuffle Split Cross-Validation.

        Args:
            model: ML model to validate
            X: Features
            y: Target
            n_splits: Number of splits
            test_size: Test set size
            scoring: Scoring metric

        Returns:
            Cross-validation results
        """
        if scoring is None:
            scoring = 'accuracy' if y.dtype in ['object', 'category'] else 'r2'

        ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=self.random_state)

        try:
            scores = cross_val_score(model, X, y, cv=ss, scoring=scoring)

            results = {
                'method': 'shuffle_split',
                'n_splits': n_splits,
                'test_size': test_size,
                'scores': scores.tolist(),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'confidence_interval': [
                    np.mean(scores) - 1.96 * np.std(scores),
                    np.mean(scores) + 1.96 * np.std(scores)
                ]
            }

            self.cv_results['shuffle_split'] = results
            return results

        except Exception as e:
            return {'error': str(e)}

    def compare_cv_methods(self, model, X: pd.DataFrame, y: pd.Series,
                         methods: Optional[List[str]] = None) -> Dict:
        """
        Compare different cross-validation methods.

        Args:
            model: ML model to validate
            X: Features
            y: Target
            methods: List of CV methods to compare

        Returns:
            Comparison results
        """
        if methods is None:
            methods = ['k_fold', 'stratified_k_fold', 'shuffle_split']

        comparison = {}

        for method in methods:
            try:
                if method == 'k_fold':
                    result = self.k_fold_cross_validation(model, X, y)
                elif method == 'stratified_k_fold':
                    result = self.stratified_k_fold_cross_validation(model, X, y)
                elif method == 'shuffle_split':
                    result = self.shuffle_split_cross_validation(model, X, y)
                elif method == 'time_series':
                    result = self.time_series_split(model, X, y)
                else:
                    continue

                if 'error' not in result:
                    # Extract main metric
                    main_metric = list(result['mean_test_scores'].keys())[0] if 'mean_test_scores' in result else 'mean_score'
                    comparison[method] = {
                        'mean_score': result.get('mean_score', result['mean_test_scores'][main_metric]),
                        'std_score': result.get('std_score', result['std_test_scores'][main_metric])
                    }

            except Exception as e:
                comparison[method] = {'error': str(e)}

        return comparison

    def get_cv_results(self) -> Dict:
        """Get all cross-validation results."""
        return self.cv_results.copy()

    def clear_results(self):
        """Clear all cross-validation results."""
        self.cv_results = {}

"""
Optimizer - Hyperparameter optimization combining Grid Search, Random Search,
Optuna, Hyperopt, and Scikit-Optimize.

Pipeline Step: Optimization (after Evaluation, before Export)

Supports: GridSearchCV, RandomizedSearchCV, Optuna (TPE + Pruning),
Hyperopt (TPE), Scikit-Optimize (GP), and comparison across all methods.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import warnings
import time
from abc import ABC, abstractmethod
warnings.filterwarnings('ignore')

from .decorators import step

# ── Advanced optimization libraries ─────────────────────────────

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    from hyperopt.pyll.base import scope
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


class Optimizer:
    """Unified hyperparameter optimization tool.

    Provides 5 optimization methods: GridSearch, RandomSearch, Optuna, Hyperopt, Skopt.
    Supports automatic model/param space detection, pruning, and method comparison.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.optimization_results = {}

    # ── Auto param spaces ──────────────────────────────────────────

    @staticmethod
    def _get_default_param_space(model_name: str) -> Dict:
        """Get default parameter search space for common models."""
        spaces = {
            'logistic_regression': {
                'C': {'type': 'float', 'low': 0.001, 'high': 100, 'log': True},
                'max_iter': {'type': 'int', 'low': 100, 'high': 1000},
            },
            'random_forest': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                'max_depth': {'type': 'int', 'low': 3, 'high': 30},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
            },
            'svm': {
                'C': {'type': 'float', 'low': 0.01, 'high': 100, 'log': True},
                'gamma': {'type': 'float', 'low': 0.0001, 'high': 10, 'log': True},
            },
            'gradient_boosting': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            },
            'xgboost': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                'max_depth': {'type': 'int', 'low': 3, 'high': 15},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
            },
            'lightgbm': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                'num_leaves': {'type': 'int', 'low': 15, 'high': 127},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
            },
            'catboost': {
                'iterations': {'type': 'int', 'low': 50, 'high': 500},
                'depth': {'type': 'int', 'low': 4, 'high': 10},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            },
            'knn': {
                'n_neighbors': {'type': 'int', 'low': 3, 'high': 30},
                'weights': {'type': 'categorical', 'choices': ['uniform', 'distance']},
            },
            'decision_tree': {
                'max_depth': {'type': 'int', 'low': 3, 'high': 30},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
            },
            'elastic_net': {
                'alpha': {'type': 'float', 'low': 0.001, 'high': 10, 'log': True},
                'l1_ratio': {'type': 'float', 'low': 0.0, 'high': 1.0},
            },
        }
        return spaces.get(model_name, {})

    def _to_sklearn_param_grid(self, param_space: Dict) -> Dict:
        """Convert generic param space to sklearn GridSearchCV format (max 3 values per param)."""
        grid = {}
        for name, config in param_space.items():
            if config['type'] == 'int':
                step = max(1, (config['high'] - config['low']) // 2)
                grid[name] = list(range(config['low'], config['high'] + 1, step))
            elif config['type'] == 'float':
                grid[name] = np.linspace(config['low'], config['high'], 3).tolist()
            elif config['type'] == 'categorical':
                grid[name] = config['choices']
        return grid

    # ── Sklearn methods ──────────────────────────────────────────

    @step('Grid Search CV')
    def grid_search(self, model, param_grid: Dict, X: pd.DataFrame, y: pd.Series,
                    cv: int = 5, scoring: Optional[str] = None,
                    n_jobs: int = -1) -> Dict:
        """Perform Grid Search Cross-Validation.

        Args:
            model: Untrained model instance
            param_grid: Dictionary of parameters to search
            X: Features
            y: Target
            cv: Number of CV folds
            scoring: Scoring metric
            n_jobs: Parallel jobs

        Returns:
            Dictionary with best params, score, and full results
        """
        if scoring is None:
            scoring = 'accuracy' if len(np.unique(y)) <= 20 else 'r2'

        try:
            gs = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=0)
            gs.fit(X, y)

            results = {
                'method': 'grid_search',
                'best_params': gs.best_params_,
                'best_score': float(gs.best_score_),
                'cv_results': {
                    'mean_test_score': gs.cv_results_['mean_test_score'].tolist(),
                    'std_test_score': gs.cv_results_['std_test_score'].tolist(),
                    'params': gs.cv_results_['params']
                },
                'n_trials': len(gs.cv_results_['params']),
                'best_estimator': str(type(gs.best_estimator_).__name__)
            }
            self.optimization_results['grid_search'] = results
            return results
        except Exception as e:
            return {'error': str(e)}

    @step('Random Search CV')
    def random_search(self, model, param_dist: Dict, X: pd.DataFrame, y: pd.Series,
                      n_iter: int = 20, cv: int = 5, scoring: Optional[str] = None,
                      n_jobs: int = -1) -> Dict:
        """Perform Randomized Search Cross-Validation.

        Args:
            model: Untrained model instance
            param_dist: Parameter distributions
            X: Features
            y: Target
            n_iter: Number of random combinations
            cv: Number of CV folds
            scoring: Scoring metric
            n_jobs: Parallel jobs

        Returns:
            Dictionary with best params, score
        """
        if scoring is None:
            scoring = 'accuracy' if len(np.unique(y)) <= 20 else 'r2'

        try:
            rs = RandomizedSearchCV(model, param_dist, n_iter=n_iter, cv=cv,
                                     scoring=scoring, n_jobs=n_jobs, random_state=self.random_state,
                                     verbose=0)
            rs.fit(X, y)

            results = {
                'method': 'random_search',
                'best_params': rs.best_params_,
                'best_score': float(rs.best_score_),
                'n_trials': n_iter,
                'best_estimator': str(type(rs.best_estimator_).__name__)
            }
            self.optimization_results['random_search'] = results
            return results
        except Exception as e:
            return {'error': str(e)}

    # ── Optuna ─────────────────────────────────────────────────────

    @step('Optuna Optimization')
    def optuna_optimize(self, model_class, param_space: Dict, X: pd.DataFrame,
                        y: pd.Series, n_trials: int = 50, cv: int = 5,
                        direction: str = 'maximize',
                        pruner: str = 'median',
                        timeout: Optional[int] = None) -> Dict:
        """Optimize hyperparameters with Optuna (TPE + pruning).

        Args:
            model_class: Untrained model class with set_params()
            param_space: Parameter space dict (same format as _get_default_param_space)
            X: Features
            y: Target
            n_trials: Number of optimization trials
            cv: CV folds for evaluation
            direction: 'maximize' or 'minimize'
            pruner: Pruning strategy ('median' or None)
            timeout: Timeout in seconds

        Returns:
            Dictionary with best params and study results
        """
        if not OPTUNA_AVAILABLE:
            return {'error': 'Optuna not available. Install with: pip install optuna'}

        scoring = 'accuracy' if len(np.unique(y)) <= 20 else 'r2'

        def objective(trial):
            params = {}
            for name, config in param_space.items():
                ptype = config['type']
                if ptype == 'int':
                    params[name] = trial.suggest_int(name, config['low'], config['high'])
                elif ptype == 'float':
                    log = config.get('log', False)
                    params[name] = trial.suggest_float(name, config['low'], config['high'], log=log)
                elif ptype == 'categorical':
                    params[name] = trial.suggest_categorical(name, config['choices'])

            model = model_class(**params)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return scores.mean()

        try:
            pruner_obj = MedianPruner(n_startup_trials=5, n_warmup_steps=10) if pruner == 'median' else None
            sampler = TPESampler(seed=self.random_state)

            study = optuna.create_study(
                direction=direction,
                sampler=sampler,
                pruner=pruner_obj
            )

            study.optimize(objective, n_trials=n_trials, timeout=timeout)

            results = {
                'method': 'optuna',
                'best_params': study.best_params,
                'best_score': float(study.best_value),
                'n_trials': len(study.trials),
                'best_estimator': model_class.__name__
            }

            # Log trial history
            trials_df = [{
                'number': t.number,
                'value': t.value if t.value is not None else None,
                'params': str(t.params),
                'state': str(t.state)
            } for t in study.trials]
            results['trials_df'] = trials_df

            self.optimization_results['optuna'] = results
            return results

        except Exception as e:
            return {'error': str(e)}

    # ── Hyperopt ───────────────────────────────────────────────────

    @step('Hyperopt Optimization')
    def hyperopt_optimize(self, model_class, param_space: Dict, X: pd.DataFrame,
                          y: pd.Series, max_evals: int = 50, cv: int = 5) -> Dict:
        """Optimize hyperparameters with Hyperopt (TPE).

        Args:
            model_class: Untrained model class
            param_space: Parameter space dict
            X: Features
            y: Target
            max_evals: Maximum evaluations
            cv: CV folds

        Returns:
            Dictionary with best params
        """
        if not HYPEROPT_AVAILABLE:
            return {'error': 'Hyperopt not available. Install with: pip install hyperopt'}

        scoring = 'accuracy' if len(np.unique(y)) <= 20 else 'r2'

        # Convert param space to hyperopt format
        hyperopt_space = {}
        for name, config in param_space.items():
            ptype = config['type']
            if ptype == 'int':
                hyperopt_space[name] = scope.int(hp.quniform(name, config['low'], config['high'], 1))
            elif ptype == 'float':
                if config.get('log', False):
                    hyperopt_space[name] = hp.loguniform(name, np.log(config['low']), np.log(config['high']))
                else:
                    hyperopt_space[name] = hp.uniform(name, config['low'], config['high'])
            elif ptype == 'categorical':
                hyperopt_space[name] = hp.choice(name, config['choices'])

        def objective(params):
            converted = {}
            for name, config in param_space.items():
                if config['type'] == 'categorical':
                    converted[name] = config['choices'][params[name]]
                else:
                    converted[name] = params[name]

            model = model_class(**converted)
            score = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1).mean()
            return {'loss': -score, 'status': STATUS_OK}

        try:
            trials = Trials()
            best = fmin(objective, hyperopt_space, algo=tpe.suggest,
                       max_evals=max_evals, trials=trials, rstate=np.random.RandomState(self.random_state))

            # Convert best params back
            best_params = {}
            for name, config in param_space.items():
                if config['type'] == 'categorical':
                    best_params[name] = config['choices'][best[name]]
                else:
                    best_params[name] = best[name]

            results = {
                'method': 'hyperopt',
                'best_params': best_params,
                'best_score': float(-trials.best_trial['result']['loss']),
                'n_trials': max_evals,
                'best_estimator': model_class.__name__
            }
            self.optimization_results['hyperopt'] = results
            return results

        except Exception as e:
            return {'error': str(e)}

    # ── Scikit-Optimize ──────────────────────────────────────────

    @step('Scikit-Optimize')
    def skopt_optimize(self, model_class, param_space: Dict, X: pd.DataFrame,
                       y: pd.Series, n_calls: int = 50, cv: int = 5) -> Dict:
        """Optimize hyperparameters with Scikit-Optimize (Gaussian Process).

        Args:
            model_class: Untrained model class
            param_space: Parameter space dict
            X: Features
            y: Target
            n_calls: Number of calls
            cv: CV folds
        """
        if not SKOPT_AVAILABLE:
            return {'error': 'Scikit-Optimize not available. Install with: pip install scikit-optimize'}

        scoring = 'accuracy' if len(np.unique(y)) <= 20 else 'r2'

        skopt_space = []
        param_names = []
        for name, config in param_space.items():
            param_names.append(name)
            ptype = config['type']
            if ptype == 'int':
                skopt_space.append(Integer(config['low'], config['high'], name=name))
            elif ptype == 'float':
                skopt_space.append(Real(config['low'], config['high'], name=name))
            elif ptype == 'categorical':
                skopt_space.append(Categorical(config['choices'], name=name))

        @use_named_args(skopt_space)
        def objective(**params):
            model = model_class(**params)
            score = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1).mean()
            return -score  # minimize negative score

        try:
            res = gp_minimize(objective, skopt_space, n_calls=n_calls, random_state=self.random_state)

            best_params = {param_names[i]: res.x[i] for i in range(len(param_names))}

            results = {
                'method': 'skopt',
                'best_params': best_params,
                'best_score': float(-res.fun),
                'n_trials': n_calls,
                'best_estimator': model_class.__name__
            }
            self.optimization_results['skopt'] = results
            return results

        except Exception as e:
            return {'error': str(e)}

    # ── Auto Optimize ──────────────────────────────────────────────

    @step('Auto Optimize')
    def auto_optimize(self, model_class, model_name: str, X: pd.DataFrame,
                      y: pd.Series, method: str = 'optuna',
                      n_trials: int = 50, cv: int = 5) -> Dict:
        """Automatically optimize a model with default param spaces.

        Args:
            model_class: Untrained model class
            model_name: Model identifier (e.g., 'random_forest', 'xgboost')
            X: Features
            y: Target
            method: Optimization method ('grid', 'random', 'optuna', 'hyperopt', 'skopt')
            n_trials: Number of trials (for non-grid methods)
            cv: CV folds

        Returns:
            Optimization results
        """
        param_space = self._get_default_param_space(model_name)
        if not param_space:
            return {'error': f"No default param space for '{model_name}'"}

        if method == 'grid':
            grid = self._to_sklearn_param_grid(param_space)
            model = model_class()
            return self.grid_search(model, grid, X, y, cv=cv)

        elif method == 'random':
            grid = self._to_sklearn_param_grid(param_space)
            model = model_class()
            return self.random_search(model, grid, X, y, n_iter=n_trials, cv=cv)

        elif method == 'optuna':
            return self.optuna_optimize(model_class, param_space, X, y, n_trials=n_trials, cv=cv)

        elif method == 'hyperopt':
            return self.hyperopt_optimize(model_class, param_space, X, y, max_evals=n_trials, cv=cv)

        elif method == 'skopt':
            return self.skopt_optimize(model_class, param_space, X, y, n_calls=n_trials, cv=cv)

        else:
            return {'error': f"Unknown method: {method}"}

    # ── Compare Methods ──────────────────────────────────────────

    @step('Compare Optimization Methods')
    def compare_methods(self, model_class, model_name: str, X: pd.DataFrame,
                        y: pd.Series, methods: Optional[List[str]] = None,
                        n_trials: int = 30, cv: int = 3) -> Dict:
        """Compare all available optimization methods on the same model.

        Args:
            model_class: Untrained model class
            model_name: Model identifier
            X: Features
            y: Target
            methods: Methods to compare (None = all available)
            n_trials: Trials per method
            cv: CV folds

        Returns:
            Dictionary with results for each method
        """
        if methods is None:
            methods = ['grid', 'random', 'optuna', 'hyperopt', 'skopt']

        results = {}
        for method in methods:
            try:
                result = self.auto_optimize(model_class, model_name, X, y,
                                            method=method, n_trials=n_trials, cv=cv)
                if 'error' not in result:
                    results[method] = {
                        'best_score': result.get('best_score', 'N/A'),
                        'best_params': result.get('best_params', {}),
                        'n_trials': result.get('n_trials', n_trials)
                    }
            except Exception as e:
                results[method] = {'error': str(e)}

        # Find best overall
        valid = {k: v for k, v in results.items() if 'error' not in v and isinstance(v.get('best_score'), (int, float))}
        if valid:
            best_method = max(valid, key=lambda k: valid[k]['best_score'])
            results['_best_method'] = best_method

        self.optimization_results['comparison'] = results
        return results

    # ── Persistence ──────────────────────────────────────────────

    def get_results(self) -> Dict:
        """Get all stored optimization results."""
        return self.optimization_results.copy()

    def clear_results(self):
        """Clear all stored optimization results."""
        self.optimization_results = {}

    def get_best_trial(self, method: str = 'optuna') -> Optional[Dict]:
        """Get the best trial parameters from a saved optimization.

        Args:
            method: Optimization method name

        Returns:
            Best params dict or None
        """
        results = self.optimization_results.get(method)
        if results and 'best_params' in results:
            return results['best_params']
        return None
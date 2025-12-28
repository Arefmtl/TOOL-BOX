"""
Advanced Optimization Tool - State-of-the-art hyperparameter optimization
Combines multiple optimization libraries for superior performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings
import time
from abc import ABC, abstractmethod

# Advanced optimization libraries
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Install with: pip install optuna")

try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    from hyperopt.pyll.base import scope
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    warnings.warn("Hyperopt not available. Install with: pip install hyperopt")

try:
    from skopt import gp_minimize, forest_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    warnings.warn("Scikit-optimize not available. Install with: pip install scikit-optimize")

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor


class OptimizationStrategy(ABC):
    """Abstract base class for optimization strategies"""

    @abstractmethod
    def optimize(self, objective_function, search_space, **kwargs) -> Dict[str, Any]:
        """Perform optimization"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name"""
        pass


class OptunaOptimizer(OptimizationStrategy):
    """Optuna-based optimization using TPE (Tree-structured Parzen Estimator)"""

    @property
    def name(self) -> str:
        return "optuna_tpe"

    def optimize(self, objective_function, search_space, n_trials=50, timeout=300, **kwargs) -> Dict[str, Any]:
        if not OPTUNA_AVAILABLE:
            return {"error": "Optuna not available"}

        def optuna_objective(trial):
            params = {}
            for param_name, param_config in search_space.items():
                param_type = param_config['type']
                if param_type == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
                elif param_type == 'float':
                    if 'log' in param_config and param_config['log']:
                        params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])

            return objective_function(params)

        study = optuna.create_study(direction='minimize')
        study.optimize(optuna_objective, n_trials=n_trials, timeout=timeout)

        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'optimizer': 'optuna'
        }


class HyperoptOptimizer(OptimizationStrategy):
    """Hyperopt-based optimization using TPE"""

    @property
    def name(self) -> str:
        return "hyperopt_tpe"

    def optimize(self, objective_function, search_space, max_evals=50, **kwargs) -> Dict[str, Any]:
        if not HYPEROPT_AVAILABLE:
            return {"error": "Hyperopt not available"}

        # Convert search space to hyperopt format
        hyperopt_space = {}
        for param_name, param_config in search_space.items():
            param_type = param_config['type']
            if param_type == 'int':
                hyperopt_space[param_name] = scope.int(hp.quniform(param_name, param_config['low'], param_config['high'], 1))
            elif param_type == 'float':
                if 'log' in param_config and param_config['log']:
                    hyperopt_space[param_name] = hp.loguniform(param_name, np.log(param_config['low']), np.log(param_config['high']))
                else:
                    hyperopt_space[param_name] = hp.uniform(param_name, param_config['low'], param_config['high'])
            elif param_type == 'categorical':
                hyperopt_space[param_name] = hp.choice(param_name, param_config['choices'])

        def hyperopt_objective(params):
            # Convert hyperopt params back to our format
            converted_params = {}
            for param_name, param_config in search_space.items():
                if param_config['type'] == 'categorical':
                    converted_params[param_name] = param_config['choices'][params[param_name]]
                else:
                    converted_params[param_name] = params[param_name]

            loss = objective_function(converted_params)
            return {'loss': loss, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(hyperopt_objective, hyperopt_space, algo=tpe.suggest,
                   max_evals=max_evals, trials=trials)

        # Convert best params back
        best_params = {}
        for param_name, param_config in search_space.items():
            if param_config['type'] == 'categorical':
                best_params[param_name] = param_config['choices'][best[param_name]]
            else:
                best_params[param_name] = best[param_name]

        return {
            'best_params': best_params,
            'best_value': trials.best_trial['result']['loss'],
            'n_trials': len(trials),
            'optimizer': 'hyperopt'
        }


class SkoptOptimizer(OptimizationStrategy):
    """Scikit-optimize based optimization using Gaussian Processes"""

    @property
    def name(self) -> str:
        return "skopt_gp"

    def optimize(self, objective_function, search_space, n_calls=50, **kwargs) -> Dict[str, Any]:
        if not SKOPT_AVAILABLE:
            return {"error": "Scikit-optimize not available"}

        # Convert search space to skopt format
        skopt_space = []
        param_names = []
        for param_name, param_config in search_space.items():
            param_names.append(param_name)
            param_type = param_config['type']
            if param_type == 'int':
                skopt_space.append(Integer(param_config['low'], param_config['high'], name=param_name))
            elif param_type == 'float':
                skopt_space.append(Real(param_config['low'], param_config['high'], name=param_name))
            elif param_type == 'categorical':
                skopt_space.append(Categorical(param_config['choices'], name=param_name))

        @use_named_args(skopt_space)
        def skopt_objective(**params):
            return objective_function(params)

        res = gp_minimize(skopt_objective, skopt_space, n_calls=n_calls, random_state=42)

        return {
            'best_params': {param_names[i]: res.x[i] for i in range(len(param_names))},
            'best_value': res.fun,
            'n_trials': n_calls,
            'optimizer': 'skopt_gp'
        }


class AdvancedOptimizationTool:
    """
    Advanced hyperparameter optimization tool combining multiple optimization libraries.
    Provides state-of-the-art optimization algorithms for superior performance.
    """

    def __init__(self, random_state: int = 42, cv_folds: int = 5):
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.optimizers = {
            'optuna': OptunaOptimizer(),
            'hyperopt': HyperoptOptimizer(),
            'skopt': SkoptOptimizer()
        }

    def get_model_search_spaces(self, model_name: str) -> Dict[str, Dict]:
        """
        Get predefined search spaces for common ML models

        Args:
            model_name: Name of the model ('rf', 'svm', 'xgb', 'lr', etc.)

        Returns:
            Search space dictionary
        """
        spaces = {
            'random_forest': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]}
            },
            'svm': {
                'C': {'type': 'float', 'low': 0.1, 'high': 100, 'log': True},
                'gamma': {'type': 'categorical', 'choices': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]},
                'kernel': {'type': 'categorical', 'choices': ['rbf', 'linear', 'poly', 'sigmoid']}
            },
            'xgboost': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0},
                'colsample_bytree': {'type': 'float', 'low': 0.5, 'high': 1.0},
                'reg_alpha': {'type': 'float', 'low': 0, 'high': 1, 'log': True},
                'reg_lambda': {'type': 'float', 'low': 0, 'high': 1, 'log': True}
            },
            'logistic_regression': {
                'C': {'type': 'float', 'low': 0.01, 'high': 100, 'log': True},
                'penalty': {'type': 'categorical', 'choices': ['l1', 'l2', 'elasticnet', None]},
                'solver': {'type': 'categorical', 'choices': ['liblinear', 'lbfgs', 'newton-cg', 'saga']}
            },
            'knn': {
                'n_neighbors': {'type': 'int', 'low': 1, 'high': 50},
                'weights': {'type': 'categorical', 'choices': ['uniform', 'distance']},
                'p': {'type': 'int', 'low': 1, 'high': 5}
            }
        }

        return spaces.get(model_name, {})

    def create_objective_function(self, model_class, X_train, y_train,
                                scoring='accuracy', task_type='classification'):
        """
        Create objective function for optimization

        Args:
            model_class: ML model class
            X_train: Training features
            y_train: Training target
            scoring: Scoring metric
            task_type: 'classification' or 'regression'

        Returns:
            Objective function
        """
        def objective(params):
            try:
                # Handle special cases
                if 'penalty' in params and params['penalty'] is None:
                    params_copy = params.copy()
                    params_copy.pop('penalty', None)
                    model = model_class(**params_copy)
                else:
                    model = model_class(**params)

                # Perform cross-validation
                cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

                if task_type == 'classification':
                    if scoring == 'accuracy':
                        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                    elif scoring == 'f1':
                        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
                    elif scoring == 'precision':
                        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision_macro')
                    elif scoring == 'recall':
                        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall_macro')
                    else:
                        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                else:  # regression
                    if scoring == 'neg_mean_squared_error':
                        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
                        scores = -scores  # Convert to positive MSE
                    elif scoring == 'neg_mean_absolute_error':
                        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error')
                        scores = -scores  # Convert to positive MAE
                    elif scoring == 'r2':
                        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
                        scores = -scores  # Minimize negative R²
                    else:
                        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
                        scores = -scores

                # Return mean score (to minimize)
                return -np.mean(scores) if task_type == 'classification' and scoring in ['accuracy', 'f1', 'precision', 'recall'] else np.mean(scores)

            except Exception as e:
                # Return high penalty for invalid configurations
                return 999999

        return objective

    def optimize_hyperparameters(self, model_name: str, model_class,
                               X_train: pd.DataFrame, y_train: pd.Series,
                               optimizer: str = 'optuna',
                               custom_search_space: Optional[Dict] = None,
                               task_type: str = 'auto',
                               scoring: str = 'auto',
                               n_trials: int = 50,
                               **kwargs) -> Dict[str, Any]:
        """
        Perform advanced hyperparameter optimization

        Args:
            model_name: Name of the model for predefined search space
            model_class: ML model class
            X_train: Training features
            y_train: Training target
            optimizer: Optimization algorithm ('optuna', 'hyperopt', 'skopt')
            custom_search_space: Custom search space (optional)
            task_type: 'classification' or 'regression' ('auto' for detection)
            scoring: Scoring metric ('auto' for default)
            n_trials: Number of optimization trials
            **kwargs: Additional arguments for optimizer

        Returns:
            Optimization results
        """

        # Auto-detect task type
        if task_type == 'auto':
            unique_targets = len(y_train.unique())
            task_type = 'classification' if unique_targets <= 20 else 'regression'

        # Set default scoring
        if scoring == 'auto':
            scoring = 'accuracy' if task_type == 'classification' else 'neg_mean_squared_error'

        # Get search space
        if custom_search_space:
            search_space = custom_search_space
        else:
            search_space = self.get_model_search_spaces(model_name)
            if not search_space:
                return {"error": f"No predefined search space for model: {model_name}"}

        # Create objective function
        objective = self.create_objective_function(model_class, X_train, y_train, scoring, task_type)

        # Get optimizer
        if optimizer not in self.optimizers:
            return {"error": f"Unknown optimizer: {optimizer}. Available: {list(self.optimizers.keys())}"}

        opt_instance = self.optimizers[optimizer]

        # Perform optimization
        start_time = time.time()
        result = opt_instance.optimize(objective, search_space, n_trials=n_trials, **kwargs)
        end_time = time.time()

        if 'error' in result:
            return result

        # Add additional information
        result.update({
            'optimization_time': end_time - start_time,
            'task_type': task_type,
            'scoring': scoring,
            'model_name': model_name,
            'search_space_size': len(search_space)
        })

        return result

    def compare_optimizers(self, model_name: str, model_class,
                          X_train: pd.DataFrame, y_train: pd.Series,
                          optimizers: List[str] = None,
                          task_type: str = 'auto',
                          n_trials: int = 30) -> pd.DataFrame:
        """
        Compare different optimizers on the same problem

        Args:
            model_name: Name of the model
            model_class: ML model class
            X_train: Training features
            y_train: Training target
            optimizers: List of optimizers to compare
            task_type: Task type
            n_trials: Number of trials per optimizer

        Returns:
            Comparison DataFrame
        """
        if optimizers is None:
            optimizers = list(self.optimizers.keys())

        results = []
        for opt_name in optimizers:
            print(f"Running {opt_name} optimization...")
            result = self.optimize_hyperparameters(
                model_name=model_name,
                model_class=model_class,
                X_train=X_train,
                y_train=y_train,
                optimizer=opt_name,
                task_type=task_type,
                n_trials=n_trials
            )

            if 'error' not in result:
                results.append({
                    'optimizer': opt_name,
                    'best_score': -result['best_value'] if task_type == 'classification' else result['best_value'],
                    'time': result['optimization_time'],
                    'trials': result['n_trials']
                })

        return pd.DataFrame(results)

    def get_available_optimizers(self) -> List[str]:
        """Get list of available optimizers"""
        available = []
        if OPTUNA_AVAILABLE:
            available.append('optuna')
        if HYPEROPT_AVAILABLE:
            available.append('hyperopt')
        if SKOPT_AVAILABLE:
            available.append('skopt')
        return available

    def create_custom_search_space(self, **params) -> Dict[str, Dict]:
        """
        Create custom search space from parameters

        Args:
            **params: Parameter definitions

        Returns:
            Search space dictionary
        """
        search_space = {}
        for param_name, param_config in params.items():
            if isinstance(param_config, dict):
                search_space[param_name] = param_config
            elif isinstance(param_config, (list, tuple)):
                search_space[param_name] = {'type': 'categorical', 'choices': param_config}
            elif isinstance(param_config, (int, float)):
                # Single value - create range around it
                if isinstance(param_config, int):
                    search_space[param_name] = {'type': 'int', 'low': max(1, param_config//2), 'high': param_config*2}
                else:
                    search_space[param_name] = {'type': 'float', 'low': param_config/10, 'high': param_config*10}
            else:
                raise ValueError(f"Unsupported parameter type for {param_name}")

        return search_space

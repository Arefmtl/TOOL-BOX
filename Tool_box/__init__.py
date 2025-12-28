"""
Tool Box - A comprehensive collection of machine learning and data science tools

This package contains specialized tools for:
- Data cleaning and preprocessing
- Classification and regression modeling (including XGBoost)
- Model evaluation and comparison
- Cross-validation techniques
- Hyperparameter tuning (Grid Search, Random Search, XGBoost optimization)
- Advanced optimization (Optuna, Hyperopt, Scikit-optimize)
- Feature importance analysis
- Clustering algorithms

Note: XGBoost support requires installation of the xgboost package:
pip install xgboost

Advanced optimization tools require additional packages:
pip install optuna hyperopt scikit-optimize
"""

from .data_processing_tool import DataProcessingTool
from .classification_tool import ClassificationTool
from .regression_tool import RegressionTool
from .model_evaluation_tool import ModelEvaluationTool
from .cross_validation_tool import CrossValidationTool
from .hyperparameter_tuning_tool import HyperparameterTuningTool
from .advanced_optimization_tool import AdvancedOptimizationTool
from .feature_importance_tool import FeatureImportanceTool
from .clustering_tool import ClusteringTool

__version__ = "2.0.0"
__author__ = "aref_mtl"
__description__ = "A comprehensive toolbox for machine learning and data science tasks"

# Define __all__ to control what gets imported with "from Tool_box import *"
__all__ = [
    'DataProcessingTool',
    'ClassificationTool',
    'RegressionTool',
    'ModelEvaluationTool',
    'CrossValidationTool',
    'HyperparameterTuningTool',
    'AdvancedOptimizationTool',
    'FeatureImportanceTool',
    'ClusteringTool'
]

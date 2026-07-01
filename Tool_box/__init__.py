"""
Tool Box - A comprehensive collection of machine learning and data science tools

Pipeline: Load Data → EDA → Preprocess → Split → Train → Evaluate → Optimize → Export

Tools:
- DataProcessingTool: Data loading, EDA, cleaning, encoding, scaling, splitting
- ClassificationTool: 17+ classification algorithms with parallel training
- RegressionTool: 16+ regression algorithms with parallel training
- ClusteringTool: 14+ clustering algorithms with evaluation metrics
- ModelEvaluationTool: 25+ metrics + 15+ visualization types + HTML reports
- CrossValidationTool: 8 CV methods + Nested CV + auto-scoring
- Optimizer: 5 optimization methods (Grid, Random, Optuna, Hyperopt, Skopt)
- FeatureSelector: Feature importance + SHAP + RFE + auto selection
- ModelInterpreter: SHAP, LIME, PDP for model explainability
- decorators: @step decorator for logging and timing

Requirements:
    pip install xgboost lightgbm catboost loguru joblib plotly

Optional:
    pip install optuna hyperopt scikit-optimize shap lime hdbscan scikit-fuzzy
"""

from loguru import logger
logger.add("tool_box.log", rotation="10 MB", level="INFO",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}")

from .data_processing_tool import DataProcessingTool
from .classification_tool import ClassificationTool
from .regression_tool import RegressionTool
from .clustering_tool import ClusteringTool
from .model_evaluation_tool import ModelEvaluationTool
from .cross_validation_tool import CrossValidationTool
from .optimizer import Optimizer
from .feature_selector import FeatureSelector
from .model_interpreter import ModelInterpreter
from .decorators import step

__version__ = "3.0.0"
__author__ = "aref_mtl"
__description__ = "TOOL-BOX v3.0 — Advanced ML Toolkit with 9 specialized tools"

__all__ = [
    'DataProcessingTool',
    'ClassificationTool',
    'RegressionTool',
    'ClusteringTool',
    'ModelEvaluationTool',
    'CrossValidationTool',
    'Optimizer',
    'FeatureSelector',
    'ModelInterpreter',
    'step',
    'logger'
]
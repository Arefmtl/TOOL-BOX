# Tool Box Documentation

A comprehensive collection of machine learning and data science tools for Python.

## Available Tools

### 1. Data Processing Tool (`data_processing_tool.py`)
**Purpose**: Comprehensive data cleaning and preprocessing utilities

**Functions**:
- `DataProcessingTool()`: Main class for data processing
- `load_data(file_path)`: Load data from various file formats
- `data_overview(data)`: Get comprehensive data overview
- `clean_data(data)`: Clean data with various options
- `detect_and_handle_outliers(data)`: Detect and handle outliers
- `explore_data(data)`: Exploratory data analysis with visualizations
- `encode_categorical(data)`: Encode categorical variables
- `scale_features(data)`: Scale numerical features
- `apply_pca(data)`: Apply PCA for dimensionality reduction
- `create_derived_features(data)`: Create derived features
- `prepare_data_for_ml(data)`: Complete pipeline for ML preparation
- `generate_data_report(data)`: Generate comprehensive data quality report

### 2. Classification Tool (`classification_tool.py`)
**Purpose**: Multiple classification algorithms for machine learning

**Supported Algorithms**:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- Gradient Boosting Classifier
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree Classifier
- **XGBoost Classifier** (NEW)

**Functions**:
- Individual training methods for each algorithm
- `train_multiple_models()`: Train multiple models at once (includes XGBoost)
- `predict_single_model()`: Make predictions with trained models
- `predict_proba_single_model()`: Get probability predictions
- `train_xgboost_classifier()`: Train XGBoost with advanced parameters

### 3. Regression Tool (`regression_tool.py`)
**Purpose**: Multiple regression algorithms for machine learning

**Supported Algorithms**:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Support Vector Regressor (SVR)
- Gradient Boosting Regressor
- K-Nearest Neighbors Regressor
- Decision Tree Regressor
- **XGBoost Regressor** (NEW)

**Functions**:
- Individual training methods for each algorithm
- `train_multiple_models()`: Train multiple models at once (includes XGBoost)
- `predict_single_model()`: Make predictions with trained models
- `train_xgboost_regressor()`: Train XGBoost with advanced parameters

### 4. Model Evaluation Tool (`model_evaluation_tool.py`)
**Purpose**: Comprehensive model evaluation and comparison utilities

**Functions**:
- `evaluate_classification_models()`: Evaluate multiple classification models
- `evaluate_regression_models()`: Evaluate multiple regression models
- `plot_confusion_matrix()`: Plot confusion matrices
- `plot_roc_curve()`: Plot ROC curves for binary classification
- `plot_model_comparison()`: Compare model performance visually
- `plot_residuals()`: Plot residuals for regression models
- `generate_evaluation_report()`: Create HTML evaluation reports
- `get_best_model()`: Find the best performing model

### 5. Cross Validation Tool (`cross_validation_tool.py`)
**Purpose**: Comprehensive cross-validation techniques

**Supported Methods**:
- K-Fold Cross-Validation
- Stratified K-Fold
- Time Series Split
- Repeated Cross-Validation
- Leave-One-Out
- Shuffle Split

**Functions**:
- Individual methods for each CV technique
- `compare_cv_methods()`: Compare different CV approaches
- Automatic scoring metric selection

### 6. Hyperparameter Tuning Tool (`hyperparameter_tuning_tool.py`)
**Purpose**: Automated hyperparameter optimization

**Supported Methods**:
- Grid Search Cross-Validation
- Randomized Search Cross-Validation

**Functions**:
- `grid_search_cv()`: Exhaustive grid search
- `random_search_cv()`: Randomized parameter search
- `tune_classification_model()`: Pre-configured classification tuning
- `tune_regression_model()`: Pre-configured regression tuning
- `compare_tuning_methods()`: Compare grid vs random search

### 7. Feature Importance Tool (`feature_importance_tool.py`)
**Purpose**: Analyze and visualize feature importance

**Supported Methods**:
- Tree-based importance (Random Forest, XGBoost)
- Linear model coefficients
- Permutation importance
- Univariate feature selection

**Functions**:
- `analyze_feature_importance()`: Multi-method importance analysis
- `plot_feature_importance()`: Visualize importance rankings
- `plot_importance_comparison()`: Compare different methods
- `get_top_features()`: Extract most important features
- `generate_importance_report()`: Create HTML reports

### 8. Clustering Tool (`clustering_tool.py`)
**Purpose**: Enhanced clustering algorithms with evaluation

**Supported Algorithms**:
- K-Means Clustering
- DBSCAN (Density-Based)
- Hierarchical Clustering
- Spectral Clustering
- BIRCH Clustering
- OPTICS
- Mean Shift
- Affinity Propagation
- Gaussian Mixture Models

**Functions**:
- Individual clustering methods for each algorithm
- `evaluate_clustering()`: Quality metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
- `plot_clusters_2d()`: 2D cluster visualization with PCA/t-SNE
- `compare_algorithms()`: Compare different clustering approaches

## Quick Start

### Data Processing Pipeline
```python
from Tool_box import DataProcessingTool

processor = DataProcessingTool()
data = processor.load_data("data.csv")
cleaned_data = processor.clean_data(data)
processed_data = processor.prepare_data_for_ml(cleaned_data, target_column="target")
```

### Complete ML Workflow
```python
from Tool_box import (
    ClassificationTool, ModelEvaluationTool,
    CrossValidationTool, HyperparameterTuningTool
)

# Initialize tools
classifier = ClassificationTool()
evaluator = ModelEvaluationTool()
cv_tool = CrossValidationTool()
tuner = HyperparameterTuningTool()

# Split your processed data
X_train, X_test, y_train, y_test = train_test_split(processed_data['X'], processed_data['y'])

# Train models (now includes XGBoost!)
models = classifier.train_multiple_models(X_train, y_train)

# Evaluate models
results = evaluator.evaluate_classification_models(models, X_test, y_test)
evaluator.plot_model_comparison(results)

# Cross-validate best model
cv_results = cv_tool.k_fold_cross_validation(models['xgboost'], processed_data['X'], processed_data['y'])

# Hyperparameter tuning (now supports XGBoost!)
tuned_model = tuner.tune_classification_model('xgboost', X_train, y_train)
```

### XGBoost-Specific Examples

#### Train XGBoost Classifier
```python
from Tool_box import ClassificationTool

classifier = ClassificationTool()

# Train XGBoost with default parameters
xgb_model = classifier.train_xgboost_classifier(X_train, y_train)

# Train XGBoost with custom parameters
xgb_model = classifier.train_xgboost_classifier(
    X_train, y_train,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.8
)

# Train with early stopping
eval_set = [(X_val, y_val)]
xgb_model = classifier.train_xgboost_classifier(
    X_train, y_train,
    eval_set=eval_set,
    early_stopping_rounds=10
)
```

#### Train XGBoost Regressor
```python
from Tool_box import RegressionTool

regressor = RegressionTool()

# Train XGBoost regressor
xgb_model = regressor.train_xgboost_regressor(X_train, y_train)

# Train with custom parameters
xgb_model = regressor.train_xgboost_regressor(
    X_train, y_train,
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05
)
```

#### XGBoost Hyperparameter Tuning
```python
from Tool_box import HyperparameterTuningTool

tuner = HyperparameterTuningTool()

# Grid search for XGBoost
xgb_results = tuner.tune_classification_model('xgboost', X_train, y_train, method='grid')

# Random search for XGBoost
xgb_results = tuner.tune_classification_model('xgboost', X_train, y_train, method='random')
```

## Installation

1. Clone the repository
2. Import tools directly:
```python
from Tool_box import DataProcessingTool, MLModelingTool
```

## Requirements

- pandas
- numpy
- matplotlib
- scikit-learn
- seaborn (for some visualizations)

## Usage Examples

### Basic Data Processing
```python
import pandas as pd
from Tool_box import DataProcessingTool

processor = DataProcessingTool()
data = processor.load_data("your_data.csv")
cleaned_data = processor.clean_data(data)
report = processor.generate_data_report(cleaned_data)
```

### Complete ML Pipeline
```python
from Tool_box import DataProcessingTool, MLModelingTool

# Data processing
processor = DataProcessingTool()
data = processor.load_data("data.csv")
processed_data = processor.prepare_data_for_ml(data, target_column="target")

# ML modeling
ml_tool = MLModelingTool()
models = ml_tool.train_classification_models(
    processed_data['X_train'], processed_data['y_train'])
results = ml_tool.evaluate_classification_models(
    models, processed_data['X_test'], processed_data['y_test'])
ml_tool.plot_model_comparison(results)
```

## Contributing

1. Follow Python naming conventions (snake_case)
2. Add comprehensive docstrings
3. Include error handling
4. Test your tools thoroughly

## License

This toolbox is provided as-is for educational and research purposes.

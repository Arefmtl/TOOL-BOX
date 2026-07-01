# Tool Box v3.0

A Python machine-learning toolkit for the full workflow:

```text
Load Data -> EDA -> Preprocess -> Split -> Train -> Evaluate -> Optimize -> Export
```

This README is synced with the current modules in this directory. The old `HyperparameterTuningTool` and `FeatureImportanceTool` names are no longer part of the public package API; use `Optimizer` and `FeatureSelector` instead.

## Package Exports

```python
from Tool_box import (
    DataProcessingTool,
    ClassificationTool,
    RegressionTool,
    ClusteringTool,
    ModelEvaluationTool,
    CrossValidationTool,
    Optimizer,
    FeatureSelector,
    ModelInterpreter,
    step,
)
```

Current metadata:

- Version: `3.0.0`
- Author: `aref_mtl`
- Logger: `loguru`, writing to `tool_box.log`

## Tools

### 1. DataProcessingTool

File: `data_processing_tool.py`

Purpose: load, inspect, clean, encode, scale, split, and prepare tabular data.

Main methods:

- `load_data(file_path, encoding='utf-8')`: Load `.csv`, `.xlsx`, `.xls`, or `.json`.
- `data_overview(data)`: Return shape, columns, dtypes, missing values, duplicates, memory usage, and summaries.
- `generate_eda_summary(data)`: Build an EDA summary with recommendations.
- `clean_data(data)`: Clean missing values and duplicate rows.
- `encode_categorical(data, method='one_hot', columns=None)`: Encode categorical columns.
- `scale_features(data, method='standard')`: Scale numeric columns.
- `handle_outliers(data, method='iqr', columns=None)`: Handle numeric outliers.
- `export_processed_dataset(data, output_path, format='csv')`: Export processed data.
- `feature_selection(data, target_column, method='correlation', k=10)`: Select useful features.
- `auto_clean(data, target_column=None)`: Run automatic cleaning.
- `split_data(data, target_column, test_size=0.2)`: Create train/test splits.
- `apply_pca(data, n_components=None, target_column=None)`: Apply PCA.
- `add_clustering_features(data, n_clusters=3, columns=None)`: Add cluster labels as features.
- `advanced_preprocessing(data, config)`: Run a configurable preprocessing pipeline.
- `prepare_data_for_ml(data, target_column=None, test_size=0.2, preprocessing_steps=None)`: Run clean/encode/scale and optionally return train/test splits.
- `plot_missing_values(data)`, `plot_correlations(data)`, `plot_distributions(data)`: Generate common EDA plots.

`prepare_data_for_ml()` returns `X_train`, `X_test`, `y_train`, `y_test`, and `full_processed` when a valid `target_column` is provided. Without a target it returns `processed_data`.

### 2. ClassificationTool

File: `classification_tool.py`

Purpose: train classification models individually or in parallel.

Supported model names for `train_multiple_models()`:

- `logistic_regression`
- `random_forest`
- `svm`
- `gradient_boosting`
- `knn`
- `naive_bayes`
- `decision_tree`
- `adaboost`
- `extra_trees`
- `mlp`
- `xgboost`
- `lightgbm`
- `catboost`
- `qda`
- `ridge`

Additional methods:

- `train_voting_ensemble(...)`
- `train_stacking_ensemble(...)`
- `train_all_classifiers(...)`
- `predict(model_name, X_test)`
- `predict_proba(model_name, X_test)`
- `save_model(model_name, path)`
- `load_model(path, model_name=None)`
- `save_all_models(directory)`

Optional classifiers: `xgboost`, `lightgbm`, and `catboost` require their matching packages.

### 3. RegressionTool

File: `regression_tool.py`

Purpose: train regression models individually or in parallel.

Supported model names for `train_multiple_models()`:

- `linear_regression`
- `ridge`
- `lasso`
- `elastic_net`
- `huber`
- `quantile`
- `random_forest`
- `svr`
- `gradient_boosting`
- `hist_gradient_boosting`
- `knn`
- `decision_tree`
- `extra_trees`
- `mlp`
- `xgboost`
- `lightgbm`
- `catboost`

Common methods:

- `train_all_regressors(...)`
- `predict(model_name, X_test)`
- `save_model(model_name, path)`
- `load_model(path, model_name=None)`
- `save_all_models(directory)`

Optional regressors: `xgboost`, `lightgbm`, and `catboost` require their matching packages. `poisson` and `gamma` depend on the installed `scikit-learn` version.

### 4. ClusteringTool

File: `clustering_tool.py`

Purpose: run clustering algorithms, evaluate cluster quality, compare algorithms, and create plots.

Available clustering methods:

- `kmeans_clustering(...)`
- `mini_batch_kmeans(...)`
- `dbscan_clustering(...)`
- `hdbscan_clustering(...)`
- `hierarchical_clustering(...)`
- `spectral_clustering(...)`
- `birch_clustering(...)`
- `optics_clustering(...)`
- `mean_shift_clustering(...)`
- `affinity_propagation(...)`
- `gmm_clustering(...)`
- `kmedoids_clustering(...)`
- `fuzzy_cmeans(...)`

Evaluation and plotting:

- `evaluate_clustering(labels, X, true_labels=None)`
- `compare_algorithms(X, algorithms=None, n_clusters_range=[3, 5, 7])`
- `find_optimal_k(X, max_k=10)`
- `plot_clusters_2d(X, labels, title='Cluster Visualization', save_path=None)`
- `plot_elbow(X, max_k=10, save_path=None)`
- `plot_silhouette(X, n_clusters=3, save_path=None)`

Optional clustering dependencies: `hdbscan`, `scikit-learn-extra`, and `scikit-fuzzy`.

### 5. ModelEvaluationTool

File: `model_evaluation_tool.py`

Purpose: evaluate classification, regression, and clustering results, then build plots and HTML reports.

Evaluation methods:

- `evaluate_classification_models(models, X_test, y_test, average='weighted')`
- `evaluate_regression_models(models, X_test, y_test)`
- `evaluate_clustering(labels, X, true_labels=None)`
- `get_best_model(results, metric='accuracy')`
- `generate_evaluation_summary(results, task_type='classification')`
- `generate_evaluation_report(results, task_type='classification')`
- `get_evaluation_results()`
- `clear_results()`

Classification metrics include accuracy, precision, recall, F1, balanced accuracy, Matthews correlation coefficient, Cohen kappa, log loss when probabilities are available, and ROC/PR AUC for binary classifiers.

Regression metrics include MSE, RMSE, MAE, MAPE, R2, adjusted R2, explained variance, max error, median absolute error, and RMSLE when valid.

Plotting methods:

- `plot_confusion_matrix(...)`
- `plot_roc_curve(...)`
- `plot_pr_curve(...)`
- `plot_residuals(...)`
- `plot_learning_curve(...)`
- `plot_prediction_vs_actual(...)`
- `plot_model_comparison(...)`
- `plot_error_distribution(...)`
- `plot_calibration_curve(...)`

### 6. CrossValidationTool

File: `cross_validation_tool.py`

Purpose: run common cross-validation strategies with automatic scoring support.

Methods:

- `k_fold_cross_validation(...)`
- `stratified_k_fold_cv(...)`
- `time_series_cv(...)`
- `group_kfold_cv(...)`
- `nested_cross_validation(...)`
- `compare_cv_methods(...)`
- `get_cv_results()`
- `clear_results()`

Supported strategies include K-Fold, Stratified K-Fold, Time Series Split, Group K-Fold, nested CV, and comparison across methods.

### 7. Optimizer

File: `optimizer.py`

Purpose: tune hyperparameters using classic search and optional Bayesian tools.

Methods:

- `grid_search(model, param_grid, X, y, cv=5, scoring=None)`
- `random_search(model, param_dist, X, y, n_iter=20, cv=5, scoring=None)`
- `optuna_optimize(model_class, param_space, X, y, n_trials=50, cv=5, timeout=None)`
- `hyperopt_optimize(model_class, param_space, X, y, max_evals=50, cv=5)`
- `skopt_optimize(model_class, param_space, X, y, n_calls=50, cv=5)`
- `auto_optimize(model_class, model_name, X, y, method='optuna', n_trials=50, cv=5)`
- `compare_methods(model_class, model_name, X, y, methods=None, n_trials=30, cv=3)`
- `get_results()`
- `clear_results()`
- `get_best_trial(method='optuna')`

Supported `auto_optimize()` methods: `grid`, `random`, `optuna`, `hyperopt`, and `skopt`.

Optional optimizer dependencies: `optuna`, `hyperopt`, and `scikit-optimize`.

### 8. FeatureSelector

File: `feature_selector.py`

Purpose: calculate feature importance and select features using multiple methods.

Importance methods:

- `calculate_tree_importance(model, feature_names, normalize=True)`
- `calculate_linear_importance(model, feature_names)`
- `calculate_permutation_importance(model, X, y, n_repeats=10, sample_size=None)`
- `calculate_shap_values(model, X, n_samples=100)`
- `analyze_feature_importance(model, X, y, feature_names=None, methods=None, use_shap=False)`

Selection methods:

- `univariate_feature_selection(X, y, task='classification', k=10)`
- `recursive_feature_elimination(X, y, estimator=None, n_features=None, step=1)`
- `select_from_model(X, y, estimator=None, threshold='mean', max_features=None)`
- `auto_feature_selection(X, y, method='auto', n_features=None)`

Reporting and plotting:

- `get_top_features(importance_df, n=10)`
- `generate_importance_report(importance_dict)`
- `plot_feature_importance(importance_df, top_n=20)`
- `plot_importance_comparison(importance_dict, top_n=20)`

Optional dependency: `shap`.

### 9. ModelInterpreter

File: `model_interpreter.py`

Purpose: explain trained models with SHAP, LIME, partial dependence, feature ranking, and HTML reports.

Methods:

- `plot_shap_summary(model, X, plot_type='bar', save_path=None)`
- `plot_shap_dependence(model, X, feature, interaction_feature=None, save_path=None)`
- `explain_instance_lime(model, X, instance_idx=0, num_features=10, save_path=None)`
- `plot_partial_dependence(model, X, features, save_path=None)`
- `feature_ranking(model, X, methods=None)`
- `generate_interpretation_report(model, X, y=None)`

Optional dependencies: `shap`, `lime`, and supported `scikit-learn` partial dependence APIs.

### 10. step Decorator

File: `decorators.py`

Purpose: log and time pipeline steps with `@step("Step Name")`.

## Quick Start

### Data Preparation

```python
from Tool_box import DataProcessingTool

processor = DataProcessingTool()
data = processor.load_data("data.csv")

prepared = processor.prepare_data_for_ml(
    data,
    target_column="target",
    preprocessing_steps=["clean", "encode", "scale"],
)

X_train = prepared["X_train"]
X_test = prepared["X_test"]
y_train = prepared["y_train"]
y_test = prepared["y_test"]
```

### Classification Workflow

```python
from Tool_box import ClassificationTool, ModelEvaluationTool

classifier = ClassificationTool(random_state=42)
models = classifier.train_multiple_models(
    X_train,
    y_train,
    models=["logistic_regression", "random_forest", "xgboost"],
)

evaluator = ModelEvaluationTool()
results = evaluator.evaluate_classification_models(models, X_test, y_test)
best_name = evaluator.get_best_model(results, metric="f1")
best_model = models[best_name]
```

### Regression Workflow

```python
from Tool_box import RegressionTool, ModelEvaluationTool

regressor = RegressionTool(random_state=42)
models = regressor.train_multiple_models(
    X_train,
    y_train,
    models=["linear_regression", "random_forest", "xgboost"],
)

evaluator = ModelEvaluationTool()
results = evaluator.evaluate_regression_models(models, X_test, y_test)
best_name = evaluator.get_best_model(results, metric="r2")
```

### Cross-Validation

```python
from Tool_box import CrossValidationTool

cv_tool = CrossValidationTool(random_state=42)
cv_results = cv_tool.k_fold_cross_validation(
    best_model,
    prepared["full_processed"].drop("target", axis=1),
    prepared["full_processed"]["target"],
    n_splits=5,
)
```

### Optimization

```python
from sklearn.ensemble import RandomForestClassifier
from Tool_box import Optimizer

optimizer = Optimizer(random_state=42)
result = optimizer.auto_optimize(
    RandomForestClassifier,
    model_name="random_forest",
    X=X_train,
    y=y_train,
    method="optuna",
    n_trials=30,
    cv=3,
)
```

### Feature Selection

```python
from Tool_box import FeatureSelector

selector = FeatureSelector(random_state=42)
selection = selector.auto_feature_selection(
    X_train,
    y_train,
    method="auto",
    n_features=10,
)
```

### Model Interpretation

```python
from Tool_box import ModelInterpreter

interpreter = ModelInterpreter(random_state=42)
ranking = interpreter.feature_ranking(best_model, X_train)
html_report = interpreter.generate_interpretation_report(best_model, X_train, y_train)
```

### Clustering

```python
from Tool_box import ClusteringTool

clusterer = ClusteringTool(random_state=42)
cluster_result = clusterer.kmeans_clustering(X_train, n_clusters=3)
metrics = clusterer.evaluate_clustering(cluster_result["labels"], X_train)
```

## Dependencies

Install core project requirements from the repository root:

```bash
pip install -r requirements.txt
```

Core dependencies include `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `loguru`, and `plotly`.

Advanced or optional features may require `xgboost`, `lightgbm`, `catboost`, `optuna`, `hyperopt`, `scikit-optimize`, `shap`, `lime`, `hdbscan`, `scikit-learn-extra`, and `scikit-fuzzy`.

## Notes

- Optional tools fail gracefully when their dependency is not installed.
- Batch model training uses `joblib.Parallel` with `n_jobs=-1` by default.
- Some plotting methods accept `save_path` and return a Matplotlib figure.
- Generated reports return HTML strings; write them to a file if you need a saved report.
- The public package API is defined in `__init__.py`.

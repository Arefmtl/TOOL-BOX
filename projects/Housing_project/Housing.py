# California Housing Price Prediction Project
# Using Tool_box library for comprehensive ML pipeline

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import Tool_box
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import Tool_box modules
from Tool_box import (
    DataProcessingTool, RegressionTool, ModelEvaluationTool,
    CrossValidationTool, HyperparameterTuningTool
)

# Load the data using DataProcessingTool
processor = DataProcessingTool()
data_path = os.path.join(os.path.dirname(__file__), "Dataset", "housing.csv")
data = processor.load_data(data_path)

print("Dataset shape:", data.shape)
print("Columns:", list(data.columns))
print("Missing values per column:")
print(data.isnull().sum())

# Prepare data for ML using DataProcessingTool
processed_data = processor.prepare_data_for_ml(
    data,
    target_column='median_house_value',
    test_size=0.2,
    preprocessing_steps=['clean', 'encode', 'scale']
)

X_train = processed_data['X_train']
X_test = processed_data['X_test']
y_train = processed_data['y_train']
y_test = processed_data['y_test']

# Use RegressionTool to train multiple models
regressor = RegressionTool()
models = regressor.train_multiple_models(X_train, y_train)

# Evaluate models using ModelEvaluationTool
evaluator = ModelEvaluationTool()
results = evaluator.evaluate_regression_models(models, X_test, y_test)

# Print results
print("Model Evaluation Results:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")

# Perform cross-validation using CrossValidationTool
cv_tool = CrossValidationTool()
print("\nCross-Validation Results:")
for model_name, model in models.items():
    try:
        cv_results = cv_tool.k_fold_cross_validation(
            model, X_train, y_train,
            n_splits=5, scoring='neg_mean_squared_error'
        )
        if 'error' not in cv_results:
            print(f"\n{model_name} CV Results:")
            print(f"  Mean CV Score: {-cv_results['mean_test_scores']['neg_mean_squared_error']:.4f}")
            print(f"  Std CV Score: {cv_results['std_test_scores']['neg_mean_squared_error']:.4f}")
    except Exception as e:
        print(f"CV failed for {model_name}: {str(e)}")

# Hyperparameter Tuning
print("\n" + "="*50)
print("HYPERPARAMETER TUNING")
print("="*50)

tuner = HyperparameterTuningTool()

# Tune Random Forest (best performing model)
print("\nTuning Random Forest...")
rf_tuned = tuner.tune_regression_model('random_forest', X_train, y_train, method='grid', cv=3)
if 'error' not in rf_tuned:
    print(f"Random Forest - Best Params: {rf_tuned['best_params']}")
    print(f"Random Forest - Best CV Score: {rf_tuned['best_score']:.4f}")

    # Evaluate tuned model on test set
    rf_best_model = rf_tuned['best_estimator']
    rf_tuned_pred = rf_best_model.predict(X_test)
    from sklearn.metrics import mean_squared_error
    rf_tuned_mse = mean_squared_error(y_test, rf_tuned_pred)
    rf_tuned_rmse = np.sqrt(rf_tuned_mse)
    print(f"Random Forest - Test RMSE: {rf_tuned_rmse:.4f}")
else:
    print(f"Random Forest tuning error: {rf_tuned['error']}")

# Tune XGBoost if available
print("\nTuning XGBoost...")
xgb_tuned = tuner.tune_regression_model('xgboost', X_train, y_train, method='grid', cv=3)
if 'error' not in xgb_tuned:
    print(f"XGBoost - Best Params: {xgb_tuned['best_params']}")
    print(f"XGBoost - Best CV Score: {xgb_tuned['best_score']:.4f}")

    # Evaluate tuned model on test set
    xgb_best_model = xgb_tuned['best_estimator']
    xgb_tuned_pred = xgb_best_model.predict(X_test)
    xgb_tuned_mse = mean_squared_error(y_test, xgb_tuned_pred)
    xgb_tuned_rmse = np.sqrt(xgb_tuned_mse)
    print(f"XGBoost - Test RMSE: {xgb_tuned_rmse:.4f}")
else:
    print(f"XGBoost tuning error: {xgb_tuned['error']}")

# Compare tuned models
tuned_models = {}
if 'error' not in rf_tuned:
    tuned_models['random_forest_tuned'] = {'rmse': rf_tuned_rmse, 'model': rf_best_model}
if 'error' not in xgb_tuned:
    tuned_models['xgboost_tuned'] = {'rmse': xgb_tuned_rmse, 'model': xgb_best_model}

if tuned_models:
    best_tuned_name = min(tuned_models.keys(), key=lambda k: tuned_models[k]['rmse'])
    best_tuned_rmse = tuned_models[best_tuned_name]['rmse']
    print(f"\nBest Tuned Model: {best_tuned_name} with RMSE {best_tuned_rmse:.4f}")

# Generate evaluation report
report = evaluator.generate_evaluation_report(results, task_type='regression')
print("\nEvaluation Report Generated Successfully!")

# Compare models and get best one
best_model = evaluator.get_best_model(results, metric='r2')
print(f"\nBest performing model: {best_model}")

# Check if we achieved good performance (R² > 0.7 is considered good)
if best_model in results:
    best_r2 = results[best_model]['r2']
    if best_r2 > 0.7:
        print(f"\n🎉 SUCCESS! Achieved R² = {best_r2:.1%} (target: >70%)")
    else:
        print(f"\n⚠️  Current best R²: {best_r2:.1%} - Below 70% target")

import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split

# Add parent directory to path to import Tool_box
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import Tool_box modules
from Tool_box import DataProcessingTool, RegressionTool, ModelEvaluationTool, CrossValidationTool

# Load the data using DataProcessingTool
processor = DataProcessingTool()
data_path = os.path.join(os.path.dirname(__file__), "Dataset", "heart_disease_uci.csv")
data = processor.load_data(data_path)

# Clean and preprocess the data
data = processor.clean_data(data, remove_duplicates=True, handle_missing='drop')

# Convert all column names to strings
data.columns = data.columns.astype(str)

# Drop unnecessary columns if they exist
columns_to_drop = ['Timestamp', 'Unnamed: 0']
for col in columns_to_drop:
    if col in data.columns:
        data = data.drop(columns=[col])

# Separate features and target
# Note: The original code had issues with target column creation
# Assuming 'Heart rate' is the target or we need to identify it from the dataset
# For demonstration, let's assume the last column is the target or check common names

# Check if 'Heart rate' column exists, if not, use the last column as target
if 'Heart rate' in data.columns:
    target = "Heart rate"
else:
    # Use the last column as target (common in such datasets)
    target = data.columns[-1]
    print(f"Using '{target}' as target column")

# Split features and target
X = data.drop(columns=[target])
y = data[target]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare data for ML using DataProcessingTool
processed_data = processor.prepare_data_for_ml(data, target_column=target, test_size=0.2, preprocessing_steps=['clean', 'encode', 'scale'])

# Use RegressionTool to train multiple models
regressor = RegressionTool()
models = regressor.train_multiple_models(processed_data['X_train'], processed_data['y_train'])

# Evaluate models using ModelEvaluationTool
evaluator = ModelEvaluationTool()
results = evaluator.evaluate_regression_models(models, processed_data['X_test'], processed_data['y_test'])

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
        cv_results = cv_tool.k_fold_cross_validation(model, processed_data['X_train'], processed_data['y_train'],
                                                   n_splits=10, scoring='neg_mean_squared_error')
        if 'error' not in cv_results:
            print(f"\n{model_name} CV Results:")
            print(f"  Mean CV Score: {-cv_results['mean_test_scores']['neg_mean_squared_error']:.4f}")
            print(f"  Std CV Score: {cv_results['std_test_scores']['neg_mean_squared_error']:.4f}")
    except Exception as e:
        print(f"CV failed for {model_name}: {str(e)}")

# Generate evaluation report
report = evaluator.generate_evaluation_report(results, task_type='regression')
print("\nEvaluation Report Generated Successfully!")

# Compare models and get best one
best_model = evaluator.get_best_model(results, metric='r2')
print(f"\nBest performing model: {best_model}")

import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Add parent directory to path to import Tool_box
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import Tool_box modules
from Tool_box import DataProcessingTool, ClassificationTool, ModelEvaluationTool, HyperparameterTuningTool

# Load the data using DataProcessingTool
processor = DataProcessingTool()
data_path = os.path.join(os.path.dirname(__file__), "Dataset", "diabetes.csv")
data = processor.load_data(data_path)

# Handle missing values (coded as 0 in medical columns)
medical_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in medical_columns:
    if col in data.columns:
        data[col] = data[col].replace(0, np.nan)

print("Dataset shape:", data.shape)
print("Missing values per column:")
print(data.isnull().sum())

# Advanced preprocessing with better missing value handling
# First clean and encode the data
processed_data_clean = processor.prepare_data_for_ml(
    data,
    target_column='Outcome',
    preprocessing_steps=['clean', 'encode'],
    test_size=0.3
)

X_train = processed_data_clean['X_train']
X_test = processed_data_clean['X_test']
y_train = processed_data_clean['y_train']
y_test = processed_data_clean['y_test']

# Scale only the features (X), not the target (y)
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# Ensure target is integer for classification
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Use ClassificationTool to train multiple models
classifier = ClassificationTool()
models = classifier.train_multiple_models(X_train, y_train)

# Evaluate models using ModelEvaluationTool
evaluator = ModelEvaluationTool()
results = evaluator.evaluate_classification_models(models, X_test, y_test)

# Print results
print("Model Evaluation Results:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")

# Hyperparameter Tuning
print("\n" + "="*50)
print("HYPERPARAMETER TUNING")
print("="*50)

tuner = HyperparameterTuningTool()

# Tune Random Forest (best performing model)
print("\nTuning Random Forest...")
rf_tuned = tuner.tune_classification_model('random_forest', X_train, y_train, method='grid', cv=5)
if 'error' not in rf_tuned:
    print(f"Random Forest - Best Params: {rf_tuned['best_params']}")
    print(f"Random Forest - Best CV Score: {rf_tuned['best_score']:.4f}")

    # Evaluate tuned model on test set
    rf_best_model = rf_tuned['best_estimator']
    rf_tuned_pred = rf_best_model.predict(X_test)
    rf_tuned_accuracy = np.mean(rf_tuned_pred == y_test)
    print(f"Random Forest - Test Accuracy: {rf_tuned_accuracy:.4f}")
else:
    print(f"Random Forest tuning error: {rf_tuned['error']}")

# Tune SVM
print("\nTuning SVM...")
svm_tuned = tuner.tune_classification_model('svm', X_train, y_train, method='grid', cv=5)
if 'error' not in svm_tuned:
    print(f"SVM - Best Params: {svm_tuned['best_params']}")
    print(f"SVM - Best CV Score: {svm_tuned['best_score']:.4f}")

    # Evaluate tuned model on test set
    svm_best_model = svm_tuned['best_estimator']
    svm_tuned_pred = svm_best_model.predict(X_test)
    svm_tuned_accuracy = np.mean(svm_tuned_pred == y_test)
    print(f"SVM - Test Accuracy: {svm_tuned_accuracy:.4f}")
else:
    print(f"SVM tuning error: {svm_tuned['error']}")

# Tune Logistic Regression
print("\nTuning Logistic Regression...")
lr_tuned = tuner.tune_classification_model('logistic', X_train, y_train, method='grid', cv=5)
if 'error' not in lr_tuned:
    print(f"Logistic Regression - Best Params: {lr_tuned['best_params']}")
    print(f"Logistic Regression - Best CV Score: {lr_tuned['best_score']:.4f}")

    # Evaluate tuned model on test set
    lr_best_model = lr_tuned['best_estimator']
    lr_tuned_pred = lr_best_model.predict(X_test)
    lr_tuned_accuracy = np.mean(lr_tuned_pred == y_test)
    print(f"Logistic Regression - Test Accuracy: {lr_tuned_accuracy:.4f}")
else:
    print(f"Logistic Regression tuning error: {lr_tuned['error']}")

# Find best tuned model
tuned_models = {}
if 'error' not in rf_tuned:
    tuned_models['random_forest_tuned'] = {'accuracy': rf_tuned_accuracy, 'model': rf_best_model}
if 'error' not in svm_tuned:
    tuned_models['svm_tuned'] = {'accuracy': svm_tuned_accuracy, 'model': svm_best_model}
if 'error' not in lr_tuned:
    tuned_models['logistic_tuned'] = {'accuracy': lr_tuned_accuracy, 'model': lr_best_model}

if tuned_models:
    best_tuned_name = max(tuned_models.keys(), key=lambda k: tuned_models[k]['accuracy'])
    best_tuned_accuracy = tuned_models[best_tuned_name]['accuracy']
    print(f"\nBest Tuned Model: {best_tuned_name} with accuracy {best_tuned_accuracy:.4f}")

# Ensemble Methods
print("\n" + "="*50)
print("ENSEMBLE METHODS")
print("="*50)

from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier

# Get the best individual models for ensemble
ensemble_models = []

# Add tuned models if available
if 'error' not in rf_tuned:
    ensemble_models.append(('random_forest', rf_best_model))
if 'error' not in svm_tuned:
    ensemble_models.append(('svm', svm_best_model))
if 'error' not in lr_tuned:
    ensemble_models.append(('logistic', lr_best_model))

# Add some default models
ensemble_models.extend([
    ('gradient_boosting', GradientBoostingClassifier(random_state=42)),
    ('decision_tree', DecisionTreeClassifier(random_state=42))
])

# Voting Classifier (Hard Voting - more robust)
print("\nTraining Voting Classifier...")
voting_clf = VotingClassifier(estimators=ensemble_models, voting='hard')
voting_clf.fit(X_train, y_train)
voting_pred = voting_clf.predict(X_test)
voting_accuracy = np.mean(voting_pred == y_test)
print(f"Voting Classifier Accuracy: {voting_accuracy:.4f}")

# Bagging Classifier with Random Forest
print("\nTraining Bagging Classifier...")
bagging_clf = BaggingClassifier(
    estimator=RandomForestClassifier(random_state=42),
    n_estimators=50,
    random_state=42
)
bagging_clf.fit(X_train, y_train)
bagging_pred = bagging_clf.predict(X_test)
bagging_accuracy = np.mean(bagging_pred == y_test)
print(f"Bagging Classifier Accuracy: {bagging_accuracy:.4f}")

# AdaBoost Classifier
print("\nTraining AdaBoost Classifier...")
ada_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=100,
    random_state=42
)
ada_clf.fit(X_train, y_train)
ada_pred = ada_clf.predict(X_test)
ada_accuracy = np.mean(ada_pred == y_test)
print(f"AdaBoost Classifier Accuracy: {ada_accuracy:.4f}")

# Extra Trees Classifier (often performs well on this dataset)
from sklearn.ensemble import ExtraTreesClassifier
print("\nTraining Extra Trees Classifier...")
extra_trees = ExtraTreesClassifier(n_estimators=100, random_state=42)
extra_trees.fit(X_train, y_train)
et_pred = extra_trees.predict(X_test)
et_accuracy = np.mean(et_pred == y_test)
print(f"Extra Trees Classifier Accuracy: {et_accuracy:.4f}")

# Compare ensemble methods
ensemble_results = {
    'voting': voting_accuracy,
    'bagging': bagging_accuracy,
    'adaboost': ada_accuracy,
    'extra_trees': et_accuracy
}

best_ensemble = max(ensemble_results.keys(), key=lambda k: ensemble_results[k])
best_ensemble_accuracy = ensemble_results[best_ensemble]
print(f"\nBest Ensemble Method: {best_ensemble} with accuracy {best_ensemble_accuracy:.4f}")

# Check if we achieved 80%+
if best_ensemble_accuracy >= 0.80:
    print(f"\n🎉 SUCCESS! Achieved {best_ensemble_accuracy:.1%} accuracy (target: 80%+)")
else:
    print(f"\n⚠️  Current best accuracy: {best_ensemble_accuracy:.1%} - Still below 80% target")

# Apply PCA for dimensionality reduction
# Prepare data without scaling the target for classification
processed_data = processor.prepare_data_for_ml(data, target_column='Outcome', preprocessing_steps=['clean', 'encode'])
X_processed = processed_data['X_train']
y_processed = processed_data['y_train']

# Scale only the features for PCA and apply PCA directly
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)

pca_model = PCA(n_components=3)
pca_data = pca_model.fit_transform(X_scaled)

print(f"\nPCA explained variance ratio: {pca_model.explained_variance_ratio_}")
print(f"Total explained variance: {np.sum(pca_model.explained_variance_ratio_):.4f}")

# Train best model on PCA-transformed data
best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])

# Train a new instance of the best model on PCA-transformed training data
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

model_classes = {
    'random_forest': RandomForestClassifier,
    'logistic': LogisticRegression,
    'svm': SVC,
    'gradient_boosting': GradientBoostingClassifier,
    'knn': KNeighborsClassifier,
    'naive_bayes': GaussianNB,
    'decision_tree': DecisionTreeClassifier
}

# Train the best model on PCA-transformed data
best_model_pca = model_classes[best_model_name](random_state=42)
best_model_pca.fit(pca_data, y_processed.astype(int))  # Ensure target is integer

# Use the best model to predict on PCA-transformed test data
X_test_processed = processed_data['X_test']
X_test_scaled = scaler.transform(X_test_processed)
X_test_pca = pca_model.transform(X_test_scaled)

y_pred_pca = best_model_pca.predict(X_test_pca)
pca_accuracy = np.mean(y_pred_pca == processed_data['y_test'].astype(int))
print(f"\nBest model ({best_model_name}) accuracy on PCA-transformed data: {pca_accuracy:.4f}")

# Generate evaluation report
report = evaluator.generate_evaluation_report(results, task_type='classification')
print("\nEvaluation Report Generated Successfully!")

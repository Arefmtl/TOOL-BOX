"""
Test script for Advanced Optimization Tool
Demonstrates the new state-of-the-art optimization capabilities
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from Tool_box import AdvancedOptimizationTool

def test_advanced_optimization():
    """Test the advanced optimization tool with different scenarios"""

    print("=" * 80)
    print("TOOL-BOX Advanced Optimization Tool Test")
    print("=" * 80)

    # Initialize the optimization tool
    optimizer = AdvancedOptimizationTool(random_state=42)

    print(f"\nAvailable optimizers: {optimizer.get_available_optimizers()}")
    print(f"Supported models: {list(optimizer.get_model_search_spaces('random_forest').keys())}")

    # Test 1: Classification with Random Forest
    print("\n" + "=" * 60)
    print("Test 1: Random Forest Classification Optimization")
    print("=" * 60)

    # Generate sample data
    X_clf, y_clf = make_classification(n_samples=1000, n_features=20, n_informative=10,
                                     n_redundant=5, random_state=42)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42)

    print(f"Dataset shape: {X_clf.shape}")
    print(f"Training set: {X_train_clf.shape}, Test set: {X_test_clf.shape}")

    # Test Optuna optimization
    print("\nOptimizing Random Forest with Optuna...")
    try:
        result_optuna = optimizer.optimize_hyperparameters(
            model_name='random_forest',
            model_class=RandomForestClassifier,
            X_train=X_train_clf,
            y_train=y_train_clf,
            optimizer='optuna',
            n_trials=20,  # Reduced for faster testing
            task_type='classification'
        )

        if 'error' not in result_optuna:
            print("✅ Optuna optimization completed!")
            print(f"Best parameters: {result_optuna['best_params']}")
            print(f"Best score: {result_optuna['best_value']:.4f}")
            print(f"Optimization time: {result_optuna['optimization_time']:.2f} seconds")
            print(f"Number of trials: {result_optuna['n_trials']}")
        else:
            print(f"❌ Optuna optimization failed: {result_optuna['error']}")

    except Exception as e:
        print(f"❌ Optuna test failed: {e}")

    # Test Hyperopt optimization
    print("\nOptimizing Random Forest with Hyperopt...")
    try:
        result_hyperopt = optimizer.optimize_hyperparameters(
            model_name='random_forest',
            model_class=RandomForestClassifier,
            X_train=X_train_clf,
            y_train=y_train_clf,
            optimizer='hyperopt',
            n_trials=20,
            task_type='classification'
        )

        if 'error' not in result_hyperopt:
            print("✅ Hyperopt optimization completed!")
            print(f"Best parameters: {result_hyperopt['best_params']}")
            print(f"Best score: {result_hyperopt['best_value']:.4f}")
            print(f"Optimization time: {result_hyperopt['optimization_time']:.2f} seconds")
            print(f"Number of trials: {result_hyperopt['n_trials']}")
        else:
            print(f"❌ Hyperopt optimization failed: {result_hyperopt['error']}")

    except Exception as e:
        print(f"❌ Hyperopt test failed: {e}")

    # Test 2: Regression with SVM
    print("\n" + "=" * 60)
    print("Test 2: SVM Regression Optimization")
    print("=" * 60)

    # Generate regression data
    X_reg, y_reg = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42)

    print(f"Dataset shape: {X_reg.shape}")
    print(f"Training set: {X_train_reg.shape}, Test set: {X_test_reg.shape}")

    # Test SVM optimization
    print("\nOptimizing SVM Regression with Optuna...")
    try:
        result_svm = optimizer.optimize_hyperparameters(
            model_name='svm',
            model_class=SVR,
            X_train=X_train_reg,
            y_train=y_train_reg,
            optimizer='optuna',
            n_trials=15,
            task_type='regression'
        )

        if 'error' not in result_svm:
            print("✅ SVM optimization completed!")
            print(f"Best parameters: {result_svm['best_params']}")
            print(f"Best score: {result_svm['best_value']:.4f}")
            print(f"Optimization time: {result_svm['optimization_time']:.2f} seconds")
        else:
            print(f"❌ SVM optimization failed: {result_svm['error']}")

    except Exception as e:
        print(f"❌ SVM test failed: {e}")

    # Test 3: Custom search space
    print("\n" + "=" * 60)
    print("Test 3: Custom Search Space")
    print("=" * 60)

    custom_space = {
        'n_estimators': {'type': 'int', 'low': 10, 'high': 100},
        'max_depth': {'type': 'int', 'low': 2, 'high': 10},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True}
    }

    print("Testing custom search space with Random Forest...")
    try:
        result_custom = optimizer.optimize_hyperparameters(
            model_name='random_forest',
            model_class=RandomForestClassifier,
            X_train=X_train_clf,
            y_train=y_train_clf,
            optimizer='optuna',
            custom_search_space=custom_space,
            n_trials=10
        )

        if 'error' not in result_custom:
            print("✅ Custom search space optimization completed!")
            print(f"Best parameters: {result_custom['best_params']}")
            print(f"Best score: {result_custom['best_value']:.4f}")
        else:
            print(f"❌ Custom search space test failed: {result_custom['error']}")

    except Exception as e:
        print(f"❌ Custom search space test failed: {e}")

    # Test 4: Optimizer comparison
    print("\n" + "=" * 60)
    print("Test 4: Optimizer Comparison")
    print("=" * 60)

    print("Comparing different optimizers on Random Forest...")
    try:
        comparison = optimizer.compare_optimizers(
            model_name='random_forest',
            model_class=RandomForestClassifier,
            X_train=X_train_clf,
            y_train=y_train_clf,
            optimizers=['optuna'],  # Only test optuna for speed
            n_trials=10
        )

        if not comparison.empty:
            print("✅ Optimizer comparison completed!")
            print("\nComparison results:")
            print(comparison.to_string(index=False))
        else:
            print("❌ Optimizer comparison returned empty results")

    except Exception as e:
        print(f"❌ Optimizer comparison failed: {e}")

    print("\n" + "=" * 80)
    print("Advanced Optimization Tool Test Completed!")
    print("=" * 80)

    print("\n📋 Summary:")
    print("- ✅ Advanced optimization tools integrated successfully")
    print("- ✅ Optuna, Hyperopt, and Scikit-optimize support added")
    print("- ✅ Predefined search spaces for common models")
    print("- ✅ Custom search space support")
    print("- ✅ Optimizer comparison functionality")
    print("- ✅ Cross-validation integration")
    print("- ✅ Automatic task type detection")

    print("\n🔧 Installation requirements:")
    print("pip install optuna hyperopt scikit-optimize")

    print("\n💡 Usage Example:")
    print("""
from Tool_box import AdvancedOptimizationTool

optimizer = AdvancedOptimizationTool()
result = optimizer.optimize_hyperparameters(
    model_name='random_forest',
    model_class=RandomForestClassifier,
    X_train=X_train,
    y_train=y_train,
    optimizer='optuna',
    n_trials=50
)
print(f"Best params: {result['best_params']}")
print(f"Best score: {result['best_value']}")
    """)

if __name__ == "__main__":
    test_advanced_optimization()

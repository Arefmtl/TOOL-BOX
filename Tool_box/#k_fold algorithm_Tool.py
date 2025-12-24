from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

def load_data(file_path):
    """Loads data from an Excel file."""
    return pd.read_excel(file_path)

def preprocess_data(data, target_column, categorical_columns):
    """Preprocesses data, handling missing values and scaling/encoding features."""
    numerical_columns = data.select_dtypes(include=['number']).columns
    categorical_columns = [col for col in categorical_columns if col in data.columns]

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed, y

def train_and_evaluate_model(X, y, k=5):
    """Trains and evaluates a linear regression model using K-fold cross-validation."""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    rmse_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        rmse_scores.append(rmse)
        r2_scores.append(r2)

    return model, np.mean(rmse_scores), np.mean(r2_scores)

def main():
    file_path = input("Enter file path: ")
    target_column = input("Enter target column name: ")
    categorical_columns = input("Enter space-separated categorical column names (if any): ")
    categorical_columns = categorical_columns.split() if categorical_columns else []

    data = load_data(file_path)
    X, y = preprocess_data(data, target_column, categorical_columns)

    model, rmse, r2 = train_and_evaluate_model(X, y)

    print(f"Average RMSE: {rmse:.4f}")
    print(f"Average R-squared: {r2:.4f}")

if __name__ == "__main__":
    main()

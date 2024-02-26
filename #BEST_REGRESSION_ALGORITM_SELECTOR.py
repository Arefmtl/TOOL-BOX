import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
file_path = r"C:\Users\ASUS\Desktop\code\karlancer\DATASETS\heart.xlsx"

if file_path.endswith('.csv'):
    data = pd.read_csv(file_path, header=0, encoding='latin-1')
else:
    data = pd.read_excel(file_path, header=0)

data = data.dropna(axis=1)

# Convert feature names to strings
data.columns = data.columns.astype(str)

# Drop the timestamp column temporarily
timestamp_column = data['Timestamp']
data = data.drop(columns=['Timestamp'])

# Scale the data
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

Outcomet = "Heart rate"
X = data_scaled.drop(columns=[Outcomet])
y = data_scaled[Outcomet]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

imputer = SimpleImputer(strategy="median")
x_train = pd.DataFrame(imputer.fit_transform(x_train), columns=x_train.columns)

# Adding a new column
new_column = pd.DataFrame({'Heart rate': [100]}, index=pd.Index([0], name='Index'))
data_scaled = pd.concat([data_scaled, new_column], ignore_index=True)


# Run for each model
def train_and_evaluate(model_name, model_class, x_train, y_train, x_test, y_test):
    model = model_class()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return mse, rmse, r2

# Define models and their respective classes
models = {
    "Linear Regression": LinearRegression,
    "Ridge Regression": Ridge,
    "Lasso Regression": Lasso,
    "ElasticNet Regression": ElasticNet,
    "SVR": SVR,
    "Decision Tree Regression": DecisionTreeRegressor,
    "Random Forest Regression": RandomForestRegressor,
    "Gradient Boosting Regression": GradientBoostingRegressor,
    "KNN Regression": KNeighborsRegressor
}

# Evaluate each model
mse_results = {}
rmse_results = {}
r2_results = {}

for model_name, model_class in models.items():
    mse, rmse, r2 = train_and_evaluate(model_name, model_class, x_train, y_train, x_test, y_test)
    mse_results[model_name] = mse
    rmse_results[model_name] = rmse
    r2_results[model_name] = r2

# Print results
print("Results:")
for model_name in mse_results:
    print(f"{model_name}:")
    print(f"MSE: {mse_results[model_name]:.2f}")
    print(f"RMSE: {rmse_results[model_name]:.2f}")
    print(f"R-squared: {r2_results[model_name]:.2f}")
    print()

# Determine the best model based on the metrics
best_model = min(mse_results, key=mse_results.get)
print(f"Best model based on MSE: {best_model}")

best_model = min(rmse_results, key=rmse_results.get)
print(f"Best model based on RMSE: {best_model}")

best_model = max(r2_results, key=r2_results.get)
print(f"Best model based on R-squared: {best_model}")

# Concatenate the timestamp column back to the scaled data
data_scaled['Timestamp'] = timestamp_column

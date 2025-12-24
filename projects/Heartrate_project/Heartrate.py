import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

# Loading the data
file_path = r"C:\Users\ASUS\Desktop\code\karlancer\DATASETS\heart.xlsx"

# Loading the data
if file_path.endswith('.csv'):
    data = pd.read_csv(file_path, header=0, encoding='latin-1')
else:
    data = pd.read_excel(file_path, header=0)

# Dropping the Timestamp column and renaming columns
data = data.dropna(axis=1)
data.drop(columns=['Timestamp'], inplace=True)

data.drop('Unnamed: 0', axis=1, inplace=True)

# Checking and processing missing values
missing_values = data.isna().sum()


# Load your data into the 'data' DataFrame

# Convert all column names to strings
data.columns = data.columns.astype(str)

# Scale the data
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)


# Adding a new feature to the data
new_column = pd.DataFrame({'Heart rate': [100]}, index=pd.Index([0], name='Index'))
data_scaled = pd.concat([data_scaled, new_column], ignore_index=True)

# Separating input and output features
target = "Heart rate"
X = data_scaled.drop(columns=[target])
y = data_scaled[target]

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handling missing values using mean imputation
imputer = SimpleImputer(strategy="mean")
X_train = pd.DataFrame(imputer.fit_transform(x_train), columns=x_train.columns)

# Defining a function for training and evaluating the model
def train_and_evaluate(model_name, model_class, X_train, y_train, X_test, y_test):
    n = 10  # Number of iterations
    mse_list, rmse_list, r2_list = [], [], []
    for _ in range(n):
        model = model_class()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)

    mean_r2 = np.mean(r2_list)

    print(f"Model: {model_name}:")
    print(f"Mean MSE: {np.mean(mse_list):.2f}")
    print(f"Mean RMSE: {np.mean(rmse_list):.2f}")
    print(f"Mean R-squared: {mean_r2:.2f}")

# Training and evaluating the model for each model
train_and_evaluate("Random Forest", RandomForestRegressor, X_train, y_train, x_test, y_test)
train_and_evaluate("Gradient Boosting", GradientBoostingRegressor, X_train, y_train, x_test, y_test)
train_and_evaluate("Support Vector Machine", SVR, X_train, y_train, x_test, y_test)
train_and_evaluate("KNN", KNeighborsRegressor, X_train, y_train, x_test, y_test)

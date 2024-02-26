import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

# Read data from file
file_path = "C:\\Users\\ASUS\\Desktop\\code\\karlancer\\DATASETS\\Stay free csv.csv"

if file_path.endswith('.csv'):
    data = pd.read_csv(file_path, header=0, encoding='latin-1')
else:
    data = pd.read_excel(file_path, header=0)


# Drop non-essential columns
data.drop(['Unnamed: 0'], axis=1, inplace=True)
data.drop(['Unnamed: 0'], axis=0, inplace=True)





# Convert non-numeric values to NaN and impute missing values
data = data.apply(pd.to_numeric, errors='coerce')
imputer = SimpleImputer(strategy='mean')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Scale the data
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Add a new column 'Heart rate' with a constant value of 100
new_column = pd.DataFrame({'Heart rate': [100]}, index=pd.Index([0], name='Index'))
data_scaled = pd.concat([data_scaled, new_column], ignore_index=True)

# Define features (X) and target variable (y)
target = "Heart rate"
X = data_scaled.drop(columns=[target])
y = data_scaled[target]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values in training set
imputer = SimpleImputer(strategy="mean")
X_train = pd.DataFrame(imputer.fit_transform(x_train), columns=x_train.columns)


# Define a function to train and evaluate the model
def train_and_evaluate(model_name, model_class, X_train, y_train, X_test, y_test):
    n = int(input("How many times do you want to run the code? "))
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
    print(f"{model_name} Model:")
    print(f"Mean MSE: {np.mean(mse_list):.2f}")
    print(f"Mean RMSE: {np.mean(rmse_list):.2f}")
    print(f"Mean R-squared: {mean_r2:.2f}")


# Train and evaluate Random Forest model
train_and_evaluate("Random Forest", RandomForestRegressor, X_train, y_train, x_test, y_test)

# Train and evaluate KNN model
train_and_evaluate("KNN", KNeighborsRegressor, X_train, y_train, x_test, y_test)

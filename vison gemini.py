import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

#################################################################################################################

# بارگیری داده
# File path
file_path = "C:\\Users\\ASUS\\Desktop\\code\\TOOL BOX\\heart_rate.xlsx"

# Check file extension to determine the file format
if file_path.endswith('.csv'):
    # Read CSV file
    data = pd.read_csv(file_path, header=0, encoding='latin-1')
else:
    # Read Excel file
    data = pd.read_excel(file_path, header=0)

data.rename(index=data.Timestamp, inplace=True)
data.drop('Timestamp', axis=1, inplace=True)



# بررسی مقادیر ناپذیرفته
data.isna().sum()

# انتقال به داده‌های عددی
data.columns = data.columns.astype(str)





# مقیاس‌بندی ویژگی‌ها
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# اضافه کردن ویژگی جدید
new_column = pd.DataFrame({'Heart rate': [100]}, index=pd.Index([0], name='Index'))
data_scaled = pd.concat([data_scaled, new_column], ignore_index=True)

# جدا کردن ویژگی‌های ورودی و خروجی
target = "Heart rate"
X = data_scaled.drop(columns=[target])
y = data_scaled[target]

# جداسازی داده‌های آموزش و آزمون
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy="mean")
X_train = pd.DataFrame(imputer.fit_transform(x_train), columns=x_train.columns)



# Assuming you have your data prepared in X and y variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train and evaluate models
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

    # Calculate mean accuracy (assuming R-squared is appropriate for your task)
    mean_r2 = np.mean(r2_list)

    print(f"{model_name} Model:")
    print(f"Mean MSE: {np.mean(mse_list):.2f}")
    print(f"Mean RMSE: {np.mean(rmse_list):.2f}")
    print(f"Mean R-squared: {mean_r2:.2f}")

# Run for both models
train_and_evaluate("Random Forest", RandomForestRegressor, X_train, y_train, X_test, y_test)
train_and_evaluate("KNN", KNeighborsRegressor, X_train, y_train, X_test, y_test)





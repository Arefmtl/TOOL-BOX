# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Loading data
data = pd.read_excel(r"C:\Users\ASUS\Desktop\code\karlancer\DATASETS\heart.xlsx", header=0, names=['name0', 'name1', ...])
data_shuffled = data.sample(frac=1, random_state=42)

# Loading CSV datasets
data1 = pd.read_csv(r"C:\Users\ASUS\Desktop\code\karlancer\clustering\maincode\MAIN\data\uscities.csv")
data2 = pd.read_csv(r"C:\Users\ASUS\Desktop\code\karlancer\clustering\maincode\MAIN\data\encoded_twitter_dataset.csv")

# Concatenating DataFrames vertically
concatenated_data = pd.concat([data1, data2])

# Saving the concatenated DataFrame to a new CSV file
concatenated_data.to_csv('concatenated_dataset.csv', index=False)

# Exploring data
data.head(5)  # Equivalent to data[0:6]
data.info()
data.columns
data['population'].unique()
data['Heart rate'].value_counts()
data[data['ocean_proximity'] == 'ISLAND']
data[[data['population'], [data['ocean_proximity'] == 'california']]]

# Plotting
data.hist(bins=20, figsize=(20, 15))
plt.show()

# Train set and test set
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
train_set.shape
train_set.head()
data = train_set.copy()
data.head()
data.plot(kind="scatter", x="Heart rate", y=0, s=data[80], label=50, c=data['Heart rate'], 
        cmap=plt.get_cmap('jet'),figsize=(150, 90), alpha=0.2)
data.shape

# Standard correlation coefficient [-1, 1]
corr_matrix = data.corr()
corr_matrix['Heart rate'].sort_values(ascending=False)

# Plotting correlation
features = ['Heart rate', 'Timestamp', 48, 46, 25, 52]
scatter_matrix(data[features], figsize=(20, 15))
plt.show()

# Making useful data
data['total_rooms_per_households'] = data['total_rooms'] / data['households']
data['total_bedrooms_per_total_rooms'] = data['total_bedrooms'] / data['total_rooms']
data['population_per_households'] = data['population'] / data['households']
data.head()

# Plotting new correlation plot
corr_matrix = data.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)
features = ['median_house_value', 'median_income', 'total_rooms', 'data_median_age']
scatter_matrix(data[features], figsize=(20, 15))
plt.show()

# Cleaning data
data = train_set.copy()
data_label = data['Heart rate'].copy()
data = data.drop("Heart rate", axis=1)
data_num = data.drop("ocean_proximity", axis=1)
data.info()

# Imputing missing data
data_num = data_num.dropna(subset=['total_bedrooms'])
# or
# f_num = data_num.drop('total_bedrooms', axis=1)
# median = data_num['total_rooms'].median()
# f_num['total_rooms'].fillna(median, inplace=True)

# Adding custom attributes
room_ix, bedroom_ix, population_ix, household_ix = 3, 4, 5, 6


class AddCustomAttribute(X, y):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, room_ix] / X[:, household_ix]
        bedrooms_per_room = X[:, bedroom_ix] / X[:, room_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]


custom = AddCustomAttribute()
data_custom_tr_tmp = custom.transform(data_num_impute_tr.values)
data_custom_tr = pd.DataFrame(data_custom_tr_tmp)

# Feature scaling (standardization and normalization)
data['Timestamp'] = pd.to_numeric(data['Timestamp'])
feature_scaler = StandardScaler()
fdata = pd.DataFrame(feature_scaler.fit_transform(data.values), columns=data.columns)
fdata.head()

# Label encoding or one-hot encoding
encoder = LabelEncoder()
data_cat = data["ocean_proximity"]
data_cat_encoded = encoder.fit_transform(data_cat)
data_cat_encoded = pd.DataFrame(data_cat_encoded, columns=['ocean_proximity'])
data_cat_encoded.head()

# One-hot encoding
encoder_1hot = OneHotEncoder(sparse=False)
data_cat_1hot_tmp = encoder_1hot.fit_transform(data[['ocean_proximity']])
data_cat_1hot = pd.DataFrame(data_cat_1hot_tmp)
feature_names_out = encoder_1hot.get_feature_names_out(input_features=['ocean_proximity'])
data_cat_1hot.columns = feature_names_out
final = pd.concat([data_num_scaled_tr, data_cat_1hot], axis=1)
final.head(10)

# Pipeline for numerical and categorical data processing
num_pipeline = Pipeline([
    ('selector', AddCustomAttribute(num_attrs)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
    ('std_scaler', StandardScaler())
])
cat_attrs = ["ocean_proximity"]
cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_attrs)), ('one_hot_encoder', OneHotEncoder(sparse=False)),])
full_pipeline = ColumnTransformer([
('num', num_pipeline, num_attrs),
('cat', cat_pipeline, cat_attrs)])

# Training a model (e.g., Linear Regression)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(data_prepared, data_labels)

# Making predictions
sample_data_prepared = data_prepared[:4]
predictions = lin_reg.predict(sample_data_prepared)
print('Predictions:\t', predictions)

# Actual labels from the sample
sample_labels = data_labels.iloc[:4]
print('Labels:\t\t', list(sample_labels))

# Calculating the accuracy of predictions
accuracy = np.mean(np.abs(predictions - sample_labels) / sample_labels)
print('Accuracy:\t', accuracy)

# Calculating root mean square error (RMSE)
from sklearn.metrics import mean_squared_error

data_prediction = lin_reg.predict(data_prepared_data)
lin_mse = mean_squared_error(data_labels, data_prediction)
lin_rmse = np.sqrt(lin_mse)
print('Root Mean Square Error (RMSE):', lin_rmse)

# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
scores = cross_val_score(tree_reg, data_prepared_data, data_label, scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-scores)


def display_scores(scores, model_name):
    print('==========', model_name, '==========')
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard Deviation:', scores.std())
    print('==================================')


display_scores(tree_rmse_scores, 'Decision Tree Regression')

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_scores = cross_val_score(forest_reg, data_prepared_data, data_label, scoring='neg_mean_squared_error', cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores, 'Random Forest Regression')

# Fine-tuning the model (Random Forest) using GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = [{'n_estimators': [3, 4, 6, 10, 30], 'max_features': [2, 6, 8, 15]}]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(data_prepared_data, data_label)
print("Best Parameters:", grid_search.best_params_)
print("Best Estimator:", grid_search.best_estimator_)
results = grid_search.cv_results_
for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print(np.sqrt(-mean_score), params)

# Evaluating the final model on the test set
final_model = grid_search.best_estimator_
X = test_set.drop("median_house_value", axis=1)
y = test_set["median_house_value"].copy()
X_prepared = full_pipeline.transform(X)
final_predictions = final_model.predict(X_prepared)
final_mse = mean_squared_error(y, final_predictions)
final_rmse = np.sqrt(final_mse)
print('Final RMSE:', final_rmse)

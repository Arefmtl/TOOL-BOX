import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Reading and shuffling data
housing = pd.read_csv(r"C:\Users\ASUS\Desktop\code\visionacademy\housing.csv")
housing_shuffled = housing.sample(frac=1, random_state=42)
housing.info()
housing.columns
housing['ocean_proximity'].unique()
housing['ocean_proximity'].value_counts()
housing[housing['ocean_proximity'] == 'ISLAND']
housing.describe()

# Plotting
housing.hist(bins=50, figsize=(20, 20))
plt.show()

# Train set and test set
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# Standard correlation coefficient
corr_matrix = train_set.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

features = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(train_set[features], figsize=(20, 15))
plt.show()

# Making useful data
train_set['total_rooms_per_households'] = train_set['total_rooms'] / train_set['households']
train_set['total_bedrooms_per_total_rooms'] = train_set['total_bedrooms'] / train_set['total_rooms']
train_set['population_per_households'] = train_set['population'] / train_set['households']

# Cleaning data
df = train_set.drop("median_house_value", axis=1)
df_num = df.drop("ocean_proximity", axis=1)
df_num = df_num.dropna(subset=['total_bedrooms'])

# Adding custom attribute
room_ix, bedroom_ix, population_ix, household_ix = 3, 4, 5, 6


class AddUsefulAttributes(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, room_ix] / X[:, household_ix]
        bedrooms_per_room = X[:, bedroom_ix] / X[:, room_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]


# Feature scaling
num_pipeline = Pipeline([
    ('attribs_adder', AddUsefulAttributes()),
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

# Label encoding or one-hot encoding
encoder = OneHotEncoder()
data_cat_encoded = encoder.fit_transform(df[['ocean_proximity']])

# Final preprocessing pipeline
num_attrs = list(df_num)
cat_attrs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attrs),
    ("cat", OneHotEncoder(), cat_attrs),
])

housing_prepared = full_pipeline.fit_transform(df)

# Linear regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, train_set['median_house_value'])

# Making predictions
sample_data_prepared = housing_prepared[:4]
predictions = lin_reg.predict(sample_data_prepared)
print('Predictions:\t', predictions)

# Root Mean Square Error
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(train_set['median_house_value'], housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print('RMSE:', lin_rmse)

# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

# Cross-validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, train_set['median_house_value'],
                         scoring='neg_mean_squared_error', cv=10)
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
forest_reg.fit(housing_prepared, train_set['median_house_value'])
forest_scores = cross_val_score(forest_reg, housing_prepared, train_set['median_house_value'],
                                scoring='neg_mean_squared_error', cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores, 'Random Forest Regression')

# Grid Search
param_grid = [{'n_estimators': [3, 4, 6, 10, 30], 'max_features': [2, 6, 8, 15]}]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, train_set['median_house_value'])

print("Best Parameters:", grid_search.best_params_)
print("Best Estimator:", grid_search.best_estimator_)

results = grid_search.cv_results_
for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print(np.sqrt(-mean_score), params)

# Final Model Evaluation
final_model = grid_search.best_estimator_
X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print('Final RMSE:', final_rmse)


########################################################################################################################
# Option 1: Drop rows with missing values in the '....' column
#data_num = data.dropna(subdata=["name of column with missing values"])
########################################################################################################################
# Option 2: Fill missing values in 'missing_value_columns' column with median value
#data_num = data.drop('coulmn_name', axis=1)  # Drop the whole column
#median = data['coulmn_name'].median()  # Calculate median of 'missing_value_columns'
#data_num['coulmn_name'].fillna(median, inplace=True)  # Fill missing values with median
########################################################################################################################
#import seaborn for ploting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder #for encoding string data 1hot is better
from sklearn.preprocessing import OneHotEncoder
#from sklearn.pipeline import Pipeline
#from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import StandardScaler
#from sklearn.compose import ColumnTransformer


########################################################################################################################
def clean_data(data):
    # Remove duplicates
    data = data.drop_duplicates()
    # Option 1: Drop rows with missing values in the '....' column
    data_num = data.dropna(subdata=["name of column with missing values"])

    # Option 2: Fill missing values in 'missing_value_columns' column with median value
    data_num = data.drop('coulmn_name', axis=1)  # Drop the whole column
    median = data['coulmn_name'].median()  # Calculate median of 'missing_value_columns'
    data_num['coulmn_name'].fillna(median, inplace=True)  # Fill missing values with median
    
    #Impute missing values using SimpleImputer
    imputer = SimpleImputer(missing_values = np.nan, strategy="median")  # or strategy="most_frequent"
    imputer.fit(data_num)
    X = imputer.transform(data_num)
    data_num_impute_tr = pd.DataFrame(X, columns=data_num.columns)

    # Convert numerical columns to integer
    #label encoding problem mybe make relation between encoded nums if neue encoded nums was too many
    encoder = LabelEncoder()
    data_cat = data["none numerical data"]
    data_cat_encoded = encoder.fit_transform(data_cat)
    data_cat_encoded = pd.DataFrame(data_cat_encoded, columns=['none numerical data'])
    data_cat_encoded.head()
    
    #1hot encodinng alternative for last labeling model
    #if coulmns not so many otherweise use label encode
    # Initialize OneHotEncoder
    encoder1hot = OneHotEncoder(sparse=False)

    # Transform categorical variable 'none numerical data' using OneHotEncoder
    data_cat_1hot_tmp = encoder1hot.fit_transform(data[['none numerical data']])

    # Convert the transformed data into a DataFrame
    data_cat_1hot = pd.DataFrame(data_cat_1hot_tmp)

    # data column names for the one-hot encoded features
    feature_names_out = encoder1hot.get_feature_names_out(input_features=['none numerical data'])
    data_cat_1hot.columns = feature_names_out

    # Concatenate the one-hot encoded features with the scaled numerical features
    final = pd.concat([data_num_scaled_tr, data_cat_1hot], axis=1)

    # Display the first 10 rows of the final DataFrame
    final.head(10)


    # Convert date columns to datetime format
    date_columns = ['date_column_1', 'date_column_2']
    for column in date_columns:
        data[column] = pd.to_datetime(data[column])

    # Convert categorical columns to lowercase
    categorical_columns = ['categorical_column_1', 'categorical_column_2']
    for column in categorical_columns:
        data[column] = data[column].str.lower()

    # Remove outliers using z-score
    numeric_columns = data.select_dtypes(include=['number']).columns
    for column in numeric_columns:
        z_scores = (data[column] - data[column].mean()) / data[column].std()
        data = data[(z_scores.abs() < 3)]

    # Redata index
    data = data.redata_index(drop=True)
    
    
    #feature scaling standardizaion and normalization [0, 1]#(for neural network)
    # Assuming data_custom_tr is your DataFrame containing numerical features

    # Initialize the StandardScaler
    feature_scaler = StandardScaler()

    # Scale the numerical features
    data_num_scaled_tr = pd.DataFrame(feature_scaler.fit_transform(data_custom_tr.values), columns=data_custom_tr.columns)

    # Display the scaled DataFrame
    data_num_scaled_tr.head()

    return data
#####################################################################################################################
# Read the CSV file# shuffling data
file_path = 'your_file_path.csv'
data = pd.read_csv(file_path)
#####################################################################################################################
#Observing data
def data_observing(data):
    data.info()
    data.columns
    data['some coulmn'].unique()
    print(data.columns)
    data['data'].value_counts()
    data.describe()

    #Plortting#histogram
    data.hist(bins=50, figsize=(20, 20))
    plt.show()
    
    #Splitting cdata into Train_cdata and test_cdata
    train_cdata, test_cdata = train_test_split(train_cdata, test_size=0.2, random_state=42)

    # Plotting #feature = F
    features = ['F1', 'F2', 'F3', 'F4']
    scatter_matrix(train_cdata[features], figsize=(20, 15))
    plt.show()

    # Correlation
    #Standard_correlation_coefficient[1, -1]
    corr_matrix = train_cdata.corr()
    corr_matrix['ranking'].sort_values(ascending=False)
    cdata.head()
    
    return
#####################################################################################################################


#####################################################################################################################
def make_high_correlation(data):

    class add_makedup_attribute(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
    
    #feature = F #newfeature = NF
    def transform(self, X, y=None):
        NF1 = X[:, F1] / X[:, F2]
        NF2 = X[:, F3] / X[:, F1]
        NF3 = X[:, F4] / X[:, F2]
        return np.c_[X, NF1, NF2, NF3]

    custom = add_makedup_attribute()
    data_custom_tr_tmp = custom.transform(df_num_impute_tr.values)
    data_custom_tr = pd.DataFrame(data_custom_tr_tmp)

    # Define 'columns' variable
    columns = df_num.columns.tolist()
    columns.extend(["NF1", "NF2", "NF3"])

    data_custom_tr.columns = columns
    data_custom_tr.head(10)

        # No fitting necessary for this transformer
    return self
#####################################################################################################################
  

    # Encoding categorical data using LabelEncoder
    encoder = LabelEncoder()
    data_cat = data['#none numerical data coulmns']
    data_cat_encoded = encoder.fit_transform(data_cat)
    data_cat_encoded = pd.DataFrame(data_cat_encoded, columns=['#none numerical data coulmns'])

    # Scaling numerical features using StandardScaler
    feature_scaler = StandardScaler()
    data_num_scaled_tr = pd.DataFrame(feature_scaler.fit_transform(data_custom_tr.values), columns=data_custom_tr.columns)
    #check new data corr # Correlation
    #Standard_correlation_coefficient[1, -1]
    corr_matrix = train_cdata.corr()
    corr_matrix['ranking'].sort_values(ascending=False)
    cdata.head()


    
# Clean the data cleaned_date = cdata
cdata = clean_data(data)
# shuffling data
cdata = cdata.sample(frac=1, random_state=42)
# Display the cleaned data
print(cdata.head())

#F= feature
cdata.info()
cdata.columns
cdata['F'].unique()
print(cdata.columns)
cdata['cdata'].value_counts()
cdata.describe()
# Plotting F= feature
features = ['F2', 'F2', 'F3', 'F4']
scatter_matrix(train_cdata[features], figsize=(20, 15))
plt.show()

cdata.hist(bins=50, figsize=(20, 20))
plt.show()
#Splitting cdata into Train_cdata and test_cdata
train_cdata, test_cdata = train_test_split(train_cdata, test_size=0.2, random_state=42)

features = ['density', 'population', 'state_name', 'id']
scatter_matrix(train_cdata[features], figsize=(20, 15))
plt.show()

#Standard correlation coefficient
corr_matrix = train_cdata.corr()
corr_matrix['ranking'].sort_values(ascending=False)
cdata.head()









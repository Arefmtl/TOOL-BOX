import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA

# File path
file_path = r"C:\Users\ASUS\Desktop\code\karlancer\DATASETS\Stay free csv.csv"


def observe_data(data):
    # Display basic data info
    print(data.info())
    print(data.columns)
    print(data['some_column'].unique())
    print(data['data'].value_counts())
    print(data.describe())

    # Plot histograms and scatter matrix
    data.hist(bins=100, figsize=(20, 20))
    plt.show()

    features = ['F1', 'F2', 'F3', 'F4']
    scatter_matrix(data[features], figsize=(20, 15))
    plt.show()


def read_data(file_path):
    # Determine the file type
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path, encoding='latin-1')
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        data = pd.read_excel(file_path, encoding='latin-1')
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    return data


def clean_data(data):
    # Remove duplicates
    data = data.drop_duplicates()

    # Drop rows with missing values in a specific column
    data = data.dropna(subset=["name_of_column_with_missing_values"])

    # Fill missing values with median
    median = data['column_name'].median()
    data['column_name'].fillna(median, inplace=True)

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy="median")
    data_num = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Apply PCA to numerical features with 3 components for dimensionality reduction
    pca = PCA(n_components=3)
    data_num_pca = pca.fit_transform(data_num)

    # Concatenate numerical features with categorical features
    # Assuming you have encoded categorical features as 'data_cat_encoded'
    cleaned_data = pd.concat([pd.DataFrame(data_num_pca), data], axis=1)

    return cleaned_data


def encode_categorical_data(data):
    # Encode categorical data using OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    data_cat_encoded = encoder.fit_transform(data[['none_numerical_data']])
    data_cat_encoded = pd.DataFrame(data_cat_encoded, columns=encoder.get_feature_names_out(['none_numerical_data']))

    return data_cat_encoded


def feature_scaling(data):
    # Scale numerical features using StandardScaler
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return data_scaled


def add_derived_features(data):
    # Add derived features
    data['NF1'] = data['F1'] / data['F2']
    data['NF2'] = data['F3'] / data['F1']
    data['NF3'] = data['F4'] / data['F2']

    return data


def main():
    # Read data from file
    data = read_data(file_path)

    # Observing data
    observe_data(data)

    # Clean the data
    cleaned_data = clean_data(data)

    # Encode categorical data
    encoded_data = encode_categorical_data(cleaned_data)

    # Feature scaling
    scaled_data = feature_scaling(encoded_data)

    # Add derived features
    final_data = add_derived_features(scaled_data)

    # Display final data
    print(final_data.head())


if __name__ == "__main__":
    main()

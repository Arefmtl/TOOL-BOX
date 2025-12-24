import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the data
file_path = r""

# Read the data from either CSV or Excel file
if file_path.endswith('.csv'):
    data = pd.read_csv(file_path, header=0, encoding='latin-1')
else:
    data = pd.read_excel(file_path, header=0)

# Split features (X) and target variable (y)
X = data.drop(columns=['Outcome'])  # Assuming 'Outcome' is the target column
y = data['Outcome']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define and train the MLPClassifier model
model = MLPClassifier(hidden_layer_sizes=64, max_iter=200)  # Increase max_iter to 200
model.fit(x_train, y_train)

# Predictions on training and testing sets
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

# Calculate accuracy, precision, and recall scores
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)

# Normalize data before PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

# Apply PCA
pca = PCA(n_components=3)
pca.fit(X_train_scaled)
X_new_train = pca.transform(X_train_scaled)
X_new_test = pca.transform(X_test_scaled)

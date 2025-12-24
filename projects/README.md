# Projects Documentation

This directory contains example projects demonstrating the usage of various machine learning and data science techniques using the Tool Box.

## Available Projects

### 1. Diabetes Prediction Project (`Diabet_project/`)
**Purpose**: Classification model to predict diabetes based on medical diagnostic measurements

**Dataset**: diabetes.csv (Pima Indians Diabetes Database)
- Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- Target: Outcome (0 = No diabetes, 1 = Diabetes)

**Implementation** (`Diabet.py`):
- Uses MLPClassifier (Neural Network) from scikit-learn
- Includes PCA for dimensionality reduction
- Standardizes features before PCA
- Evaluates model with accuracy, precision, and recall

**Key Features**:
- Neural network classification
- Feature scaling and PCA
- Performance metrics calculation

### 2. Heart Rate Prediction Project (`Heartrate_project/`)
**Purpose**: Regression model to predict heart rate based on various physiological measurements

**Dataset**: heart.csv (Heart rate dataset)
- Multiple physiological features
- Target: Heart rate values

**Implementation** (`Heartrate.py`):
- Compares multiple regression algorithms:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Support Vector Regressor (SVR)
  - K-Nearest Neighbors Regressor
- Uses 10-fold cross-validation for robust evaluation
- Standardizes features and handles missing values

**Evaluation Metrics**:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²) coefficient

### 3. Housing Price Prediction Project (`Housing_project/`)
**Purpose**: Comprehensive regression analysis for California housing prices

**Dataset**: housing.csv (California Housing dataset)
- Features: longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity
- Target: median_house_value

**Implementation** (`Housing.py`):
- Complete machine learning pipeline:
  - Data exploration and visualization
  - Feature engineering (derived features)
  - Multiple regression algorithms
  - Hyperparameter tuning with GridSearchCV

**Algorithms Used**:
- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- Pipeline with preprocessing steps

**Key Features**:
- Correlation analysis with visualization
- Custom feature engineering
- Cross-validation evaluation
- Hyperparameter optimization

## Project Structure

Each project follows this structure:
```
Project_Name/
├── Project_Name.py    # Main implementation file
├── Project_Name.md    # Project documentation (if available)
├── Dataset/          # Data files
│   └── dataset.csv   # Raw data
└── README.md         # Project-specific documentation
```

## Running the Projects

### Prerequisites
- Python 3.x
- Required libraries: pandas, numpy, scikit-learn, matplotlib
- Dataset files in appropriate directories

### Execution
1. Navigate to the project directory
2. Run the Python script: `python Project_Name.py`
3. Check console output for results and metrics

## Learning Objectives

These projects demonstrate:
- **Data Preprocessing**: Cleaning, scaling, encoding
- **Feature Engineering**: Creating meaningful features
- **Model Selection**: Comparing different algorithms
- **Model Evaluation**: Using appropriate metrics
- **Pipeline Construction**: End-to-end ML workflows

## Tool Box Integration

Each project showcases different tools from the Tool Box:
- `data_cleaner_tool.py`: Data preprocessing
- `regression_tool.py`: Complete regression pipelines
- `regression_selector_tool.py`: Model comparison
- `k_fold_algorithm_tool.py`: Cross-validation
- `reading_csv_and_plotting_corr_tool.py`: EDA and visualization

## Contributing

To add new projects:
1. Create a new directory with project name
2. Include main Python script and dataset
3. Add comprehensive documentation
4. Follow the established naming conventions
5. Test the implementation thoroughly

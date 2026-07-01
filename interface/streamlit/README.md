# 🧠 TOOL-BOX v2.0 - Streamlit Interface

**Professional ML Platform with Interactive Web Interface**

## 📖 Overview

The Streamlit interface provides a complete end-to-end machine learning pipeline through an intuitive web application. Built with Streamlit, this interface offers explainable UX, backend-driven intelligence, and production-grade ML capabilities.

## 🚀 Quick Start

### Prerequisites

```bash
# Ensure Python 3.8+ is installed
python --version

# Install dependencies from root directory
pip install -r ../../requirements.txt
```

### Running the Application

```bash
# From interface/streamlit directory
python run_streamlit_app.py

# Or directly with Streamlit
streamlit run app.py --server.headless true --server.port 8501
```

The application will open at `http://localhost:8501`

## 🔧 Pipeline Steps

### 1️⃣ 📁 Load Data
- **File Upload**: Support for CSV and Excel files
- **Sample Datasets**: Built-in diabetes, heart disease, and housing datasets
- **Data Validation**: Automatic shape and quality metrics
- **Preview**: Interactive data preview with statistics

### 2️⃣ 📊 Exploratory Data Analysis (EDA)
- **Data Quality Scoring**: Comprehensive quality assessment
- **Interactive Visualizations**: Correlation matrices, distribution plots
- **Target Detection**: Automatic target column suggestions
- **Missing Value Analysis**: Detailed missing data reports
- **Outlier Detection**: Statistical outlier identification

### 3️⃣ ⚙️ Data Preprocessing
- **Missing Value Handling**: Median/mode imputation strategies
- **Categorical Encoding**: One-hot encoding for categorical variables
- **Feature Scaling**: StandardScaler for numeric features
- **Outlier Removal**: IQR and Z-score based outlier detection
- **Feature Selection**: Correlation and statistical feature selection
- **Clustering Features**: Unsupervised feature engineering

### 4️⃣ 🤖 Model Training
- **Multi-Model Training**: Train multiple algorithms simultaneously
- **AutoML Mode**: Recommended default parameters
- **Manual Configuration**: Custom parameter settings
- **Progress Tracking**: Real-time training progress
- **Error Handling**: Graceful failure management

### 5️⃣ 📈 Model Evaluation
- **Comprehensive Metrics**: Accuracy, F1, AUC, RMSE, R², etc.
- **Interactive Charts**: Plotly-based performance visualizations
- **Model Comparison**: Side-by-side performance analysis
- **Confusion Matrices**: Classification insights
- **Residual Plots**: Regression diagnostics

### 6️⃣ 🎯 Hyperparameter Optimization
- **Advanced Algorithms**: Optuna, Hyperopt, Scikit-optimize
- **Bayesian Optimization**: State-of-the-art HPO methods
- **Cross-Validation**: Robust evaluation during optimization
- **Performance Tracking**: Optimization progress and results

### 7️⃣ 💾 Export & Reports
- **Model Export**: Pickle/joblib format model serialization
- **HTML Reports**: Comprehensive project documentation
- **Dataset Export**: CSV/Parquet format data export
- **Project Serialization**: Save/load entire pipeline states
- **Artifact Management**: Download generated deliverables

## 🛠️ Supported Algorithms

### Classification Models
- Logistic Regression, Random Forest, SVM
- Gradient Boosting, KNN, Naive Bayes
- Decision Tree, XGBoost, AdaBoost
- Extra Trees, Neural Networks

### Regression Models
- Linear Regression, Ridge, Lasso
- Random Forest, SVM, Gradient Boosting
- KNN, Decision Tree, XGBoost

### Optimization Algorithms
- Optuna (TPE), Hyperopt (TPE)
- Scikit-optimize (GP), Grid Search, Random Search

## 🎯 Key Features

### Backend-Driven Intelligence
- All ML logic resides in specialized backend tools
- Clean separation between UI and ML operations
- Consistent API across all components

### Explainable UX
- Every configuration includes helpful tooltips
- Data-driven recommendations and suggestions
- Step-by-step guidance through the pipeline

### Production-Grade Export
- Model serialization with metadata
- Comprehensive HTML project reports
- Cross-platform compatibility
- Artifact preservation and management

### Session Management
- Pipeline state persistence across sessions
- Step-by-step progress tracking
- Configuration history and recovery

## 📊 Technical Architecture

```
interface/streamlit/
├── app.py                          # Main Streamlit application
├── run_streamlit_app.py           # Application launcher
├── README.md                      # This documentation
└── ...
    └── Tool-box/                 # Backend ML tools (../../Tool-box/)
        ├── data_processing_tool.py
        ├── classification_tool.py
        ├── regression_tool.py
        ├── model_evaluation_tool.py
        ├── advanced_optimization_tool.py
        └── ...
```

## 🐛 Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure you're in the correct directory
cd interface/streamlit

# Check Python path
python -c "import sys; print(sys.path)"
```

**Streamlit Not Found:**
```bash
# Install Streamlit
pip install streamlit

# Check version
streamlit --version
```

**Port Already in Use:**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

**Memory Issues:**
- Reduce model count during training
- Use feature selection to reduce dimensionality
- Close other applications

## 🔄 Session Management

The application maintains pipeline state using Streamlit's session state:
- **Pipeline Progress**: Current step and completion status
- **Data Storage**: Loaded and processed datasets
- **Model Storage**: Trained model instances
- **Configuration**: All preprocessing and training settings
- **Artifacts**: Generated reports and export files

## 📈 Performance Considerations

- **Large Datasets**: Consider sampling for EDA
- **Model Training**: Start with fewer models for testing
- **Optimization**: Begin with smaller trial counts
- **Memory Usage**: Monitor system resources during training

## 🤝 Development

### Adding New Features

1. **Backend First**: Implement in appropriate tool class
2. **UI Integration**: Add Streamlit controls in app.py
3. **State Management**: Update session state handling
4. **Error Handling**: Add appropriate error messages
5. **Documentation**: Update this README

### Code Organization

- **app.py**: Main application with step functions
- **run_streamlit_app.py**: Launcher script
- **Session State**: Pipeline state management
- **Tool Integration**: Backend tool imports and usage

## 📞 Support

- **Issues**: [TOOL-BOX Issues](../../issues)
- **Discussions**: [TOOL-BOX Discussions](../../discussions)
- **Documentation**: [TOOL-BOX Wiki](../../wiki)

---

**TOOL-BOX v2.0 Streamlit Interface** - Interactive ML pipeline for modern data scientists! 🚀
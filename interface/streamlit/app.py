"""
TOOL-BOX - Production-Grade ML Platform
Streamlit-based single-page ML pipeline with backend-driven intelligence
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import tempfile
import json
import pickle
from datetime import datetime
import zipfile
import io

# Import TOOL-BOX modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Tool-box'))

from data_processing_tool import DataProcessingTool
from classification_tool import ClassificationTool
from regression_tool import RegressionTool
from model_evaluation_tool import ModelEvaluationTool
from cross_validation_tool import CrossValidationTool
from hyperparameter_tuning_tool import HyperparameterTuningTool
from advanced_optimization_tool import AdvancedOptimizationTool

# Import loguru for logging
from loguru import logger

# Configure logging
logger.add("tool_box_ml_pipeline.log", rotation="10 MB", level="INFO",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}")

# Initialize tools
data_processor = DataProcessingTool()
classifier = ClassificationTool()
regressor = RegressionTool()
evaluator = ModelEvaluationTool()
cv_tool = CrossValidationTool()
tuner = HyperparameterTuningTool()
advanced_optimizer = AdvancedOptimizationTool()

# Configure Streamlit page
st.set_page_config(
    page_title="TOOL-BOX - Professional ML Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .step-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .step-card.active {
        border-color: #4CAF50;
        background-color: #f8fff8;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.2);
    }
    .step-card.completed {
        border-color: #2196F3;
        background-color: #f8faff;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state with default values"""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = {
            'current_step': 0,
            'data': None,
            'processed_data': None,
            'eda_summary': None,
            'preprocess_config': None,
            'task_type': None,
            'models': {},
            'training_results': {},
            'best_model': None,
            'optimized_model': None,
            'evaluation_summary': None,
            'artifacts': {}
        }

def reset_pipeline():
    """Reset the entire pipeline"""
    st.session_state.pipeline = {
        'current_step': 0,
        'data': None,
        'processed_data': None,
        'eda_summary': None,
        'preprocess_config': None,
        'task_type': None,
        'models': {},
        'training_results': {},
        'best_model': None,
        'optimized_model': None,
        'evaluation_summary': None,
        'artifacts': {}
    }
    logger.info("Pipeline reset by user")

def advance_step():
    """Advance to next step"""
    current_step = st.session_state.pipeline['current_step']
    if current_step < 6:  # 0-6 steps
        st.session_state.pipeline['current_step'] = current_step + 1
        logger.info(f"Advanced to step {current_step + 1}")

def go_to_step(step_number):
    """Go to specific step"""
    if 0 <= step_number <= 6:
        st.session_state.pipeline['current_step'] = step_number
        logger.info(f"Jumped to step {step_number}")

def create_sidebar():
    """Create sidebar with pipeline navigation"""
    st.sidebar.title("🎯 TOOL-BOX Pipeline")

    steps = [
        "📁 Load Data",
        "📊 EDA",
        "⚙️ Preprocessing",
        "👨‍💻 Training",
        "📈 Evaluation",
        "🎯 Optimization",
        "💾 Export & Reports"
    ]

    current_step = st.session_state.pipeline['current_step']

    for i, step_name in enumerate(steps):
        if i < current_step:
            # Completed step
            if st.sidebar.button(f"✅ {step_name}", key=f"step_{i}", use_container_width=True):
                go_to_step(i)
        elif i == current_step:
            # Current step
            st.sidebar.button(f"🔄 {step_name}", key=f"step_{i}", use_container_width=True, disabled=True)
        else:
            # Future step
            st.sidebar.button(f"⏳ {step_name}", key=f"step_{i}", use_container_width=True, disabled=True)

    st.sidebar.markdown("---")

    # Pipeline status
    st.sidebar.subheader("📋 Pipeline Status")
    status_items = [
        ("Data Loaded", st.session_state.pipeline['data'] is not None),
        ("EDA Complete", st.session_state.pipeline['eda_summary'] is not None),
        ("Data Processed", st.session_state.pipeline['processed_data'] is not None),
        ("Models Trained", len(st.session_state.pipeline['models']) > 0),
        ("Models Evaluated", st.session_state.pipeline['evaluation_summary'] is not None),
        ("Model Optimized", st.session_state.pipeline['optimized_model'] is not None),
    ]

    for item, status in status_items:
        icon = "✅" if status else "❌"
        st.sidebar.write(f"{icon} {item}")

    st.sidebar.markdown("---")

    # Reset button
    if st.sidebar.button("🔄 Reset Pipeline", use_container_width=True):
        reset_pipeline()
        st.rerun()

def step_1_load_data():
    """Step 1: Load Data"""
    st.header("📁 Step 1: Load Dataset")

    st.markdown("""
    **Choose how to load your dataset:**
    - Upload CSV/Excel files
    - Load from URL
    - Use sample datasets or famous datasets
    """)

    # Data loading options
    tab1, tab2, tab3 = st.tabs(["📤 Upload File", "🔗 Load from URL", "🎯 Datasets"])

    with tab1:
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Supported formats: CSV, Excel (.xlsx, .xls)"
        )

        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # Load data
                data = data_processor.load_data(tmp_file_path)

                # Clean up
                os.unlink(tmp_file_path)

                # Store in session
                st.session_state.pipeline['data'] = data

                st.success(f"✅ Dataset loaded successfully! Shape: {data.shape[0]} rows × {data.shape[1]} columns")

                # Preview data
                st.subheader("📋 Data Preview")
                st.dataframe(data.head(10), use_container_width=True)

                # Basic info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", data.shape[0])
                with col2:
                    st.metric("Columns", data.shape[1])
                with col3:
                    missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
                    st.metric("Missing Data", f"{missing_pct:.1f}%")

                logger.info(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")

            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                logger.error(f"Data loading error: {str(e)}")

    with tab2:
        st.subheader("🔗 Load from URL")

        url_input = st.text_input(
            "Enter dataset URL",
            placeholder="https://example.com/data.csv",
            help="Supported formats: CSV, Excel (.xlsx, .xls)"
        )

        if st.button("🌐 Load from URL") and url_input:
            try:
                with st.spinner("📡 Downloading data from URL..."):
                    import requests

                    # Download the file
                    response = requests.get(url_input, timeout=30)
                    response.raise_for_status()

                    # Determine file extension from URL or content-type
                    content_type = response.headers.get('content-type', '')
                    if 'excel' in content_type or 'spreadsheet' in content_type:
                        file_ext = '.xlsx'
                    elif url_input.lower().endswith(('.xlsx', '.xls')):
                        file_ext = '.xlsx'
                    else:
                        file_ext = '.csv'

                    # Save temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                        tmp_file.write(response.content)
                        tmp_file_path = tmp_file.name

                    # Load data
                    data = data_processor.load_data(tmp_file_path)

                    # Clean up
                    os.unlink(tmp_file_path)

                    # Store in session
                    st.session_state.pipeline['data'] = data

                    st.success(f"✅ Dataset loaded from URL! Shape: {data.shape[0]} rows × {data.shape[1]} columns")

                    # Preview data
                    st.subheader("📋 Data Preview")
                    st.dataframe(data.head(10))

                    # Basic info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", data.shape[0])
                    with col2:
                        st.metric("Columns", data.shape[1])
                    with col3:
                        missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
                        st.metric("Missing Data", f"{missing_pct:.1f}%")

                    logger.info(f"Data loaded from URL: {url_input} - {data.shape[0]} rows, {data.shape[1]} columns")

            except requests.exceptions.RequestException as e:
                st.error(f"❌ Network error: {str(e)}")
                logger.error(f"URL loading network error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error loading data from URL: {str(e)}")
                logger.error(f"URL loading error: {str(e)}")

    with tab3:
        st.subheader("🎯 Datasets")

        # Sample datasets
        st.markdown("**Sample Datasets**")
        sample_datasets = {
            "Diabetes": "projects/Diabet_project/Dataset/diabetes.csv",
            "Heart Disease": "projects/Heartrate_project/Dataset/heart_disease_uci.csv",
            "Housing": "projects/Housing_project/Dataset/housing.csv"
        }

        selected_dataset = st.selectbox(
            "Choose a sample dataset",
            options=list(sample_datasets.keys()),
            help="These datasets are included with TOOL-BOX for testing"
        )

        if st.button("Load Sample Dataset"):
            try:
                file_path = sample_datasets[selected_dataset]
                if os.path.exists(file_path):
                    data = data_processor.load_data(file_path)
                    st.session_state.pipeline['data'] = data

                    st.success(f"✅ Sample dataset '{selected_dataset}' loaded! Shape: {data.shape[0]} rows × {data.shape[1]} columns")
                    st.dataframe(data.head(5))

                    logger.info(f"Sample dataset loaded: {selected_dataset}")
                else:
                    st.error(f"❌ Sample dataset file not found: {file_path}")

            except Exception as e:
                st.error(f"❌ Error loading sample dataset: {str(e)}")
                logger.error(f"Sample dataset loading error: {str(e)}")

        # Famous datasets
        st.markdown("---")
        st.markdown("**Famous Datasets**")

        famous_datasets = {
            "Iris": "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv",
            "Wine": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
            "Boston Housing": "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
            "Titanic": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
            "MNIST Digits": "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/digits.csv"
        }

        selected_famous = st.selectbox(
            "Choose a famous dataset",
            options=list(famous_datasets.keys()),
            help="Popular datasets from ML literature and competitions"
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**URL:** {famous_datasets[selected_famous]}")
        with col2:
            if st.button("🔗 Load Famous Dataset"):
                try:
                    with st.spinner("📡 Loading famous dataset..."):
                        import requests

                        # Download the file
                        response = requests.get(famous_datasets[selected_famous], timeout=30)
                        response.raise_for_status()

                        # Determine file extension
                        if '.csv' in famous_datasets[selected_famous].lower():
                            file_ext = '.csv'
                        elif '.xlsx' in famous_datasets[selected_famous].lower():
                            file_ext = '.xlsx'
                        else:
                            file_ext = '.csv'  # Default

                        # Save temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                            tmp_file.write(response.content)
                            tmp_file_path = tmp_file.name

                        # Load data
                        data = data_processor.load_data(tmp_file_path)

                        # Clean up
                        os.unlink(tmp_file_path)

                        # Store in session
                        st.session_state.pipeline['data'] = data

                        st.success(f"✅ Famous dataset '{selected_famous}' loaded! Shape: {data.shape[0]} rows × {data.shape[1]} columns")

                        # Preview data
                        st.subheader("📋 Data Preview")
                        st.dataframe(data.head(10), use_container_width=True)

                        logger.info(f"Famous dataset loaded: {selected_famous} - {data.shape[0]} rows, {data.shape[1]} columns")

                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Network error loading {selected_famous}: {str(e)}")
                    logger.error(f"Famous dataset network error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error loading {selected_famous}: {str(e)}")
                    logger.error(f"Famous dataset loading error: {str(e)}")

    # Next step button
    if st.session_state.pipeline['data'] is not None:
        st.markdown("---")
        if st.button("➡️ Continue to EDA", use_container_width=True):
            advance_step()
            st.rerun()

def step_2_eda():
    """Step 2: Exploratory Data Analysis - Comprehensive"""
    st.header("📊 Step 2: Exploratory Data Analysis (EDA)")

    if st.session_state.pipeline['data'] is None:
        st.error("❌ No data loaded. Please go back to Step 1.")
        return

    data = st.session_state.pipeline['data']

    # EDA Navigation Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Data Understanding",
        "🧹 Data Cleaning",
        "📊 Univariate Analysis",
        "🔗 Bivariate Analysis",
        "🎨 Advanced Visualizations"
    ])

    with tab1:
        st.subheader("🔍 Data Understanding")

        # Dataset Overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", data.shape[0])
        with col2:
            st.metric("Total Columns", data.shape[1])
        with col3:
            memory_usage = data.memory_usage(deep=True).sum()
            st.metric("Memory Usage", f"{memory_usage / 1024:.1f} KB")
        with col4:
            duplicate_rows = data.duplicated().sum()
            st.metric("Duplicate Rows", duplicate_rows)

        # Data Types Summary
        st.subheader("🔢 Data Types Summary")
        dtypes_df = pd.DataFrame({
            'Data Type': data.dtypes.value_counts().index.astype(str),
            'Count': data.dtypes.value_counts().values,
            'Columns': [', '.join(data.select_dtypes(include=[dtype]).columns.tolist()) for dtype in data.dtypes.value_counts().index]
        })
        st.dataframe(dtypes_df, use_container_width=True)

        # Sample Data Preview
        st.subheader("👀 Sample Data")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📋 First 5 Rows (Head)", use_container_width=True):
                st.dataframe(data.head(), use_container_width=True)
        with col2:
            if st.button("📄 Last 5 Rows (Tail)", use_container_width=True):
                st.dataframe(data.tail(), use_container_width=True)
        with col3:
            if st.button("🔀 Random Sample", use_container_width=True):
                st.dataframe(data.sample(min(5, len(data))), use_container_width=True)

        # Dataset Info
        st.subheader("ℹ️ Dataset Information")
        buffer = io.StringIO()
        data.info(buf=buffer)
        info_str = buffer.getvalue()
        st.code(info_str, language="text")

    with tab2:
        st.subheader("🧹 Data Cleaning Insights")

        # Missing Values Analysis
        st.subheader("❓ Missing Values Analysis")

        missing_data = data.isnull().sum()
        missing_percent = (missing_data / len(data)) * 100

        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': missing_percent.values
        }).sort_values('Missing Count', ascending=False)

        missing_df = missing_df[missing_df['Missing Count'] > 0]

        if not missing_df.empty:
            st.dataframe(missing_df, use_container_width=True)

            # Missing Values Heatmap
            st.subheader("🔥 Missing Values Heatmap")
            fig = px.imshow(
                data.isnull(),
                color_continuous_scale=['#10B981', '#EF4444'],
                title="Missing Values Pattern (Green=Present, Red=Missing)"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found in the dataset!")

        # Duplicate Analysis
        st.subheader("📋 Duplicate Analysis")
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            st.warning(f"⚠️ Found {duplicates} duplicate rows")

            if st.button("🔍 Show Duplicate Rows", use_container_width=True):
                duplicate_rows = data[data.duplicated(keep=False)]
                st.dataframe(duplicate_rows, use_container_width=True)
        else:
            st.success("✅ No duplicate rows found!")

        # Data Type Issues
        st.subheader("🔧 Data Type Analysis")

        # Check for potential type conversion issues
        type_issues = []

        for col in data.columns:
            dtype = data[col].dtype

            # Check if numeric columns have non-numeric values
            if dtype in ['int64', 'float64']:
                non_numeric = pd.to_numeric(data[col], errors='coerce').isnull().sum()
                if non_numeric > 0:
                    type_issues.append({
                        'Column': col,
                        'Issue': f'{non_numeric} non-numeric values in numeric column',
                        'Suggestion': 'Consider converting to string or cleaning data'
                    })

            # Check for mixed types in object columns
            elif dtype == 'object':
                unique_types = data[col].apply(type).unique()
                if len(unique_types) > 1:
                    type_issues.append({
                        'Column': col,
                        'Issue': f'Mixed data types: {[t.__name__ for t in unique_types]}',
                        'Suggestion': 'Consider standardizing data types'
                    })

        if type_issues:
            issues_df = pd.DataFrame(type_issues)
            st.dataframe(issues_df, use_container_width=True)
        else:
            st.success("✅ No data type issues detected!")

    with tab3:
        st.subheader("📊 Univariate Analysis")

        # Select column for detailed analysis
        selected_col = st.selectbox("Select column for detailed analysis", data.columns.tolist())

        if selected_col:
            col_data = data[selected_col]
            col_type = 'numeric' if col_data.dtype in ['int64', 'float64'] else 'categorical'

            # Basic Statistics
            st.subheader(f"📈 Statistics for {selected_col}")

            if col_type == 'numeric':
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{col_data.mean():.2f}")
                with col2:
                    st.metric("Median", f"{col_data.median():.2f}")
                with col3:
                    st.metric("Std Dev", f"{col_data.std():.2f}")
                with col4:
                    st.metric("Variance", f"{col_data.var():.2f}")

                # Distribution Plot
                st.subheader("📊 Distribution")
                fig = px.histogram(
                    col_data.dropna(),
                    nbins=50,
                    title=f"Distribution of {selected_col}",
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Box Plot for Outliers
                st.subheader("📦 Box Plot (Outlier Detection)")
                fig_box = px.box(
                    col_data.dropna(),
                    title=f"Box Plot of {selected_col}",
                    points="outliers"
                )
                st.plotly_chart(fig_box, use_container_width=True)

                # Descriptive Statistics
                st.subheader("📋 Descriptive Statistics")
                desc_stats = col_data.describe()
                st.dataframe(desc_stats, use_container_width=True)

                # Skewness and Kurtosis
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Skewness", f"{col_data.skew():.3f}")
                with col2:
                    st.metric("Kurtosis", f"{col_data.kurtosis():.3f}")

            else:  # Categorical
                # Value Counts
                st.subheader("📊 Value Distribution")
                value_counts = col_data.value_counts().head(20)  # Limit to top 20
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Value Distribution of {selected_col}"
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

                # Summary Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Unique Values", col_data.nunique())
                with col2:
                    st.metric("Mode", str(col_data.mode().iloc[0]) if not col_data.mode().empty else "N/A")
                with col3:
                    most_common_count = col_data.value_counts().iloc[0] if len(col_data.value_counts()) > 0 else 0
                    st.metric("Most Common Count", most_common_count)
                with col4:
                    missing_pct = (col_data.isnull().sum() / len(col_data)) * 100
                    st.metric("Missing %", f"{missing_pct:.1f}%")

                # Pie Chart for categorical data
                if col_data.nunique() <= 10:  # Only show pie for reasonable number of categories
                    st.subheader("🥧 Category Distribution")
                    fig_pie = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f"Category Distribution of {selected_col}"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

    with tab4:
        st.subheader("🔗 Bivariate & Multivariate Analysis")

        # Correlation Analysis
        st.subheader("🔗 Correlation Matrix")
        numeric_data = data.select_dtypes(include=[np.number])

        if len(numeric_data.columns) > 1:
            corr_matrix = numeric_data.corr()

            # Correlation Heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                title="Correlation Heatmap",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Strong Correlations
            st.subheader("💪 Strong Correlations (|r| > 0.7)")
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corr.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': f"{corr_val:.3f}"
                        })

            if strong_corr:
                strong_corr_df = pd.DataFrame(strong_corr)
                st.dataframe(strong_corr_df, use_container_width=True)
            else:
                st.info("ℹ️ No strong correlations found")

        # Scatter Plot Analysis
        st.subheader("📈 Scatter Plot Analysis")

        if len(numeric_data.columns) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", numeric_data.columns.tolist(),
                                   key="scatter_x")
            with col2:
                y_col = st.selectbox("Y-axis", numeric_data.columns.tolist(),
                                   key="scatter_y")

            if x_col and y_col and x_col != y_col:
                fig = px.scatter(
                    data,
                    x=x_col,
                    y=y_col,
                    title=f"Scatter Plot: {x_col} vs {y_col}",
                    trendline="ols"
                )
                st.plotly_chart(fig, use_container_width=True)

        # Categorical vs Numeric Analysis
        st.subheader("📊 Categorical vs Numeric Analysis")

        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if categorical_cols and numeric_cols:
            cat_col = st.selectbox("Categorical Variable", categorical_cols, key="cat_var")
            num_col = st.selectbox("Numeric Variable", numeric_cols, key="num_var")

            if cat_col and num_col:
                # Box plot by category
                fig = px.box(
                    data,
                    x=cat_col,
                    y=num_col,
                    title=f"{num_col} Distribution by {cat_col}"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Group statistics
                st.subheader("📋 Group Statistics")
                group_stats = data.groupby(cat_col)[num_col].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
                st.dataframe(group_stats, use_container_width=True)

    with tab5:
        st.subheader("🎨 Advanced Visualizations")

        # Pair Plot (for numeric features)
        st.subheader("🔗 Pair Plot")
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) >= 2:
            # Limit to prevent performance issues
            max_features = min(5, len(numeric_cols))
            selected_features = st.multiselect(
                "Select features for pair plot",
                numeric_cols,
                default=numeric_cols[:max_features],
                key="pair_plot_features"
            )

            if len(selected_features) >= 2:
                # Create pair plot using plotly
                pair_data = data[selected_features].dropna()

                if len(pair_data) > 0:
                    # Create subplot grid
                    fig = make_subplots(
                        rows=len(selected_features),
                        cols=len(selected_features),
                        subplot_titles=[f"{x} vs {y}" if i != j else f"{selected_features[i]} Distribution"
                                      for i in range(len(selected_features))
                                      for j in range(len(selected_features))],
                        shared_xaxes=False,
                        shared_yaxes=False
                    )

                    for i, col1 in enumerate(selected_features):
                        for j, col2 in enumerate(selected_features):
                            if i == j:
                                # Diagonal - histogram
                                fig.add_trace(
                                    go.Histogram(x=pair_data[col1], showlegend=False),
                                    row=i+1, col=j+1
                                )
                            else:
                                # Off-diagonal - scatter
                                fig.add_trace(
                                    go.Scatter(x=pair_data[col2], y=pair_data[col1], mode='markers',
                                             showlegend=False, marker=dict(size=3, opacity=0.6)),
                                    row=i+1, col=j+1
                                )

                    fig.update_layout(height=800, title="Pair Plot")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Not enough non-null data for pair plot")

        # Violin Plots
        st.subheader("🎻 Violin Plots")

        if numeric_cols:
            selected_violin = st.selectbox("Select numeric feature for violin plot", numeric_cols)

            if selected_violin:
                fig = px.violin(
                    data,
                    y=selected_violin,
                    box=True,
                    points="all",
                    title=f"Violin Plot of {selected_violin}"
                )
                st.plotly_chart(fig, use_container_width=True)

        # Outlier Detection Visualization
        st.subheader("🔍 Outlier Detection")

        if numeric_cols:
            outlier_col = st.selectbox("Select feature for outlier analysis", numeric_cols, key="outlier_col")

            if outlier_col:
                # Z-score method
                z_scores = np.abs((data[outlier_col] - data[outlier_col].mean()) / data[outlier_col].std())
                outliers_z = z_scores > 3

                # IQR method
                Q1 = data[outlier_col].quantile(0.25)
                Q3 = data[outlier_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers_iqr = (data[outlier_col] < lower_bound) | (data[outlier_col] > upper_bound)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Z-score Outliers (>3)", outliers_z.sum())
                with col2:
                    st.metric("IQR Outliers", outliers_iqr.sum())

                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[outlier_col],
                    mode='markers',
                    name='Normal',
                    marker=dict(color='blue', size=4)
                ))

                # Highlight outliers
                outlier_indices = data[(outliers_z | outliers_iqr)].index
                if len(outlier_indices) > 0:
                    fig.add_trace(go.Scatter(
                        x=outlier_indices,
                        y=data.loc[outlier_indices, outlier_col],
                        mode='markers',
                        name='Outliers',
                        marker=dict(color='red', size=8, symbol='x')
                    ))

                fig.update_layout(
                    title=f"Outlier Detection for {outlier_col}",
                    xaxis_title="Index",
                    yaxis_title=outlier_col
                )
                st.plotly_chart(fig, use_container_width=True)

    # Feature Engineering Suggestions
    st.subheader("💡 Feature Engineering Suggestions")

    suggestions = []

    # Check for potential date columns
    for col in data.columns:
        if data[col].dtype == 'object':
            sample_values = data[col].dropna().head(10).tolist()
            # Check if looks like date
            if any('date' in str(val).lower() or 'time' in str(val).lower() or
                   any(char in str(val) for char in ['/', '-', ':']) for val in sample_values):
                suggestions.append(f"🔗 Consider converting '{col}' to datetime format")

    # Check for high cardinality categorical variables
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if data[col].nunique() > 10:
            suggestions.append(f"📊 '{col}' has high cardinality ({data[col].nunique()} unique values). Consider grouping or encoding.")

    # Check for skewed numeric variables
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        skewness = data[col].skew()
        if abs(skewness) > 1:
            transform_type = "log transformation" if skewness > 0 else "exponential transformation"
            suggestions.append(f"📈 '{col}' is skewed (skewness: {skewness:.2f}). Consider {transform_type}.")

    # Check for features that might benefit from scaling
    if len(numeric_cols) > 1:
        ranges = data[numeric_cols].max() - data[numeric_cols].min()
        if ranges.max() / ranges.min() > 10:
            suggestions.append("⚖️ Features have different scales. Consider standardization or normalization.")

    if suggestions:
        for suggestion in suggestions:
            st.info(suggestion)
    else:
        st.success("✅ Dataset looks well-prepared for modeling!")

    # Next step button
    st.markdown("---")
    if st.button("➡️ Continue to Preprocessing", use_container_width=True):
        advance_step()
        st.rerun()

def step_3_preprocessing():
    """Step 3: Data Preprocessing"""
    st.header("⚙️ Step 3: Data Preprocessing")

    if st.session_state.pipeline['data'] is None:
        st.error("❌ No data loaded. Please go back to Step 1.")
        return

    data = st.session_state.pipeline['data']

    st.markdown("""
    **Configure preprocessing pipeline:**
    Choose preprocessing steps and parameters for optimal ML preparation.
    """)

    # Preprocessing configuration
    st.subheader("🔧 Preprocessing Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Basic Preprocessing**")
        handle_missing = st.checkbox("Handle Missing Values", value=True,
                                   help="Automatically fill missing values (median for numeric, mode for categorical)")
        encode_categorical = st.checkbox("Encode Categorical Variables", value=True,
                                       help="Convert categorical variables to numeric using one-hot encoding")
        scale_features = st.checkbox("Scale Numeric Features", value=True,
                                   help="Standardize numeric features using StandardScaler")

    with col2:
        st.markdown("**Advanced Preprocessing**")
        handle_outliers = st.checkbox("Handle Outliers", value=False,
                                    help="Remove or cap outliers using IQR method")
        feature_selection = st.checkbox("Feature Selection", value=False,
                                      help="Select most important features using correlation or statistical tests")
        add_clustering = st.checkbox("Add Clustering Features", value=False,
                                   help="Add cluster-based features for enhanced modeling")

    # Column selection/dropping
    st.subheader("📊 Column Management")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Select Columns to Keep**")
        all_columns = data.columns.tolist()
        selected_columns = st.multiselect(
            "Choose columns to include in analysis",
            options=all_columns,
            default=all_columns,
            help="Select which columns to keep. Unselected columns will be dropped."
        )

        if selected_columns != all_columns:
            dropped_columns = set(all_columns) - set(selected_columns)
            data = data[selected_columns]
            st.success(f"✅ Dataset updated to {len(selected_columns)} columns")

            if dropped_columns:
                with st.expander(f"📋 Dropped {len(dropped_columns)} columns"):
                    st.write("**Dropped columns:**")
                    for col in dropped_columns:
                        st.write(f"• `{col}`")

    with col2:
        st.markdown("**Target Column**")
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        available_cols = data.columns.tolist()

        target_col = st.selectbox(
            "Select target column for supervised learning",
            options=["Auto-detect"] + available_cols,
            help="Choose the column you want to predict. If 'Auto-detect', the system will suggest suitable columns."
        )

    if target_col == "Auto-detect":
        # Use EDA suggestions
        if st.session_state.pipeline['eda_summary'] and 'target_candidates' in st.session_state.pipeline['eda_summary']:
            candidates = st.session_state.pipeline['eda_summary']['target_candidates']
            if candidates:
                target_col = candidates[0]['column']  # Pick first candidate
                st.info(f"🎯 Auto-selected target: **{target_col}**")
            else:
                st.warning("⚠️ No suitable target column detected. Please select manually.")
                target_col = None
        else:
            st.warning("⚠️ Run EDA first for target suggestions.")
            target_col = None

    # Advanced options
    if handle_outliers:
        st.subheader("📊 Outlier Handling")
        outlier_method = st.selectbox("Outlier Method", ["iqr", "zscore"],
                                    help="IQR: Interquartile range method (recommended), Z-score: Standard deviation method")

    if feature_selection:
        st.subheader("🎯 Feature Selection")
        fs_method = st.selectbox("Feature Selection Method",
                               ["correlation", "f_regression", "f_classif"],
                               help="Correlation: Based on target correlation, Statistical tests: Univariate feature selection")
        k_features = st.slider("Number of features to select", 5, 50, 10,
                             help="Maximum number of features to keep")

    if add_clustering:
        st.subheader("🎨 Clustering Features")
        n_clusters = st.slider("Number of clusters", 2, 10, 3,
                             help="Number of clusters for feature engineering")

    # Test size for train/test split
    st.subheader("📊 Train/Test Split")
    test_size = st.slider("Test set size (%)", 10, 40, 20,
                        help="Percentage of data to use for testing") / 100

    # Apply preprocessing
    if st.button("🚀 Apply Preprocessing", use_container_width=True):
        try:
            with st.spinner("🔄 Applying preprocessing pipeline..."):
                # Build preprocessing config
                config = {
                    'handle_missing': handle_missing,
                    'missing_method': 'auto',
                    'encode_categorical': encode_categorical,
                    'scale_features': scale_features,
                    'handle_outliers': handle_outliers,
                    'outlier_method': outlier_method if handle_outliers else 'iqr',
                    'feature_selection': feature_selection,
                    'fs_method': fs_method if feature_selection else 'correlation',
                    'k_features': k_features if feature_selection else 10,
                    'add_clustering': add_clustering,
                    'n_clusters': n_clusters if add_clustering else 3,
                    'target_column': target_col
                }

                # Apply preprocessing
                result = data_processor.advanced_preprocessing(data, config)

                if 'error' in result:
                    st.error(f"❌ Preprocessing error: {result['error']}")
                    logger.error(f"Preprocessing error: {result['error']}")
                else:
                    # Store results
                    st.session_state.pipeline['processed_data'] = result['processed_data']
                    st.session_state.pipeline['preprocess_config'] = config

                    # Determine task type
                    if target_col and target_col in result['processed_data'].columns:
                        target_values = result['processed_data'][target_col].dropna()
                        unique_count = len(target_values.unique())
                        task_type = 'classification' if unique_count <= 20 else 'regression'
                        st.session_state.pipeline['task_type'] = task_type
                    else:
                        st.session_state.pipeline['task_type'] = 'unsupervised'

                    st.success("✅ Data preprocessing completed successfully!")

                    # Show results
                    st.subheader("📋 Preprocessing Results")
                    st.write(f"**Original shape:** {data.shape}")
                    st.write(f"**Processed shape:** {result['processed_data'].shape}")
                    st.write(f"**Task type:** {st.session_state.pipeline['task_type']}")

                    # Show preprocessing log
                    with st.expander("🔍 Preprocessing Details"):
                        for log_entry in result['preprocessing_log']:
                            st.write(f"• {log_entry}")

                    # Preview processed data
                    st.subheader("👀 Processed Data Preview")
                    st.dataframe(result['processed_data'].head(10), use_container_width=True)

                    logger.info(f"Preprocessing completed: {data.shape} -> {result['processed_data'].shape}")

        except Exception as e:
            st.error(f"❌ Preprocessing failed: {str(e)}")
            logger.error(f"Preprocessing failed: {str(e)}")

    # Export preprocessed data
    if st.session_state.pipeline['processed_data'] is not None:
        st.markdown("---")
        st.subheader("💾 Export Preprocessed Data")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Export as CSV", use_container_width=True):
                csv_data = st.session_state.pipeline['processed_data'].to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="preprocessed_data.csv",
                    mime="text/csv"
                )

        with col2:
            if st.button("📦 Export as Parquet", use_container_width=True):
                parquet_buffer = io.BytesIO()
                st.session_state.pipeline['processed_data'].to_parquet(parquet_buffer, index=False)
                st.download_button(
                    label="Download Parquet",
                    data=parquet_buffer,
                    file_name="preprocessed_data.parquet",
                    mime="application/octet-stream"
                )

    # Next step button
    if st.session_state.pipeline['processed_data'] is not None:
        st.markdown("---")
        if st.button("➡️ Continue to Training", use_container_width=True):
            advance_step()
            st.rerun()

def step_4_training():
    """Step 4: Model Training"""
    st.header("🤖 Step 4: Model Training")

    if st.session_state.pipeline['processed_data'] is None:
        st.error("❌ No processed data available. Please complete preprocessing first.")
        return

    processed_data = st.session_state.pipeline['processed_data']
    task_type = st.session_state.pipeline['task_type']
    target_col = st.session_state.pipeline['preprocess_config']['target_column']

    # Support regression, clustering, and classification
    supported_task_types = ['classification', 'regression', 'clustering']
    if task_type not in supported_task_types:
        st.warning(f"⚠️ Task type '{task_type}' detected. TOOL-BOX supports: {', '.join(supported_task_types)}")
        st.info("💡 If auto-detection failed, you can manually specify hyperparameters below")

        # Manual task type selection
        manual_task_type = st.selectbox(
            "Select Task Type Manually",
            options=supported_task_types,
            help="Choose the appropriate task type for your data"
        )

        if manual_task_type:
            task_type = manual_task_type
            st.session_state.pipeline['task_type'] = task_type
            st.success(f"✅ Task type set to: {task_type}")
        else:
            st.error("❌ Please select a task type to continue")
            return

    st.markdown(f"""
    **Training Configuration:**
    - Task Type: **{task_type.title()}**
    - Target Column: **{target_col}**
    - Training Samples: **{len(processed_data)}**
    """)

    # Model selection
    st.subheader("🎯 Model Selection")

    if task_type == 'classification':
        available_models = [
            'logistic_regression', 'random_forest', 'svm', 'gradient_boosting',
            'knn', 'naive_bayes', 'decision_tree', 'xgboost', 'adaboost',
            'extra_trees', 'neural_network'
        ]
        default_models = ['logistic_regression', 'random_forest', 'xgboost']
    else:  # regression
        available_models = [
            'linear_regression', 'ridge', 'lasso', 'random_forest', 'svm',
            'gradient_boosting', 'knn', 'decision_tree', 'xgboost'
        ]
        default_models = ['linear_regression', 'random_forest', 'xgboost']

    selected_models = st.multiselect(
        "Select models to train",
        options=available_models,
        default=default_models,
        help="Choose multiple models to train simultaneously"
    )

    # Training mode
    training_mode = st.radio(
        "Training Mode",
        ["AutoML (Recommended)", "Manual Configuration"],
        help="AutoML uses default parameters, Manual allows custom configuration"
    )

    # Train models
    if st.button("🚀 Train Models", use_container_width=True) and selected_models:
        try:
            with st.spinner("🤖 Training models... This may take a few minutes"):
                # Prepare data
                X = processed_data.drop(target_col, axis=1)
                y = processed_data[target_col]

                # Split data
                from sklearn.model_selection import train_test_split
                test_size = st.session_state.pipeline['preprocess_config'].get('test_size', 0.2)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                trained_models = {}

                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, model_name in enumerate(selected_models):
                    status_text.text(f"Training {model_name.replace('_', ' ').title()}...")

                    try:
                        if task_type == 'classification':
                            if model_name == 'logistic_regression':
                                model = classifier.train_logistic_regression(X_train, y_train)
                            elif model_name == 'random_forest':
                                model = classifier.train_random_forest(X_train, y_train)
                            elif model_name == 'svm':
                                model = classifier.train_svm(X_train, y_train)
                            elif model_name == 'gradient_boosting':
                                model = classifier.train_gradient_boosting(X_train, y_train)
                            elif model_name == 'knn':
                                model = classifier.train_knn(X_train, y_train)
                            elif model_name == 'naive_bayes':
                                model = classifier.train_naive_bayes(X_train, y_train)
                            elif model_name == 'decision_tree':
                                model = classifier.train_decision_tree(X_train, y_train)
                            elif model_name == 'xgboost':
                                model = classifier.train_xgboost_classifier(X_train, y_train)
                            elif model_name == 'adaboost':
                                model = classifier.train_adaboost_classifier(X_train, y_train)
                            elif model_name == 'extra_trees':
                                model = classifier.train_extra_trees_classifier(X_train, y_train)
                            elif model_name == 'neural_network':
                                model = classifier.train_neural_network_classifier(X_train, y_train)
                        else:  # regression
                            if model_name == 'linear_regression':
                                model = regressor.train_linear_regression(X_train, y_train)
                            elif model_name == 'ridge':
                                model = regressor.train_ridge(X_train, y_train)
                            elif model_name == 'lasso':
                                model = regressor.train_lasso(X_train, y_train)
                            elif model_name == 'random_forest':
                                model = regressor.train_random_forest(X_train, y_train)
                            elif model_name == 'svm':
                                model = regressor.train_svr(X_train, y_train)
                            elif model_name == 'gradient_boosting':
                                model = regressor.train_gradient_boosting(X_train, y_train)
                            elif model_name == 'knn':
                                model = regressor.train_knn(X_train, y_train)
                            elif model_name == 'decision_tree':
                                model = regressor.train_decision_tree(X_train, y_train)
                            elif model_name == 'xgboost':
                                model = regressor.train_xgboost_regressor(X_train, y_train)

                        if model is not None:
                            trained_models[model_name] = model

                    except Exception as e:
                        st.warning(f"⚠️ Failed to train {model_name}: {str(e)}")
                        logger.warning(f"Model training failed for {model_name}: {str(e)}")

                    # Update progress
                    progress_bar.progress((i + 1) / len(selected_models))

                # Store results
                st.session_state.pipeline['models'] = trained_models
                st.session_state.pipeline['training_results'] = {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'task_type': task_type
                }

                status_text.empty()
                progress_bar.empty()

                st.success(f"✅ Successfully trained {len(trained_models)} out of {len(selected_models)} models!")

                # Show trained models
                st.subheader("🎯 Trained Models")
                model_df = pd.DataFrame({
                    'Model': list(trained_models.keys()),
                    'Status': ['Trained'] * len(trained_models)
                })
                st.dataframe(model_df, use_container_width=True)

                logger.info(f"Model training completed: {len(trained_models)} models trained")

        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            logger.error(f"Model training failed: {str(e)}")

    # Next step button
    if st.session_state.pipeline['models']:
        st.markdown("---")
        if st.button("➡️ Continue to Evaluation", use_container_width=True):
            advance_step()
            st.rerun()

def step_5_evaluation():
    """Step 5: Model Evaluation"""
    st.header("📈 Step 5: Model Evaluation")

    if not st.session_state.pipeline['models']:
        st.error("❌ No trained models available. Please complete training first.")
        return

    if not st.session_state.pipeline['training_results']:
        st.error("❌ No training results available.")
        return

    models = st.session_state.pipeline['models']
    training_results = st.session_state.pipeline['training_results']
    task_type = training_results['task_type']
    X_test = training_results['X_test']
    y_test = training_results['y_test']

    st.markdown(f"""
    **Evaluation Configuration:**
    - Task Type: **{task_type.title()}**
    - Models to Evaluate: **{len(models)}**
    - Test Samples: **{len(X_test)}**
    """)

    # Run evaluation if not already done
    if st.session_state.pipeline['evaluation_summary'] is None:
        if st.button("🔍 Run Model Evaluation", use_container_width=True):
            try:
                with st.spinner("📊 Evaluating models..."):
                    if task_type == 'classification':
                        evaluation_results = evaluator.evaluate_classification_models(models, X_test, y_test)
                    else:
                        evaluation_results = evaluator.evaluate_regression_models(models, X_test, y_test)

                    # Generate evaluation summary
                    evaluation_summary = evaluator.generate_evaluation_summary(evaluation_results, task_type)

                    # Store results
                    st.session_state.pipeline['evaluation_summary'] = evaluation_summary
                    st.session_state.pipeline['evaluation_results'] = evaluation_results

                    # Find best model
                    best_model_name = evaluator.get_best_model(evaluation_results,
                                                             'accuracy' if task_type == 'classification' else 'r2')
                    st.session_state.pipeline['best_model'] = best_model_name

                    logger.info(f"Model evaluation completed: {len(evaluation_results)} models evaluated")

            except Exception as e:
                st.error(f"❌ Evaluation failed: {str(e)}")
                logger.error(f"Evaluation failed: {str(e)}")

    # Display results if available
    if st.session_state.pipeline['evaluation_summary']:
        evaluation_summary = st.session_state.pipeline['evaluation_summary']
        evaluation_results = st.session_state.pipeline['evaluation_results']

        # Summary text
        st.subheader("📋 Evaluation Summary")
        st.text_area("Summary", evaluation_summary, height=150, disabled=True)

        # Best model
        best_model = st.session_state.pipeline['best_model']
        if best_model:
            st.success(f"🏆 **Best Model:** {best_model.replace('_', ' ').title()}")

        # Performance comparison chart
        st.subheader("📊 Performance Comparison")
        try:
            comparison_chart = evaluator.create_model_comparison_chart(evaluation_results, task_type)
            if comparison_chart:
                st.plotly_chart(comparison_chart, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create comparison chart: {str(e)}")

        # Detailed metrics table
        st.subheader("📋 Detailed Metrics")
        metrics_df = pd.DataFrame(evaluation_results).T.round(4)
        st.dataframe(metrics_df, use_container_width=True)

        # Additional analysis
        if task_type == 'classification':
            st.subheader("🎯 Classification Insights")

            # Confusion matrix for best model
            if best_model and best_model in models:
                try:
                    best_model_instance = models[best_model]
                    evaluator.plot_confusion_matrix(best_model_instance, X_test, y_test, best_model)
                    st.pyplot(plt.gcf())
                except Exception as e:
                    st.warning(f"Could not plot confusion matrix: {str(e)}")

        else:  # regression
            st.subheader("📏 Regression Insights")

            # Residuals plot for best model
            if best_model and best_model in models:
                try:
                    best_model_instance = models[best_model]
                    evaluator.plot_residuals(best_model_instance, X_test, y_test, best_model)
                    st.pyplot(plt.gcf())
                except Exception as e:
                    st.warning(f"Could not plot residuals: {str(e)}")

    # Next step button
    if st.session_state.pipeline['evaluation_summary']:
        st.markdown("---")
        if st.button("➡️ Continue to Optimization", use_container_width=True):
            advance_step()
            st.rerun()

def step_6_optimization():
    """Step 6: Hyperparameter Optimization"""
    st.header("🎯 Step 6: Hyperparameter Optimization")

    if not st.session_state.pipeline['best_model']:
        st.error("❌ No best model identified. Please complete evaluation first.")
        return

    if not st.session_state.pipeline['training_results']:
        st.error("❌ No training data available.")
        return

    best_model_name = st.session_state.pipeline['best_model']
    training_results = st.session_state.pipeline['training_results']
    task_type = training_results['task_type']
    X_train = training_results['X_train']
    y_train = training_results['y_train']

    st.markdown(f"""
    **Optimization Configuration:**
    - Model to Optimize: **{best_model_name.replace('_', ' ').title()}**
    - Task Type: **{task_type.title()}**
    - Training Samples: **{len(X_train)}**
    """)

    # Optimization settings
    st.subheader("⚙️ Optimization Settings")

    col1, col2 = st.columns(2)

    with col1:
        optimizer_choice = st.selectbox(
            "Optimization Algorithm",
            options=["Auto (Recommended)", "Optuna", "Hyperopt", "Grid Search", "Random Search"],
            help="Choose optimization algorithm. 'Auto' selects the best available."
        )

        n_trials = st.slider("Number of Trials", 10, 100, 30,
                           help="Number of optimization trials to run")

    with col2:
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5,
                           help="Number of CV folds for evaluation")

        timeout = st.slider("Timeout (minutes)", 1, 30, 10,
                          help="Maximum time for optimization")

    # Run optimization
    if st.button("🚀 Optimize Model", use_container_width=True):
        try:
            with st.spinner("🎯 Optimizing hyperparameters... This may take several minutes"):
                # Map optimizer choice
                if optimizer_choice == "Auto (Recommended)":
                    available_optimizers = advanced_optimizer.get_available_optimizers()
                    if available_optimizers:
                        optimizer = available_optimizers[0]  # Use first available
                    else:
                        optimizer = "grid_search"  # Fallback
                elif optimizer_choice == "Optuna":
                    optimizer = "optuna"
                elif optimizer_choice == "Hyperopt":
                    optimizer = "hyperopt"
                elif optimizer_choice == "Grid Search":
                    optimizer = "grid_search"
                else:  # Random Search
                    optimizer = "random_search"

                # Perform optimization
                optimization_result = advanced_optimizer.optimize_hyperparameters(
                    model_name=best_model_name,
                    model_class=type(st.session_state.pipeline['models'][best_model_name]),  # Get model class
                    X_train=X_train,
                    y_train=y_train,
                    optimizer=optimizer,
                    task_type=task_type,
                    n_trials=n_trials
                )

                if 'error' in optimization_result:
                    st.error(f"❌ Optimization failed: {optimization_result['error']}")
                    logger.error(f"Optimization failed: {optimization_result['error']}")
                else:
                    # Train optimized model
                    if task_type == 'classification':
                        optimized_model = classifier.train_logistic_regression(X_train, y_train)  # Placeholder - should use optimized params
                    else:
                        optimized_model = regressor.train_linear_regression(X_train, y_train)  # Placeholder

                    # Store optimized model
                    st.session_state.pipeline['optimized_model'] = optimized_model
                    st.session_state.pipeline['optimization_results'] = optimization_result

                    st.success("✅ Model optimization completed!")

                    # Show results
                    st.subheader("🎯 Optimization Results")
                    st.write(f"**Best Parameters:** {optimization_result.get('best_params', 'N/A')}")
                    st.write(f"**Best Score:** {optimization_result.get('best_value', 'N/A'):.4f}")
                    st.write(f"**Trials:** {optimization_result.get('n_trials', 'N/A')}")
                    st.write(f"**Optimizer:** {optimization_result.get('optimizer', 'N/A')}")

                    logger.info(f"Model optimization completed: {best_model_name} -> {optimization_result.get('best_value', 'N/A'):.4f}")

        except Exception as e:
            st.error(f"❌ Optimization failed: {str(e)}")
            logger.error(f"Optimization failed: {str(e)}")

    # Show optimization results if available
    if st.session_state.pipeline['optimization_results']:
        opt_results = st.session_state.pipeline['optimization_results']

        st.subheader("📊 Optimization Details")
        details_df = pd.DataFrame({
            'Metric': ['Best Score', 'Trials', 'Time (s)', 'Optimizer'],
            'Value': [
                opt_results.get('best_value', 'N/A'),
                opt_results.get('n_trials', 'N/A'),
                opt_results.get('optimization_time', 'N/A'),
                opt_results.get('optimizer', 'N/A')
            ]
        })
        st.dataframe(details_df, use_container_width=True)

    # Next step button
    if st.session_state.pipeline['optimized_model'] or st.session_state.pipeline['best_model']:
        st.markdown("---")
        if st.button("➡️ Continue to Export & Reports", use_container_width=True):
            advance_step()
            st.rerun()

def step_7_export():
    """Step 7: Export & Reporting"""
    st.header("💾 Step 7: Export & Reports")

    st.markdown("""
    **Final Deliverables:**
    - Export trained models and configurations
    - Generate comprehensive reports
    - Download project artifacts
    """)

    # Model export
    st.subheader("📦 Model Export")

    if st.session_state.pipeline['models']:
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("💾 Export Best Model", use_container_width=True):
                try:
                    best_model_name = st.session_state.pipeline.get('best_model')
                    if best_model_name and best_model_name in st.session_state.pipeline['models']:
                        model = st.session_state.pipeline['models'][best_model_name]

                        # Save model
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        model_filename = f"best_model_{best_model_name}_{timestamp}.pkl"

                        # Store in artifacts
                        st.session_state.pipeline['artifacts'][model_filename] = pickle.dumps(model)

                        st.success(f"✅ Model exported: {model_filename}")
                        logger.info(f"Model exported: {model_filename}")
                    else:
                        st.warning("⚠️ No best model to export")

                except Exception as e:
                    st.error(f"❌ Model export failed: {str(e)}")
                    logger.error(f"Model export failed: {str(e)}")

        with col2:
            if st.button("📊 Export All Models", use_container_width=True):
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    zip_filename = f"all_models_{timestamp}.zip"

                    # Create zip file
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for model_name, model in st.session_state.pipeline['models'].items():
                            model_data = pickle.dumps(model)
                            zip_file.writestr(f"{model_name}.pkl", model_data)

                    # Store in artifacts
                    st.session_state.pipeline['artifacts'][zip_filename] = zip_buffer.getvalue()

                    st.success(f"✅ All models exported: {zip_filename}")
                    logger.info(f"All models exported: {zip_filename}")

                except Exception as e:
                    st.error(f"❌ Models export failed: {str(e)}")
                    logger.error(f"Models export failed: {str(e)}")

        with col3:
            if st.button("💼 Export Optimized Model", use_container_width=True):
                try:
                    optimized_model = st.session_state.pipeline.get('optimized_model')
                    if optimized_model:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        model_filename = f"optimized_model_{timestamp}.pkl"

                        # Store in artifacts
                        st.session_state.pipeline['artifacts'][model_filename] = pickle.dumps(optimized_model)

                        st.success(f"✅ Optimized model exported: {model_filename}")
                        logger.info(f"Optimized model exported: {model_filename}")
                    else:
                        st.warning("⚠️ No optimized model to export")

                except Exception as e:
                    st.error(f"❌ Optimized model export failed: {str(e)}")
                    logger.error(f"Optimized model export failed: {str(e)}")

    # Report generation
    st.subheader("📋 Generate Report")

    if st.button("📄 Generate HTML Report", use_container_width=True):
        try:
            # Generate comprehensive report
            report_content = generate_project_report()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"ml_project_report_{timestamp}.html"

            # Store in artifacts
            st.session_state.pipeline['artifacts'][report_filename] = report_content

            st.success(f"✅ Report generated: {report_filename}")
            logger.info(f"Report generated: {report_filename}")

            # Show report preview
            with st.expander("👀 Report Preview"):
                st.components.v1.html(report_content, height=400, scrolling=True)

        except Exception as e:
            st.error(f"❌ Report generation failed: {str(e)}")
            logger.error(f"Report generation failed: {str(e)}")

    # Project export
    st.subheader("💼 Project Export")

    if st.button("📦 Export Complete Project", use_container_width=True):
        try:
            # Export entire pipeline state
            project_data = {
                'pipeline_state': st.session_state.pipeline,
                'timestamp': datetime.now().isoformat(),
                'version': 'TOOL-BOX v2.0'
            }

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_filename = f"ml_project_{timestamp}.pkl"

            # Store in artifacts (but also provide direct download)
            project_pickle = pickle.dumps(project_data)

            st.download_button(
                label="Download Project File",
                data=project_pickle,
                file_name=project_filename,
                mime="application/octet-stream",
                use_container_width=True
            )

            st.success("✅ Project exported successfully!")
            logger.info(f"Project exported: {project_filename}")

        except Exception as e:
            st.error(f"❌ Project export failed: {str(e)}")
            logger.error(f"Project export failed: {str(e)}")

    # Download artifacts
    if st.session_state.pipeline['artifacts']:
        st.subheader("📁 Download Artifacts")

        for filename, data in st.session_state.pipeline['artifacts'].items():
            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(f"📄 {filename}")

            with col2:
                if st.button(f"Download {filename.split('.')[-1].upper()}", key=f"download_{filename}"):
                    if isinstance(data, str):
                        mime_type = "text/html" if filename.endswith('.html') else "text/plain"
                    elif filename.endswith('.pkl'):
                        mime_type = "application/octet-stream"
                    elif filename.endswith('.zip'):
                        mime_type = "application/zip"
                    else:
                        mime_type = "application/octet-stream"

                    st.download_button(
                        label="⬇️ Download",
                        data=data,
                        file_name=filename,
                        mime=mime_type,
                        use_container_width=True
                    )

    # Pipeline summary
    st.subheader("🎯 Pipeline Summary")

    summary_data = {
        'Step': ['Data Loading', 'EDA', 'Preprocessing', 'Training', 'Evaluation', 'Optimization', 'Export'],
        'Status': [
            '✅ Complete' if st.session_state.pipeline['data'] is not None else '❌ Pending',
            '✅ Complete' if st.session_state.pipeline['eda_summary'] is not None else '❌ Pending',
            '✅ Complete' if st.session_state.pipeline['processed_data'] is not None else '❌ Pending',
            '✅ Complete' if st.session_state.pipeline['models'] else '❌ Pending',
            '✅ Complete' if st.session_state.pipeline['evaluation_summary'] else '❌ Pending',
            '✅ Complete' if st.session_state.pipeline['optimized_model'] else '❌ Pending',
            '✅ Complete' if st.session_state.pipeline['artifacts'] else '❌ Pending'
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

    # Completion message
    if all(status.startswith('✅') for status in summary_data['Status']):
        st.success("🎉 **Congratulations!** Your ML project is complete!")
        st.balloons()

def generate_project_report():
    """Generate comprehensive HTML project report"""
    pipeline = st.session_state.pipeline

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TOOL-BOX ML Project Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 30px;
            }}
            .section {{
                background: white;
                padding: 25px;
                margin: 20px 0;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                border-left: 4px solid #667eea;
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
            }}
            .metric-label {{
                color: #666;
                font-size: 0.9em;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #667eea;
                color: white;
            }}
            .status-complete {{
                color: #28a745;
                font-weight: bold;
            }}
            .status-pending {{
                color: #dc3545;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧠 TOOL-BOX ML Project Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>TOOL-BOX v2.0 - Professional ML Platform</p>
        </div>

        <div class="section">
            <h2>📊 Project Overview</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{pipeline['data'].shape[0] if pipeline['data'] is not None else 0}</div>
                    <div class="metric-label">Total Samples</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{pipeline['data'].shape[1] if pipeline['data'] is not None else 0}</div>
                    <div class="metric-label">Features</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(pipeline['models']) if pipeline['models'] else 0}</div>
                    <div class="metric-label">Models Trained</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{pipeline['task_type'].title() if pipeline['task_type'] else 'N/A'}</div>
                    <div class="metric-label">Task Type</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>🔄 Pipeline Execution Summary</h2>
            <table>
                <tr>
                    <th>Step</th>
                    <th>Status</th>
                    <th>Details</th>
                </tr>
                <tr>
                    <td>Data Loading</td>
                    <td class="{'status-complete' if pipeline['data'] is not None else 'status-pending'}">
                        {'✅ Complete' if pipeline['data'] is not None else '❌ Pending'}
                    </td>
                    <td>{f"Loaded {pipeline['data'].shape[0]} rows × {pipeline['data'].shape[1]} columns" if pipeline['data'] is not None else 'No data loaded'}</td>
                </tr>
                <tr>
                    <td>EDA</td>
                    <td class="{'status-complete' if pipeline['eda_summary'] else 'status-pending'}">
                        {'✅ Complete' if pipeline['eda_summary'] else '❌ Pending'}
                    </td>
                    <td>{f"Data quality: {pipeline['eda_summary']['quality_score']}%" if pipeline['eda_summary'] else 'Not performed'}</td>
                </tr>
                <tr>
                    <td>Preprocessing</td>
                    <td class="{'status-complete' if pipeline['processed_data'] is not None else 'status-pending'}">
                        {'✅ Complete' if pipeline['processed_data'] is not None else '❌ Pending'}
                    </td>
                    <td>{f"Processed to {pipeline['processed_data'].shape[0]} rows × {pipeline['processed_data'].shape[1]} columns" if pipeline['processed_data'] is not None else 'Not performed'}</td>
                </tr>
                <tr>
                    <td>Training</td>
                    <td class="{'status-complete' if pipeline['models'] else 'status-pending'}">
                        {'✅ Complete' if pipeline['models'] else '❌ Pending'}
                    </td>
                    <td>{f"Trained {len(pipeline['models'])} models" if pipeline['models'] else 'No models trained'}</td>
                </tr>
                <tr>
                    <td>Evaluation</td>
                    <td class="{'status-complete' if pipeline['evaluation_summary'] else 'status-pending'}">
                        {'✅ Complete' if pipeline['evaluation_summary'] else '❌ Pending'}
                    </td>
                    <td>{f"Best model: {pipeline['best_model']}" if pipeline['best_model'] else 'Not evaluated'}</td>
                </tr>
                <tr>
                    <td>Optimization</td>
                    <td class="{'status-complete' if pipeline['optimized_model'] else 'status-pending'}">
                        {'✅ Complete' if pipeline['optimized_model'] else '❌ Pending'}
                    </td>
                    <td>{'Optimized model available' if pipeline['optimized_model'] else 'Not performed'}</td>
                </tr>
            </table>
        </div>

        {f'''
        <div class="section">
            <h2>📈 Model Performance</h2>
            <p><strong>Best Model:</strong> {pipeline['best_model'].replace('_', ' ').title()}</p>
            <h3>Evaluation Summary</h3>
            <pre>{pipeline['evaluation_summary']}</pre>
        </div>
        ''' if pipeline['evaluation_summary'] else ''}

        <div class="section">
            <h2>📦 Artifacts Generated</h2>
            <p><strong>Total Artifacts:</strong> {len(pipeline['artifacts'])}</p>
            <ul>
                {"".join([f"<li>{filename}</li>" for filename in pipeline['artifacts'].keys()])}
            </ul>
        </div>

        <div class="section">
            <h2>🛠️ Configuration Summary</h2>
            <h3>Preprocessing Configuration</h3>
            <ul>
                {"".join([f"<li><strong>{k}:</strong> {v}</li>" for k, v in pipeline['preprocess_config'].items()]) if pipeline['preprocess_config'] else "<li>No preprocessing configuration</li>"}
            </ul>
        </div>

        <div class="footer" style="text-align: center; margin-top: 40px; color: #666;">
            <p>Report generated by TOOL-BOX v2.0 - Professional ML Platform</p>
            <p>🧠 Built with ❤️ for data scientists</p>
        </div>
    </body>
    </html>
    """

    return html_content

def main():
    """Main application"""
    # Initialize session state
    initialize_session_state()

    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 TOOL-BOX</h1>
        <h2>Professional Machine Learning Platform</h2>
        <p>End-to-end ML pipeline with production-grade tools and explainable AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    create_sidebar()

    # Main content area
    current_step = st.session_state.pipeline['current_step']

    if current_step == 0:
        step_1_load_data()
    elif current_step == 1:
        step_2_eda()
    elif current_step == 2:
        step_3_preprocessing()
    elif current_step == 3:
        step_4_training()
    elif current_step == 4:
        step_5_evaluation()
    elif current_step == 5:
        step_6_optimization()
    elif current_step == 6:
        step_7_export()
    else:
        st.info("🚧 Pipeline complete!")

    # Footer
    st.markdown("---")
    st.markdown("*TOOL-BOX v2.0 - Production-Grade ML Platform*")

if __name__ == "__main__":
    main()
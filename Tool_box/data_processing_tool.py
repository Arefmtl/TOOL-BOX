"""
Data Processing Tool - Comprehensive data cleaning, exploration, and preprocessing utilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class DataProcessingTool:
    """A comprehensive tool for data processing, cleaning, and exploration."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.encoders = {}

    @staticmethod
    def load_data(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Load data from various file formats.

        Args:
            file_path: Path to the data file
            encoding: File encoding (default: 'utf-8')

        Returns:
            Loaded DataFrame
        """
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, encoding=encoding)
        elif file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Supported: .csv, .xlsx, .xls, .json")

    def data_overview(self, data: pd.DataFrame) -> Dict:
        """
        Comprehensive data overview including statistics and characteristics.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary containing data overview information
        """
        overview = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'duplicates': data.duplicated().sum(),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'numerical_summary': data.describe().to_dict()
        }
        return overview

    def detect_and_handle_outliers(self, data: pd.DataFrame,
                                 columns: Optional[List[str]] = None,
                                 method: str = 'iqr',
                                 threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect and handle outliers using IQR or Z-score methods.

        Args:
            data: Input DataFrame
            columns: Columns to check for outliers (default: all numerical)
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outliers handled
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        cleaned_data = data.copy()

        for col in columns:
            if method == 'iqr':
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
            elif method == 'zscore':
                z_scores = stats.zscore(cleaned_data[col].dropna())
                abs_z_scores = np.abs(z_scores)
                lower_bound = cleaned_data[col].mean() - threshold * cleaned_data[col].std()
                upper_bound = cleaned_data[col].mean() + threshold * cleaned_data[col].std()

            # Replace outliers with median
            median_val = cleaned_data[col].median()
            cleaned_data[col] = np.where(
                (cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound),
                median_val,
                cleaned_data[col]
            )

        return cleaned_data

    def clean_data(self, data: pd.DataFrame,
                  handle_missing: str = 'auto',
                  remove_duplicates: bool = True,
                  handle_outliers: bool = True) -> pd.DataFrame:
        """
        Comprehensive data cleaning pipeline.

        Args:
            data: Input DataFrame
            handle_missing: 'auto', 'drop', 'median', 'mean', 'mode', or 'interpolate'
            remove_duplicates: Whether to remove duplicate rows
            handle_outliers: Whether to handle outliers

        Returns:
            Cleaned DataFrame
        """
        cleaned_data = data.copy()

        # Remove duplicates
        if remove_duplicates:
            initial_shape = cleaned_data.shape
            cleaned_data = cleaned_data.drop_duplicates()
            print(f"Removed {initial_shape[0] - cleaned_data.shape[0]} duplicate rows")

        # Handle missing values
        if handle_missing == 'auto':
            # Auto-detect and handle missing values
            for col in cleaned_data.columns:
                if cleaned_data[col].isnull().sum() > 0:
                    if cleaned_data[col].dtype in ['int64', 'float64']:
                        cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
                    else:
                        cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mode().iloc[0] if not cleaned_data[col].mode().empty else 'Unknown')
        elif handle_missing == 'drop':
            cleaned_data = cleaned_data.dropna()
        elif handle_missing == 'median':
            num_cols = cleaned_data.select_dtypes(include=[np.number]).columns
            cleaned_data[num_cols] = cleaned_data[num_cols].fillna(cleaned_data[num_cols].median())
        elif handle_missing == 'mean':
            num_cols = cleaned_data.select_dtypes(include=[np.number]).columns
            cleaned_data[num_cols] = cleaned_data[num_cols].fillna(cleaned_data[num_cols].mean())
        elif handle_missing == 'mode':
            for col in cleaned_data.columns:
                if cleaned_data[col].isnull().sum() > 0:
                    cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mode().iloc[0])

        # Handle outliers
        if handle_outliers:
            cleaned_data = self.detect_and_handle_outliers(cleaned_data)

        # Convert date columns
        for col in cleaned_data.columns:
            if 'date' in col.lower():
                try:
                    cleaned_data[col] = pd.to_datetime(cleaned_data[col])
                except:
                    pass

        return cleaned_data

    def explore_data(self, data: pd.DataFrame, plot: bool = True) -> Dict:
        """
        Exploratory data analysis with visualizations.

        Args:
            data: Input DataFrame
            plot: Whether to generate plots

        Returns:
            Dictionary containing EDA results
        """
        results = {}

        # Basic statistics
        results['overview'] = self.data_overview(data)

        # Correlation analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            results['correlation'] = corr_matrix

            if plot:
                plt.figure(figsize=(12, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('Correlation Matrix')
                plt.show()

        # Distribution plots
        if plot and len(numeric_cols) > 0:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            axes = axes.ravel()

            for i, col in enumerate(numeric_cols[:6]):
                if i < len(axes):
                    sns.histplot(data[col], ax=axes[i], kde=True)
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.show()

        # Categorical analysis
        cat_cols = data.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            results['categorical_summary'] = {}
            for col in cat_cols:
                results['categorical_summary'][col] = data[col].value_counts().head(10).to_dict()

        return results

    def encode_categorical(self, data: pd.DataFrame,
                          columns: Optional[List[str]] = None,
                          method: str = 'onehot') -> Tuple[pd.DataFrame, Dict]:
        """
        Encode categorical variables.

        Args:
            data: Input DataFrame
            columns: Columns to encode (default: auto-detect)
            method: 'onehot' or 'label'

        Returns:
            Tuple of (encoded_data, encoders_dict)
        """
        if columns is None:
            columns = data.select_dtypes(include=['object']).columns.tolist()

        encoded_data = data.copy()

        if method == 'onehot':
            for col in columns:
                if col in encoded_data.columns:
                    encoder = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
                    encoded_values = encoder.fit_transform(encoded_data[[col]])
                    feature_names = encoder.get_feature_names_out([col])
                    encoded_df = pd.DataFrame(encoded_values, columns=feature_names, index=encoded_data.index)
                    encoded_data = pd.concat([encoded_data.drop(col, axis=1), encoded_df], axis=1)
                    self.encoders[col] = encoder
        elif method == 'label':
            for col in columns:
                if col in encoded_data.columns:
                    encoder = LabelEncoder()
                    encoded_data[col] = encoder.fit_transform(encoded_data[col].astype(str))
                    self.encoders[col] = encoder

        return encoded_data, self.encoders

    def scale_features(self, data: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features.

        Args:
            data: Input DataFrame
            columns: Columns to scale (default: all numerical)
            method: 'standard', 'minmax', or 'robust'

        Returns:
            Scaled DataFrame
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        scaled_data = data.copy()

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()

        scaled_values = scaler.fit_transform(scaled_data[columns])
        scaled_data[columns] = scaled_values

        return scaled_data

    def apply_pca(self, data: pd.DataFrame,
                 n_components: Optional[int] = None,
                 explained_variance: float = 0.95) -> Tuple[pd.DataFrame, PCA]:
        """
        Apply Principal Component Analysis for dimensionality reduction.

        Args:
            data: Input DataFrame (only numerical columns)
            n_components: Number of components or None for auto
            explained_variance: Target explained variance ratio

        Returns:
            Tuple of (pca_data, pca_model)
        """
        numeric_data = data.select_dtypes(include=[np.number])

        if n_components is None:
            pca = PCA(n_components=explained_variance, svd_solver='full')
        else:
            pca = PCA(n_components=n_components)

        pca_data = pca.fit_transform(numeric_data)
        pca_columns = [f'PC{i+1}' for i in range(pca_data.shape[1])]
        pca_df = pd.DataFrame(pca_data, columns=pca_columns, index=data.index)

        return pca_df, pca

    def create_derived_features(self, data: pd.DataFrame,
                               operations: List[Dict]) -> pd.DataFrame:
        """
        Create derived features from existing columns.

        Args:
            data: Input DataFrame
            operations: List of operation dictionaries
                       Example: [{'name': 'ratio', 'cols': ['num1', 'num2'], 'op': 'div'}]

        Returns:
            DataFrame with new features
        """
        enhanced_data = data.copy()

        for op in operations:
            name = op.get('name', f"{op['op']}_{'_'.join(op['cols'])}")
            cols = op['cols']
            operation = op['op']

            if operation == 'div' and len(cols) == 2:
                enhanced_data[name] = enhanced_data[cols[0]] / enhanced_data[cols[1]].replace(0, 1e-10)
            elif operation == 'mul' and len(cols) >= 2:
                enhanced_data[name] = enhanced_data[cols[0]]
                for col in cols[1:]:
                    enhanced_data[name] *= enhanced_data[col]
            elif operation == 'add' and len(cols) >= 2:
                enhanced_data[name] = enhanced_data[cols[0]]
                for col in cols[1:]:
                    enhanced_data[name] += enhanced_data[col]
            elif operation == 'sub' and len(cols) == 2:
                enhanced_data[name] = enhanced_data[cols[0]] - enhanced_data[cols[1]]

        return enhanced_data

    def prepare_data_for_ml(self, data: pd.DataFrame,
                           target_column: str = None,
                           test_size: float = 0.2,
                           preprocessing_steps: List[str] = None) -> Dict:
        """
        Complete pipeline for preparing data for machine learning.

        Args:
            data: Raw input DataFrame
            target_column: Name of target column (for supervised learning)
            test_size: Test set size for train/test split
            preprocessing_steps: List of preprocessing steps to apply

        Returns:
            Dictionary containing processed data splits
        """
        if preprocessing_steps is None:
            preprocessing_steps = ['clean', 'encode', 'scale']

        processed_data = data.copy()

        # Clean data
        if 'clean' in preprocessing_steps or 'all' in preprocessing_steps:
            processed_data = self.clean_data(processed_data)

        # Encode categorical
        if 'encode' in preprocessing_steps or 'all' in preprocessing_steps:
            processed_data, _ = self.encode_categorical(processed_data)

        # Scale features
        if 'scale' in preprocessing_steps or 'all' in preprocessing_steps:
            processed_data = self.scale_features(processed_data)

        # Split data
        if target_column and target_column in processed_data.columns:
            X = processed_data.drop(target_column, axis=1)
            y = processed_data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state)

            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'full_processed': processed_data
            }
        else:
            return {'processed_data': processed_data}

    def generate_data_report(self, data: pd.DataFrame,
                           output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive data quality report.

        Args:
            data: Input DataFrame
            output_path: Path to save HTML report (optional)

        Returns:
            HTML report as string
        """
        overview = self.data_overview(data)

        html_report = f"""
        <html>
        <head>
            <title>Data Quality Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; color: #2e7d32; }}
            </style>
        </head>
        <body>
            <h1>Data Quality Report</h1>
            <h2>Overview</h2>
            <p><strong>Shape:</strong> {overview['shape'][0]} rows, {overview['shape'][1]} columns</p>
            <p><strong>Memory Usage:</strong> {overview['memory_usage'] / 1024:.2f} KB</p>
            <p><strong>Duplicates:</strong> {overview['duplicates']}</p>

            <h2>Missing Values</h2>
            <table>
                <tr><th>Column</th><th>Missing Count</th><th>Missing Percentage</th></tr>
                {"".join([f"<tr><td>{col}</td><td>{count}</td><td>{count/overview['shape'][0]*100:.1f}%</td></tr>"
                         for col, count in overview['missing_values'].items() if count > 0])}
            </table>

            <h2>Data Types</h2>
            <table>
                <tr><th>Column</th><th>Data Type</th></tr>
                {"".join([f"<tr><td>{col}</td><td>{dtype}</td></tr>"
                         for col, dtype in overview['dtypes'].items()])}
            </table>
        </body>
        </html>
        """

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_report)

        return html_report

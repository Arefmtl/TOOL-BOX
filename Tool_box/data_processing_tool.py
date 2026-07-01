"""
Data Processing Tool - Comprehensive data cleaning, exploration, and preprocessing utilities.

Pipeline Step: Load Data → EDA → Preprocess → Split
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
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

from .decorators import step


class DataProcessingTool:
    """A comprehensive tool for data processing, cleaning, and exploration."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.encoders = {}

    @staticmethod
    @step('Data Loading')
    def load_data(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
        """Load data from various file formats.

        Args:
            file_path: Path to the file (.csv, .xlsx, .xls, .json)
            encoding: File encoding (default: utf-8)

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

    @step('Data Overview')
    def data_overview(self, data: pd.DataFrame) -> Dict:
        """Comprehensive data overview including statistics and characteristics.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary with shape, columns, dtypes, missing values, duplicates, etc.
        """
        overview = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
            'duplicates': data.duplicated().sum(),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'numerical_summary': data.describe(include=[np.number]).to_dict() if len(data.select_dtypes(include=[np.number]).columns) > 0 else {},
            'categorical_summary': data.describe(include=['object', 'category']).to_dict() if len(data.select_dtypes(include=['object', 'category']).columns) > 0 else {}
        }
        return overview

    @step('EDA Summary')
    def generate_eda_summary(self, data: pd.DataFrame) -> Dict:
        """Generate comprehensive EDA summary for pipeline state.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary with quality score, correlations, outlier counts, recommendations
        """
        try:
            overview = self.data_overview(data)

            missing_pct = sum(overview['missing_values'].values()) / (overview['shape'][0] * overview['shape'][1]) * 100
            duplicate_pct = overview['duplicates'] / overview['shape'][0] * 100

            quality_score = 100
            quality_score -= min(missing_pct * 2, 40)
            quality_score -= min(duplicate_pct * 10, 20)

            numeric_cols = data.select_dtypes(include=[np.number]).columns
            cat_cols = data.select_dtypes(include=['object', 'category']).columns

            # Numeric analysis
            numeric_stats = {}
            corr_matrix = pd.DataFrame()
            strong_corr = pd.DataFrame()
            if len(numeric_cols) > 0:
                numeric_stats = data[numeric_cols].describe().to_dict()

                if len(numeric_cols) > 1:
                    corr_matrix = data[numeric_cols].corr()
                    upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool))
                    strong_corr = upper.stack().reset_index()
                    strong_corr.columns = ['Feature 1', 'Feature 2', 'Correlation']
                    strong_corr['Abs Correlation'] = strong_corr['Correlation'].abs()
                    strong_corr = strong_corr[strong_corr['Abs Correlation'] > 0.7]

            # Categorical analysis
            cat_summary = {}
            target_candidates = []
            for col in data.columns:
                unique_vals = data[col].nunique()
                dtype = data[col].dtype

                if unique_vals <= 20 or dtype in ['object', 'category']:
                    target_candidates.append({
                        'column': col,
                        'type': 'classification' if unique_vals <= 20 else 'potential_classification',
                        'unique_values': unique_vals,
                        'reason': '<=20 unique values' if unique_vals <= 20 else f'Categorical with {unique_vals} values'
                    })

                if col in cat_cols:
                    cat_summary[col] = {
                        'cardinality': unique_vals,
                        'top_categories': data[col].value_counts().head(5).to_dict(),
                        'missing_pct': data[col].isnull().sum() / len(data) * 100
                    }

            # Vectorized outlier analysis
            outlier_counts = {}
            if len(numeric_cols) > 0:
                Q1 = data[numeric_cols].quantile(0.25)
                Q3 = data[numeric_cols].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outlier_counts = ((data[numeric_cols] < lower) | (data[numeric_cols] > upper)).sum().to_dict()

            eda_summary = {
                'overview': overview,
                'quality_score': max(0, round(quality_score)),
                'data_types': {
                    'numeric': len(numeric_cols),
                    'categorical': len(cat_cols),
                    'other': len(data.columns) - len(numeric_cols) - len(cat_cols)
                },
                'numeric_analysis': {
                    'stats': numeric_stats,
                    'correlation_matrix': corr_matrix.to_dict() if not corr_matrix.empty else {},
                    'strong_correlations': strong_corr.to_dict('records') if not strong_corr.empty else [],
                    'outlier_counts': outlier_counts
                },
                'categorical_analysis': cat_summary,
                'target_candidates': target_candidates,
                'recommendations': self._generate_eda_recommendations(
                    missing_pct, duplicate_pct, len(numeric_cols), len(cat_cols),
                    outlier_counts, strong_corr if not strong_corr.empty else pd.DataFrame()
                )
            }

            return eda_summary

        except Exception as e:
            return {'error': str(e)}

    def _generate_eda_recommendations(self, missing_pct: float, duplicate_pct: float,
                                      n_numeric: int, n_cat: int,
                                      outlier_counts: Dict, strong_corr: pd.DataFrame) -> List[str]:
        """Generate preprocessing recommendations based on EDA."""
        recommendations = []

        if missing_pct > 5:
            recommendations.append("High missing data detected - consider imputation strategies")
        if duplicate_pct > 1:
            recommendations.append("Duplicate rows detected - consider deduplication")
        if outlier_counts and sum(outlier_counts.values()) > 0:
            recommendations.append("Outliers detected - consider robust preprocessing methods")
        if not strong_corr.empty:
            recommendations.append("Strong feature correlations detected - consider dimensionality reduction")
        if n_numeric >= 3:
            recommendations.append("Multiple numeric features - consider correlation-based feature selection")

        if not recommendations:
            recommendations.append("Dataset appears clean - standard preprocessing pipeline should suffice")

        return recommendations

    @step('Data Cleaning')
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset by removing duplicates and handling missing values.

        Args:
            data: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        cleaned_data = data.copy()

        # Remove duplicate rows
        n_before = len(cleaned_data)
        cleaned_data = cleaned_data.drop_duplicates()
        n_removed = n_before - len(cleaned_data)

        # Handle missing values for numeric columns (fill with median) — vectorized
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            median_vals = cleaned_data[numeric_cols].median()
            cleaned_data[numeric_cols] = cleaned_data[numeric_cols].fillna(median_vals)

        # Handle missing values for categorical columns (fill with mode)
        cat_cols = cleaned_data.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if cleaned_data[col].isnull().sum() > 0:
                mode_val = cleaned_data[col].mode()
                if not mode_val.empty:
                    cleaned_data.loc[:, col] = cleaned_data[col].fillna(mode_val[0])

        return cleaned_data

    @step('Feature Encoding')
    def encode_categorical(self, data: pd.DataFrame,
                           method: str = 'one_hot',
                           columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
        """Encode categorical variables.

        Args:
            data: Input DataFrame
            method: Encoding method ('one_hot', 'label')
            columns: Specific columns to encode (None = all categorical)

        Returns:
            Tuple of (encoded DataFrame, encoding info dict)
        """
        encoded_data = data.copy()
        encoding_info = {}

        if columns is None:
            columns = encoded_data.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in columns:
            if col not in encoded_data.columns:
                continue

            if method == 'one_hot' or method == 'auto':
                dummies = pd.get_dummies(encoded_data[col], prefix=col, drop_first=True)
                encoded_data = pd.concat([encoded_data.drop(col, axis=1), dummies], axis=1)
                encoding_info[col] = {
                    'original_unique': data[col].nunique(),
                    'new_features': len(dummies.columns),
                    'method': 'one_hot'
                }
            elif method == 'label':
                le = LabelEncoder()
                encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))
                self.encoders[col] = le
                encoding_info[col] = {
                    'original_unique': data[col].nunique(),
                    'new_features': 1,
                    'method': 'label',
                    'classes': le.classes_.tolist()
                }

        return encoded_data, encoding_info

    @step('Feature Scaling')
    def scale_features(self, data: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Scale numeric features.

        Args:
            data: Input DataFrame
            method: Scaling method ('standard', 'minmax', 'robust')

        Returns:
            Scaled DataFrame
        """
        scaled_data = data.copy()
        numeric_cols = scaled_data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return scaled_data

        if method == 'standard':
            scaled_features = self.scaler.fit_transform(scaled_data[numeric_cols])
            scaled_data[numeric_cols] = scaled_features
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            mm_scaler = MinMaxScaler()
            scaled_features = mm_scaler.fit_transform(scaled_data[numeric_cols])
            scaled_data[numeric_cols] = scaled_features
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            r_scaler = RobustScaler()
            scaled_features = r_scaler.fit_transform(scaled_data[numeric_cols])
            scaled_data[numeric_cols] = scaled_features

        return scaled_data

    @step('Outlier Handling')
    def handle_outliers(self, data: pd.DataFrame, method: str = 'iqr',
                        columns: Optional[List[str]] = None,
                        threshold: float = 1.5) -> pd.DataFrame:
        """Handle outliers in numeric columns. Optimized with vectorized operations.

        Args:
            data: Input DataFrame
            method: 'iqr' (clip) or 'zscore' (remove)
            columns: Specific columns to process (None = all numeric)
            threshold: IQR multiplier (default: 1.5) or z-score threshold (default: 3)

        Returns:
            DataFrame with outliers handled
        """
        cleaned_data = data.copy()

        if columns is None:
            columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()

        if len(columns) == 0:
            return cleaned_data

        numeric_data = cleaned_data[columns]

        if method == 'iqr':
            # Vectorized IQR clipping — no loop needed
            Q1 = numeric_data.quantile(0.25)
            Q3 = numeric_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            cleaned_data[columns] = numeric_data.clip(lower_bound, upper_bound, axis=1)

        elif method == 'zscore':
            # Z-score filtering — removes rows where ANY selected col exceeds threshold
            z_scores = np.abs(stats.zscore(numeric_data, nan_policy='omit'))
            mask = (z_scores < threshold).all(axis=1)
            cleaned_data = cleaned_data[mask]

        return cleaned_data

    @step('Data Export')
    def export_processed_dataset(self, data: pd.DataFrame, output_path: str,
                                 format: str = 'csv') -> bool:
        """Export processed dataset to file.

        Args:
            data: DataFrame to export
            output_path: Output file path
            format: Export format ('csv', 'parquet', 'json')

        Returns:
            True if successful, False otherwise
        """
        try:
            if format.lower() == 'csv':
                data.to_csv(output_path, index=False)
            elif format.lower() == 'parquet':
                data.to_parquet(output_path, index=False)
            elif format.lower() == 'json':
                data.to_json(output_path, orient='records')
            else:
                raise ValueError(f"Unsupported format: {format}")
            return True
        except Exception as e:
            print(f"Error exporting dataset: {str(e)}")
            return False

    @step('Feature Selection')
    def feature_selection(self, data: pd.DataFrame, target_column: str,
                          method: str = 'correlation', k: int = 10) -> pd.DataFrame:
        """Perform feature selection.

        Args:
            data: Input DataFrame
            target_column: Target column name
            method: 'correlation', 'f_regression', 'f_classif'
            k: Number of features to select

        Returns:
            DataFrame with selected features
        """
        from sklearn.feature_selection import SelectKBest, f_regression, f_classif

        if target_column not in data.columns:
            return data

        X = data.drop(target_column, axis=1)
        y = data[target_column]

        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]

        if len(X_numeric.columns) == 0:
            return data

        if method == 'correlation':
            corr_matrix = X_numeric.corrwith(y).abs().sort_values(ascending=False)
            selected_features = corr_matrix.head(k).index.tolist()
        elif method in ['f_regression', 'f_classif']:
            n_cols = min(k, len(X_numeric.columns))
            if y.dtype in ['int64', 'float64'] and len(y.unique()) > 10:
                selector = SelectKBest(score_func=f_regression, k=n_cols)
            else:
                selector = SelectKBest(score_func=f_classif, k=n_cols)
            X_selected = selector.fit_transform(X_numeric, y)
            selected_features = X_numeric.columns[selector.get_support()].tolist()
        else:
            return data

        selected_features.append(target_column)
        return data[selected_features]

    @step('Auto Clean')
    def auto_clean(self, data: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """Automatically detect data issues and apply best preprocessing.

        Scans data types, missing rates, cardinality, and outliers,
        then applies appropriate cleaning without manual config.

        Args:
            data: Input DataFrame
            target_column: Optional target column (excluded from scaling)

        Returns:
            Cleaned and preprocessed DataFrame
        """
        result = data.copy()

        # Step 1: Remove duplicates
        result = result.drop_duplicates()

        # Step 2: Handle missing values
        for col in result.columns:
            missing_rate = result[col].isnull().mean()
            if missing_rate == 0:
                continue
            if missing_rate > 0.5:
                # Drop column if more than 50% missing
                result = result.drop(col, axis=1)
            elif result[col].dtype in ['object', 'category']:
                result[col] = result[col].fillna(result[col].mode()[0] if not result[col].mode().empty else 'Unknown')
            else:
                result[col] = result[col].fillna(result[col].median())

        # Step 3: Handle outliers (IQR clipping for numeric)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        if target_column and target_column in numeric_cols:
            numeric_cols = numeric_cols.drop(target_column)

        if len(numeric_cols) > 0:
            Q1 = result[numeric_cols].quantile(0.25)
            Q3 = result[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            result[numeric_cols] = result[numeric_cols].clip(lower, upper, axis=1)

        return result

    @step('Train/Test Split')
    def split_data(self, data: pd.DataFrame, target_column: str,
                   test_size: float = 0.2, val_size: float = 0.0,
                   stratify: bool = False) -> Dict:
        """Split data into train/test/val sets. Must be called BEFORE scaling to avoid leakage.

        Args:
            data: Input DataFrame
            target_column: Target column name
            test_size: Test set proportion (default: 0.2)
            val_size: Validation set proportion (default: 0.0)
            stratify: Whether to stratify (for classification)

        Returns:
            Dictionary with X_train, X_test, X_val, y_train, y_test, y_val
        """
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        X = data.drop(target_column, axis=1)
        y = data[target_column]

        stratify_y = y if stratify else None

        if val_size > 0:
            # Split into train+val and test
            test_val_size = test_size + val_size
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=test_val_size, random_state=self.random_state,
                stratify=stratify_y
            )
            # Split temp into val and test
            val_ratio = val_size / test_val_size
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=(test_size / test_val_size),
                random_state=self.random_state,
                stratify=(y_temp if stratify else None)
            )
            return {
                'X_train': X_train, 'X_test': X_test, 'X_val': X_val,
                'y_train': y_train, 'y_test': y_test, 'y_val': y_val
            }
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state,
                stratify=stratify_y
            )
            return {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test
            }

    @step('PCA Dimensionality Reduction')
    def apply_pca(self, data: pd.DataFrame, n_components: Optional[int] = None,
                  target_column: Optional[str] = None) -> pd.DataFrame:
        """Apply PCA for dimensionality reduction.

        Args:
            data: Input DataFrame
            n_components: Number of components (None = auto based on 95% variance)
            target_column: Optional target column to preserve

        Returns:
            DataFrame with PCA features
        """
        if target_column and target_column in data.columns:
            X = data.drop(target_column, axis=1)
            y = data[target_column]
        else:
            X = data
            y = None

        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]

        if len(numeric_cols) < 2:
            return data

        if n_components is None:
            n_components = min(len(numeric_cols), 10)

        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X_numeric)

        pca_columns = [f'PC{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=data.index)

        if y is not None:
            pca_df[target_column] = y.values

        return pca_df

    @step('Clustering Features')
    def add_clustering_features(self, data: pd.DataFrame, n_clusters: int = 3,
                                columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Add clustering-based features.

        Args:
            data: Input DataFrame
            n_clusters: Number of clusters
            columns: Columns to use for clustering

        Returns:
            DataFrame with added cluster features
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        enhanced_data = data.copy()

        if columns is None:
            columns = enhanced_data.select_dtypes(include=[np.number]).columns.tolist()

        if len(columns) < 2:
            return enhanced_data

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(enhanced_data[columns])

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        enhanced_data['cluster'] = clusters
        cluster_dummies = pd.get_dummies(enhanced_data['cluster'], prefix='cluster')
        enhanced_data = pd.concat([enhanced_data, cluster_dummies], axis=1)

        return enhanced_data

    @step('Advanced Preprocessing Pipeline')
    def advanced_preprocessing(self, data: pd.DataFrame, config: Dict) -> Dict:
        """Advanced preprocessing pipeline with configuration.

        Args:
            data: Input DataFrame
            config: Configuration dictionary with preprocessing options

        Returns:
            Dictionary with processed_data, preprocessing_log, and config
        """
        processed_data = data.copy()
        preprocessing_log = []

        if config.get('handle_missing', True):
            missing_method = config.get('missing_method', 'auto')
            if missing_method == 'auto':
                processed_data = self.clean_data(processed_data)
            preprocessing_log.append(f"Handled missing values using {missing_method} method")

        if config.get('handle_outliers', False):
            outlier_method = config.get('outlier_method', 'iqr')
            outlier_columns = config.get('outlier_columns', None)
            processed_data = self.handle_outliers(processed_data, outlier_method, outlier_columns)
            preprocessing_log.append(f"Handled outliers using {outlier_method} method")

        if config.get('feature_selection', False) and config.get('target_column'):
            fs_method = config.get('fs_method', 'correlation')
            k_features = config.get('k_features', 10)
            processed_data = self.feature_selection(processed_data, config['target_column'], fs_method, k_features)
            preprocessing_log.append(f"Selected {k_features} features using {fs_method} method")

        if config.get('add_clustering', False):
            n_clusters = config.get('n_clusters', 3)
            cluster_columns = config.get('cluster_columns', None)
            processed_data = self.add_clustering_features(processed_data, n_clusters, cluster_columns)
            preprocessing_log.append(f"Added {n_clusters} clustering features")

        if config.get('encode_categorical', True):
            enc_method = config.get('encoding_method', 'one_hot')
            processed_data, encoding_info = self.encode_categorical(processed_data, method=enc_method)
            preprocessing_log.append(f"Encoded categorical variables using {enc_method}")

        if config.get('scale_features', True):
            scale_method = config.get('scaling_method', 'standard')
            processed_data = self.scale_features(processed_data, method=scale_method)
            preprocessing_log.append(f"Scaled numeric features using {scale_method}")

        return {
            'processed_data': processed_data,
            'preprocessing_log': preprocessing_log,
            'config': config
        }

    @step('Prepare Data for ML')
    def prepare_data_for_ml(self, data: pd.DataFrame,
                            target_column: str = None,
                            test_size: float = 0.2,
                            preprocessing_steps: List[str] = None) -> Dict:
        """Complete pipeline for preparing data for machine learning.

        Args:
            data: Raw input DataFrame
            target_column: Name of target column
            test_size: Test set size for train/test split
            preprocessing_steps: List of steps ['clean', 'encode', 'scale']

        Returns:
            Dictionary containing processed data splits
        """
        if preprocessing_steps is None:
            preprocessing_steps = ['clean', 'encode', 'scale']

        processed_data = data.copy()

        if 'clean' in preprocessing_steps or 'all' in preprocessing_steps:
            processed_data = self.clean_data(processed_data)

        if 'encode' in preprocessing_steps or 'all' in preprocessing_steps:
            processed_data, _ = self.encode_categorical(processed_data)

        if target_column and target_column in processed_data.columns:
            X = processed_data.drop(target_column, axis=1)
            y = processed_data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state)

            if 'scale' in preprocessing_steps or 'all' in preprocessing_steps:
                numeric_cols = X_train.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    self.scaler.fit(X_train[numeric_cols])
                    X_train = X_train.copy()
                    X_test = X_test.copy()
                    X_train[numeric_cols] = self.scaler.transform(X_train[numeric_cols])
                    X_test[numeric_cols] = self.scaler.transform(X_test[numeric_cols])

            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'full_processed': processed_data
            }
        else:
            if 'scale' in preprocessing_steps or 'all' in preprocessing_steps:
                processed_data = self.scale_features(processed_data)
            return {'processed_data': processed_data}

    # ─── Plotting helpers ─────────────────────────────────────────────

    def plot_missing_values(self, data: pd.DataFrame) -> plt.Figure:
        """Plot missing value heatmap."""
        fig, ax = plt.subplots(figsize=(12, 6))
        missing = data.isnull()
        sns.heatmap(missing, cbar=False, cmap='viridis', yticklabels=False, ax=ax)
        ax.set_title('Missing Values Heatmap')
        plt.tight_layout()
        return fig

    def plot_correlations(self, data: pd.DataFrame) -> plt.Figure:
        """Plot correlation matrix heatmap."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return None
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = data[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu', center=0, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        return fig

    def plot_distributions(self, data: pd.DataFrame, n_cols: int = 4) -> plt.Figure:
        """Plot distribution of numeric features."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns[:8]
        if len(numeric_cols) == 0:
            return None
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            axes[i].hist(data[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_xlabel(col)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        return fig
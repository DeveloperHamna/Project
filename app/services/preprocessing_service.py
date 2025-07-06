"""
Comprehensive Preprocessing Service for Dataset Handling
"""

import pandas as pd
import numpy as np
import json
import zipfile
import chardet
import io
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging
from scipy import stats
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
import category_encoders as ce
from feature_engine.outliers import OutlierTrimmer
from feature_engine.imputation import (
    MeanMedianImputer, CategoricalImputer, RandomSampleImputer
)
from feature_engine.encoding import (
    OneHotEncoder as FEOneHotEncoder,
    OrdinalEncoder as FEOrdinalEncoder
)
from feature_engine.selection import DropFeatures
from feature_engine.transformation import (
    LogTransformer, BoxCoxTransformer, YeoJohnsonTransformer
)

from app.core.config import settings
from app.core.exceptions import DataProcessingError

logger = logging.getLogger(__name__)

class PreprocessingService:
    """Comprehensive preprocessing service for dataset handling"""
    
    def __init__(self):
        self.settings = settings
        self.scalers = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(),
            "MaxAbsScaler": MaxAbsScaler()
        }
        
    async def load_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load dataset from various file formats with automatic detection
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.csv':
                return await self._load_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                return await self._load_excel(file_path)
            elif file_ext == '.json':
                return await self._load_json(file_path)
            elif file_ext == '.tsv':
                return await self._load_tsv(file_path)
            elif file_ext == '.zip':
                return await self._load_zip(file_path)
            else:
                raise DataProcessingError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise DataProcessingError(f"Failed to load dataset: {str(e)}")
    
    async def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV file with automatic encoding and delimiter detection"""
        try:
            # Detect encoding
            encoding = 'utf-8'
            if self.settings.AUTO_DETECT_ENCODING:
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    result = chardet.detect(raw_data)
                    encoding = result['encoding'] or 'utf-8'
            
            # Detect delimiter
            delimiter = ','
            if self.settings.AUTO_DETECT_DELIMITER:
                with open(file_path, 'r', encoding=encoding) as f:
                    first_line = f.readline()
                    for sep in [',', ';', '\t', '|']:
                        if first_line.count(sep) > 0:
                            delimiter = sep
                            break
            
            # Load CSV
            df = pd.read_csv(file_path, encoding=encoding, sep=delimiter)
            
            logger.info(f"Loaded CSV file with encoding: {encoding}, delimiter: '{delimiter}'")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise DataProcessingError(f"Failed to load CSV file: {str(e)}")
    
    async def _load_excel(self, file_path: str) -> pd.DataFrame:
        """Load Excel file"""
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            logger.info(f"Loaded Excel file: {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading Excel file: {str(e)}")
            raise DataProcessingError(f"Failed to load Excel file: {str(e)}")
    
    async def _load_json(self, file_path: str) -> pd.DataFrame:
        """Load JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                else:
                    df = pd.DataFrame([data])
            else:
                raise DataProcessingError("Invalid JSON structure")
            
            logger.info(f"Loaded JSON file: {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading JSON file: {str(e)}")
            raise DataProcessingError(f"Failed to load JSON file: {str(e)}")
    
    async def _load_tsv(self, file_path: str) -> pd.DataFrame:
        """Load TSV file"""
        try:
            df = pd.read_csv(file_path, sep='\t')
            logger.info(f"Loaded TSV file: {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading TSV file: {str(e)}")
            raise DataProcessingError(f"Failed to load TSV file: {str(e)}")
    
    async def _load_zip(self, file_path: str) -> pd.DataFrame:
        """Load ZIP file containing dataset"""
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                file_names = zip_ref.namelist()
                
                # Find the first CSV or Excel file
                dataset_file = None
                for file_name in file_names:
                    if file_name.endswith(('.csv', '.xlsx', '.xls', '.json', '.tsv')):
                        dataset_file = file_name
                        break
                
                if not dataset_file:
                    raise DataProcessingError("No valid dataset file found in ZIP")
                
                # Extract and load the dataset
                with zip_ref.open(dataset_file) as f:
                    content = f.read()
                    
                    # Determine file type and load accordingly
                    if dataset_file.endswith('.csv'):
                        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                    elif dataset_file.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(io.BytesIO(content))
                    elif dataset_file.endswith('.json'):
                        data = json.loads(content.decode('utf-8'))
                        df = pd.DataFrame(data)
                    elif dataset_file.endswith('.tsv'):
                        df = pd.read_csv(io.StringIO(content.decode('utf-8')), sep='\t')
                
                logger.info(f"Loaded dataset from ZIP: {dataset_file}")
                return df
                
        except Exception as e:
            logger.error(f"Error loading ZIP file: {str(e)}")
            raise DataProcessingError(f"Failed to load ZIP file: {str(e)}")
    
    async def perform_data_inspection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data inspection
        
        Args:
            df: Input dataframe
            
        Returns:
            Dict containing inspection results
        """
        try:
            inspection = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'head': df.head().to_dict('records'),
                'tail': df.tail().to_dict('records'),
                'sample': df.sample(n=min(5, len(df))).to_dict('records'),
                'info': {
                    'memory_usage': df.memory_usage(deep=True).sum(),
                    'non_null_count': df.count().to_dict(),
                    'null_count': df.isnull().sum().to_dict(),
                    'null_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
                },
                'describe_numeric': df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else {},
                'describe_categorical': df.describe(include='object').to_dict() if not df.select_dtypes(include='object').empty else {},
                'describe_all': df.describe(include='all').to_dict(),
                'unique_values': {},
                'value_counts': {},
                'duplicates': df.duplicated().sum(),
                'data_quality': await self._assess_data_quality(df)
            }
            
            # Get unique values and value counts for categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if df[col].nunique() <= 50:  # Only for columns with reasonable unique values
                    inspection['unique_values'][col] = df[col].unique().tolist()
                    inspection['value_counts'][col] = df[col].value_counts().to_dict()
                else:
                    inspection['unique_values'][col] = f"Too many unique values ({df[col].nunique()})"
                    inspection['value_counts'][col] = df[col].value_counts().head(10).to_dict()
            
            return inspection
            
        except Exception as e:
            logger.error(f"Error in data inspection: {str(e)}")
            raise DataProcessingError(f"Failed to perform data inspection: {str(e)}")
    
    async def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality and provide recommendations"""
        try:
            quality_score = 100.0
            issues = []
            recommendations = []
            
            # Missing values assessment
            missing_pct = df.isnull().sum() / len(df) * 100
            high_missing_cols = missing_pct[missing_pct > 50].index.tolist()
            if high_missing_cols:
                quality_score -= 20
                issues.append(f"High missing values in columns: {high_missing_cols}")
                recommendations.append("Consider dropping columns with >50% missing values")
            
            # Duplicates assessment
            duplicate_pct = df.duplicated().sum() / len(df) * 100
            if duplicate_pct > 5:
                quality_score -= 15
                issues.append(f"High duplicate rows: {duplicate_pct:.1f}%")
                recommendations.append("Remove duplicate rows")
            
            # Data type consistency
            mixed_type_cols = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        pd.to_numeric(df[col], errors='raise')
                        mixed_type_cols.append(col)
                    except:
                        pass
            
            if mixed_type_cols:
                quality_score -= 10
                issues.append(f"Mixed data types in columns: {mixed_type_cols}")
                recommendations.append("Fix data type inconsistencies")
            
            # Outlier detection for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outlier_cols = []
            for col in numeric_cols:
                if len(df[col].dropna()) > 0:
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    outlier_pct = (z_scores > 3).sum() / len(df) * 100
                    if outlier_pct > 10:
                        outlier_cols.append(col)
            
            if outlier_cols:
                quality_score -= 10
                issues.append(f"High outliers in columns: {outlier_cols}")
                recommendations.append("Consider outlier treatment")
            
            # Data imbalance for potential target columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            imbalanced_cols = []
            for col in categorical_cols:
                if df[col].nunique() <= 10:  # Potential target column
                    value_counts = df[col].value_counts()
                    min_class_pct = value_counts.min() / len(df) * 100
                    if min_class_pct < 5:
                        imbalanced_cols.append(col)
            
            if imbalanced_cols:
                quality_score -= 5
                issues.append(f"Imbalanced classes in columns: {imbalanced_cols}")
                recommendations.append("Consider class balancing techniques")
            
            return {
                'quality_score': max(0, quality_score),
                'issues': issues,
                'recommendations': recommendations,
                'missing_value_summary': missing_pct.to_dict(),
                'duplicate_percentage': duplicate_pct,
                'outlier_summary': {col: self._detect_outliers(df[col]) for col in numeric_cols}
            }
            
        except Exception as e:
            logger.error(f"Error in data quality assessment: {str(e)}")
            return {
                'quality_score': 0,
                'issues': [f"Error in quality assessment: {str(e)}"],
                'recommendations': ["Manual data review required"]
            }
    
    def _detect_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers in a series"""
        try:
            if series.dtype not in [np.number]:
                return {'count': 0, 'percentage': 0.0}
            
            clean_series = series.dropna()
            if len(clean_series) == 0:
                return {'count': 0, 'percentage': 0.0}
            
            # IQR method
            Q1 = clean_series.quantile(0.25)
            Q3 = clean_series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
            
            return {
                'count': len(outliers),
                'percentage': len(outliers) / len(clean_series) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            return {'count': 0, 'percentage': 0.0}
    
    async def clean_data(self, df: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Perform comprehensive data cleaning
        
        Args:
            df: Input dataframe
            config: Cleaning configuration
            
        Returns:
            Cleaned dataframe
        """
        try:
            cleaned_df = df.copy()
            config = config or {}
            
            # Remove duplicates
            if config.get('remove_duplicates', True):
                initial_rows = len(cleaned_df)
                cleaned_df = cleaned_df.drop_duplicates()
                removed_rows = initial_rows - len(cleaned_df)
                if removed_rows > 0:
                    logger.info(f"Removed {removed_rows} duplicate rows")
            
            # Handle missing values
            if config.get('handle_missing_values', True):
                cleaned_df = await self._handle_missing_values(cleaned_df, config)
            
            # Fix data types
            if config.get('fix_data_types', True):
                cleaned_df = await self._fix_data_types(cleaned_df)
            
            # Handle outliers
            if config.get('handle_outliers', True):
                cleaned_df = await self._handle_outliers(cleaned_df, config)
            
            # Drop irrelevant columns
            if config.get('drop_irrelevant_columns', True):
                cleaned_df = await self._drop_irrelevant_columns(cleaned_df, config)
            
            logger.info(f"Data cleaning completed. Shape: {cleaned_df.shape}")
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Error in data cleaning: {str(e)}")
            raise DataProcessingError(f"Failed to clean data: {str(e)}")
    
    async def _handle_missing_values(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Handle missing values in the dataframe"""
        try:
            # Drop columns with too many missing values
            missing_threshold = config.get('missing_threshold', self.settings.MISSING_VALUE_THRESHOLD)
            missing_pct = df.isnull().sum() / len(df)
            cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
            
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                logger.info(f"Dropped columns with >{missing_threshold*100}% missing values: {cols_to_drop}")
            
            # Impute remaining missing values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            # Numeric imputation
            if len(numeric_cols) > 0:
                imputer = MeanMedianImputer(imputation_method='median', variables=numeric_cols.tolist())
                df = imputer.fit_transform(df)
            
            # Categorical imputation
            if len(categorical_cols) > 0:
                imputer = CategoricalImputer(imputation_method='frequent', variables=categorical_cols.tolist())
                df = imputer.fit_transform(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise DataProcessingError(f"Failed to handle missing values: {str(e)}")
    
    async def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix data type inconsistencies"""
        try:
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Try to convert to numeric
                    try:
                        converted = pd.to_numeric(df[col], errors='coerce')
                        if converted.notna().sum() > len(df) * 0.8:  # If 80% can be converted
                            df[col] = converted
                            logger.info(f"Converted column {col} to numeric")
                    except:
                        pass
                    
                    # Try to convert to datetime
                    try:
                        converted = pd.to_datetime(df[col], errors='coerce')
                        if converted.notna().sum() > len(df) * 0.8:  # If 80% can be converted
                            df[col] = converted
                            logger.info(f"Converted column {col} to datetime")
                    except:
                        pass
            
            return df
            
        except Exception as e:
            logger.error(f"Error fixing data types: {str(e)}")
            return df
    
    async def _handle_outliers(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Handle outliers in numeric columns"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return df
            
            method = config.get('outlier_method', self.settings.OUTLIER_DETECTION_METHOD)
            
            if method == 'IQR':
                # Use IQR method for outlier detection and capping
                outlier_trimmer = OutlierTrimmer(
                    capping_method='iqr',
                    tail='both',
                    fold=1.5,
                    variables=numeric_cols.tolist()
                )
                df = outlier_trimmer.fit_transform(df)
                logger.info(f"Applied IQR outlier treatment to {len(numeric_cols)} columns")
            
            elif method == 'Z-Score':
                # Use Z-score method
                threshold = config.get('outlier_threshold', self.settings.OUTLIER_THRESHOLD)
                for col in numeric_cols:
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    outliers = z_scores > threshold
                    if outliers.sum() > 0:
                        # Cap outliers at threshold
                        mean_val = df[col].mean()
                        std_val = df[col].std()
                        lower_bound = mean_val - threshold * std_val
                        upper_bound = mean_val + threshold * std_val
                        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                        logger.info(f"Applied Z-score outlier treatment to column {col}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error handling outliers: {str(e)}")
            return df
    
    async def _drop_irrelevant_columns(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Drop irrelevant columns"""
        try:
            cols_to_drop = []
            
            # Drop columns with single unique value
            for col in df.columns:
                if df[col].nunique() <= 1:
                    cols_to_drop.append(col)
            
            # Drop columns with too many unique values (potential IDs)
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].nunique() > len(df) * 0.8:
                    cols_to_drop.append(col)
            
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                logger.info(f"Dropped irrelevant columns: {cols_to_drop}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error dropping irrelevant columns: {str(e)}")
            return df
    
    async def apply_transformations(self, df: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Apply various data transformations
        
        Args:
            df: Input dataframe
            config: Transformation configuration
            
        Returns:
            Transformed dataframe
        """
        try:
            transformed_df = df.copy()
            config = config or {}
            
            # Encoding
            if config.get('apply_encoding', True):
                transformed_df = await self._apply_encoding(transformed_df, config)
            
            # Scaling
            if config.get('apply_scaling', True):
                transformed_df = await self._apply_scaling(transformed_df, config)
            
            # Feature engineering
            if config.get('apply_feature_engineering', True):
                transformed_df = await self._apply_feature_engineering(transformed_df, config)
            
            # Dimensionality reduction
            if config.get('apply_dimensionality_reduction', False):
                transformed_df = await self._apply_dimensionality_reduction(transformed_df, config)
            
            logger.info(f"Applied transformations. Final shape: {transformed_df.shape}")
            return transformed_df
            
        except Exception as e:
            logger.error(f"Error applying transformations: {str(e)}")
            raise DataProcessingError(f"Failed to apply transformations: {str(e)}")
    
    async def _apply_encoding(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply encoding to categorical variables"""
        try:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_cols) == 0:
                return df
            
            encoding_method = config.get('encoding_method', 'auto')
            
            for col in categorical_cols:
                unique_values = df[col].nunique()
                
                if encoding_method == 'auto':
                    # Auto-select encoding method based on cardinality
                    if unique_values <= 2:
                        # Binary encoding
                        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
                    elif unique_values <= 10:
                        # One-hot encoding
                        encoder = FEOneHotEncoder(variables=[col], drop_last=True)
                        df = encoder.fit_transform(df)
                    else:
                        # Target encoding (placeholder - requires target variable)
                        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
                
                elif encoding_method == 'onehot':
                    encoder = FEOneHotEncoder(variables=[col], drop_last=True)
                    df = encoder.fit_transform(df)
                
                elif encoding_method == 'label':
                    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
                
                elif encoding_method == 'ordinal':
                    encoder = FEOrdinalEncoder(variables=[col])
                    df = encoder.fit_transform(df)
            
            logger.info(f"Applied encoding to {len(categorical_cols)} categorical columns")
            return df
            
        except Exception as e:
            logger.error(f"Error applying encoding: {str(e)}")
            return df
    
    async def _apply_scaling(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply scaling to numeric variables"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return df
            
            scaler_name = config.get('scaler', self.settings.DEFAULT_SCALER)
            scaler = self.scalers.get(scaler_name, StandardScaler())
            
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            logger.info(f"Applied {scaler_name} scaling to {len(numeric_cols)} columns")
            
            return df
            
        except Exception as e:
            logger.error(f"Error applying scaling: {str(e)}")
            return df
    
    async def _apply_feature_engineering(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply feature engineering techniques"""
        try:
            if not config.get('feature_engineering', self.settings.AUTO_FEATURE_ENGINEERING):
                return df
            
            # Create interaction features
            if config.get('interaction_features', self.settings.CREATE_INTERACTION_FEATURES):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    # Create a few interaction features (to avoid explosion)
                    for i, col1 in enumerate(numeric_cols[:3]):
                        for col2 in numeric_cols[i+1:4]:
                            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                    logger.info("Created interaction features")
            
            # Create polynomial features
            if config.get('polynomial_features', self.settings.CREATE_POLYNOMIAL_FEATURES):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                degree = config.get('polynomial_degree', self.settings.POLYNOMIAL_DEGREE)
                
                if len(numeric_cols) > 0 and len(numeric_cols) <= 5:  # Limit to avoid explosion
                    for col in numeric_cols[:3]:  # Only first 3 columns
                        df[f'{col}_squared'] = df[col] ** 2
                        if degree >= 3:
                            df[f'{col}_cubed'] = df[col] ** 3
                    logger.info("Created polynomial features")
            
            # Handle datetime features
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            for col in datetime_cols:
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_weekday'] = df[col].dt.weekday
                df[f'{col}_quarter'] = df[col].dt.quarter
                logger.info(f"Extracted datetime features from {col}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            return df
    
    async def _apply_dimensionality_reduction(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply dimensionality reduction techniques"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) <= 10:  # Only apply if we have many features
                return df
            
            # Remove low variance features
            threshold = config.get('variance_threshold', self.settings.LOW_VARIANCE_THRESHOLD)
            selector = VarianceThreshold(threshold=threshold)
            df_numeric = df[numeric_cols]
            selected_features = selector.fit_transform(df_numeric)
            selected_cols = df_numeric.columns[selector.get_support()]
            
            # Replace numeric columns with selected features
            df_result = df.drop(columns=numeric_cols)
            df_result[selected_cols] = selected_features
            
            logger.info(f"Applied variance threshold. Reduced from {len(numeric_cols)} to {len(selected_cols)} features")
            
            # Apply PCA if still too many features
            if len(selected_cols) > 20:
                n_components = min(20, len(selected_cols) - 1)
                pca = PCA(n_components=n_components)
                pca_features = pca.fit_transform(df_result[selected_cols])
                
                # Replace with PCA components
                df_result = df_result.drop(columns=selected_cols)
                for i in range(n_components):
                    df_result[f'PC_{i+1}'] = pca_features[:, i]
                
                logger.info(f"Applied PCA. Reduced to {n_components} components")
            
            return df_result
            
        except Exception as e:
            logger.error(f"Error in dimensionality reduction: {str(e)}")
            return df
    
    async def handle_imbalanced_data(self, X: pd.DataFrame, y: pd.Series, config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle imbalanced data using various techniques
        
        Args:
            X: Feature matrix
            y: Target variable
            config: Configuration for imbalanced data handling
            
        Returns:
            Tuple of resampled X and y
        """
        try:
            config = config or {}
            
            # Check if data is imbalanced
            value_counts = y.value_counts()
            min_class_ratio = value_counts.min() / value_counts.sum()
            
            if min_class_ratio >= config.get('imbalance_threshold', self.settings.IMBALANCE_THRESHOLD):
                logger.info("Data is already balanced")
                return X, y
            
            method = config.get('sampling_method', 'SMOTE')
            
            if method == 'SMOTE':
                sampler = SMOTE(
                    sampling_strategy=config.get('sampling_strategy', self.settings.DEFAULT_SAMPLING_STRATEGY),
                    k_neighbors=config.get('k_neighbors', self.settings.SMOTE_K_NEIGHBORS),
                    random_state=self.settings.DEFAULT_RANDOM_STATE
                )
            elif method == 'RandomOverSampler':
                sampler = RandomOverSampler(
                    sampling_strategy=config.get('sampling_strategy', self.settings.DEFAULT_SAMPLING_STRATEGY),
                    random_state=self.settings.DEFAULT_RANDOM_STATE
                )
            elif method == 'RandomUnderSampler':
                sampler = RandomUnderSampler(
                    sampling_strategy=config.get('sampling_strategy', self.settings.DEFAULT_SAMPLING_STRATEGY),
                    random_state=self.settings.DEFAULT_RANDOM_STATE
                )
            elif method == 'SMOTEENN':
                sampler = SMOTEENN(
                    sampling_strategy=config.get('sampling_strategy', self.settings.DEFAULT_SAMPLING_STRATEGY),
                    random_state=self.settings.DEFAULT_RANDOM_STATE
                )
            elif method == 'SMOTETomek':
                sampler = SMOTETomek(
                    sampling_strategy=config.get('sampling_strategy', self.settings.DEFAULT_SAMPLING_STRATEGY),
                    random_state=self.settings.DEFAULT_RANDOM_STATE
                )
            else:
                logger.warning(f"Unknown sampling method: {method}")
                return X, y
            
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            logger.info(f"Applied {method} sampling. Shape changed from {X.shape} to {X_resampled.shape}")
            
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=y.name)
            
        except Exception as e:
            logger.error(f"Error handling imbalanced data: {str(e)}")
            return X, y
    
    async def split_dataset(self, df: pd.DataFrame, target_column: str, config: Dict[str, Any] = None) -> Dict[str, pd.DataFrame]:
        """
        Split dataset into train, validation, and test sets
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            config: Split configuration
            
        Returns:
            Dictionary containing train, validation, and test sets
        """
        try:
            config = config or {}
            
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Get split ratios
            train_size = config.get('train_size', self.settings.TRAIN_SIZE)
            val_size = config.get('val_size', self.settings.VAL_SIZE)
            test_size = config.get('test_size', self.settings.TEST_SIZE)
            
            # Normalize ratios
            total = train_size + val_size + test_size
            train_size = train_size / total
            val_size = val_size / total
            test_size = test_size / total
            
            # First split: train + val vs test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=self.settings.DEFAULT_RANDOM_STATE,
                stratify=y if y.dtype == 'object' or y.nunique() <= 20 else None
            )
            
            # Second split: train vs val
            val_ratio = val_size / (train_size + val_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_ratio,
                random_state=self.settings.DEFAULT_RANDOM_STATE,
                stratify=y_temp if y_temp.dtype == 'object' or y_temp.nunique() <= 20 else None
            )
            
            # Combine back into dataframes
            train_df = pd.concat([X_train, y_train], axis=1)
            val_df = pd.concat([X_val, y_val], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)
            
            logger.info(f"Dataset split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            
            return {
                'train': train_df,
                'validation': val_df,
                'test': test_df,
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            }
            
        except Exception as e:
            logger.error(f"Error splitting dataset: {str(e)}")
            raise DataProcessingError(f"Failed to split dataset: {str(e)}")
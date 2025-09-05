"""
RTGS AI Analyst - Data Helpers
Utility functions for data manipulation, encoding detection, and file processing
"""

import pandas as pd
import numpy as np
import chardet
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import json
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


def detect_encoding(file_path: str, sample_size: int = 10000) -> str:
    """Detect file encoding using chardet with fallback options"""
    try:
        with open(file_path, 'rb') as file:
            sample = file.read(sample_size)
            result = chardet.detect(sample)
            
            if result['confidence'] > 0.7:
                return result['encoding']
            
            # Fallback encodings to try
            fallback_encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in fallback_encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as test_file:
                        test_file.read(1000)
                    return encoding
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            # Last resort
            return 'utf-8'
            
    except Exception:
        return 'utf-8'


def detect_separator(file_path: str, encoding: str = 'utf-8', sample_lines: int = 10) -> str:
    """Detect CSV separator by analyzing first few lines"""
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            sample_text = ""
            for _ in range(sample_lines):
                line = file.readline()
                if not line:
                    break
                sample_text += line
        
        # Count potential separators
        separators = [',', ';', '\t', '|', ':']
        separator_counts = {}
        
        for sep in separators:
            # Count occurrences per line
            lines = sample_text.split('\n')
            counts_per_line = [line.count(sep) for line in lines if line.strip()]
            
            if counts_per_line:
                # Check consistency (same count per line suggests it's the separator)
                most_common_count = Counter(counts_per_line).most_common(1)[0][1]
                consistency = most_common_count / len(counts_per_line)
                avg_count = sum(counts_per_line) / len(counts_per_line)
                
                separator_counts[sep] = {
                    'avg_count': avg_count,
                    'consistency': consistency,
                    'score': avg_count * consistency
                }
        
        # Select separator with highest score
        if separator_counts:
            best_sep = max(separator_counts.keys(), key=lambda x: separator_counts[x]['score'])
            if separator_counts[best_sep]['score'] > 1:  # Minimum threshold
                return best_sep
        
        # Default fallback
        return ','
        
    except Exception:
        return ','


def estimate_row_count(file_path: str, sample_size: int = 1024*1024) -> int:
    """Estimate total row count for large files"""
    try:
        path = Path(file_path)
        file_size = path.stat().st_size
        
        if file_size < sample_size:
            # Small file, count exactly
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return sum(1 for _ in f)
        
        # Large file, estimate from sample
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            sample = f.read(sample_size)
            lines_in_sample = sample.count('\n')
            
            if lines_in_sample == 0:
                return 0
            
            # Estimate total lines
            estimated_lines = int((file_size / sample_size) * lines_in_sample)
            return max(1, estimated_lines)
            
    except Exception:
        return 0


def standardize_column_names(columns: List[str]) -> Dict[str, str]:
    """Standardize column names to snake_case format"""
    mapping = {}
    
    for col in columns:
        # Convert to snake_case
        standardized = col.strip()
        
        # Replace common patterns
        standardized = re.sub(r'[^\w\s]', '', standardized)  # Remove special chars except underscore
        standardized = re.sub(r'\s+', '_', standardized)  # Spaces to underscores
        standardized = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', standardized)  # CamelCase to snake_case
        standardized = standardized.lower()
        
        # Remove multiple underscores
        standardized = re.sub(r'_+', '_', standardized)
        standardized = standardized.strip('_')
        
        # Ensure it's a valid Python identifier
        if not standardized or standardized[0].isdigit():
            standardized = f"col_{standardized}"
        
        # Handle duplicates
        original_standardized = standardized
        counter = 1
        while standardized in mapping.values():
            standardized = f"{original_standardized}_{counter}"
            counter += 1
        
        mapping[col] = standardized
    
    return mapping


def detect_column_types(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Detect and suggest column types with confidence scores"""
    type_suggestions = {}
    
    for col in df.columns:
        series = df[col].dropna()
        
        if len(series) == 0:
            type_suggestions[col] = {
                'suggested_type': 'object',
                'confidence': 0.0,
                'rationale': 'Column is entirely null'
            }
            continue
        
        # Calculate basic statistics
        null_frac = df[col].isnull().sum() / len(df)
        unique_frac = len(series.unique()) / len(series) if len(series) > 0 else 0
        
        # Type detection logic
        type_info = _detect_single_column_type(series)
        type_info.update({
            'null_fraction': null_frac,
            'unique_fraction': unique_frac,
            'sample_values': series.head(10).tolist()
        })
        
        type_suggestions[col] = type_info
    
    return type_suggestions


def _detect_single_column_type(series: pd.Series) -> Dict[str, Any]:
    """Detect type for a single column"""
    
    # Convert to string for pattern analysis
    str_series = series.astype(str)
    
    # Check for numeric types
    numeric_score = _check_numeric_type(series)
    if numeric_score['confidence'] > 0.8:
        return numeric_score
    
    # Check for datetime
    datetime_score = _check_datetime_type(str_series)
    if datetime_score['confidence'] > 0.7:
        return datetime_score
    
    # Check for boolean
    boolean_score = _check_boolean_type(str_series)
    if boolean_score['confidence'] > 0.8:
        return boolean_score
    
    # Check for categorical
    categorical_score = _check_categorical_type(series)
    if categorical_score['confidence'] > 0.6:
        return categorical_score
    
    # Check for geographic coordinates
    geo_score = _check_geographic_type(str_series)
    if geo_score['confidence'] > 0.7:
        return geo_score
    
    # Check for ID/identifier
    id_score = _check_id_type(series, str_series)
    if id_score['confidence'] > 0.8:
        return id_score
    
    # Default to text
    return {
        'suggested_type': 'object',
        'confidence': 0.5,
        'rationale': 'Mixed or unrecognized patterns, defaulting to text'
    }


def _check_numeric_type(series: pd.Series) -> Dict[str, Any]:
    """Check if column is numeric"""
    try:
        # Try converting to numeric
        numeric_series = pd.to_numeric(series, errors='coerce')
        success_rate = (~numeric_series.isnull()).sum() / len(series)
        
        if success_rate > 0.9:
            # Determine if integer or float
            if (numeric_series == numeric_series.astype(int)).all():
                return {
                    'suggested_type': 'int64',
                    'confidence': success_rate,
                    'rationale': f'{success_rate:.1%} of values are integers'
                }
            else:
                return {
                    'suggested_type': 'float64',
                    'confidence': success_rate,
                    'rationale': f'{success_rate:.1%} of values are numeric (float)'
                }
        
        return {
            'suggested_type': 'object',
            'confidence': success_rate * 0.5,
            'rationale': f'Only {success_rate:.1%} of values are numeric'
        }
        
    except Exception:
        return {'suggested_type': 'object', 'confidence': 0.0, 'rationale': 'Numeric conversion failed'}


def _check_datetime_type(str_series: pd.Series) -> Dict[str, Any]:
    """Check if column contains datetime values"""
    
    # Common date patterns
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY or DD/MM/YYYY
        r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY or DD-MM-YYYY
        r'\d{1,2}/\d{1,2}/\d{2,4}',  # M/D/YY or M/D/YYYY
    ]
    
    datetime_patterns = [
        r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # YYYY-MM-DD HH:MM:SS
        r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}',  # MM/DD/YYYY HH:MM
    ]
    
    all_patterns = date_patterns + datetime_patterns
    
    # Check pattern matches
    total_matches = 0
    for pattern in all_patterns:
        matches = str_series.str.match(pattern).sum()
        total_matches = max(total_matches, matches)
    
    match_rate = total_matches / len(str_series)
    
    # Try pandas datetime parsing
    try:
        pd.to_datetime(str_series.head(min(100, len(str_series))), errors='raise')
        pandas_success = True
    except:
        pandas_success = False
    
    confidence = match_rate
    if pandas_success:
        confidence = min(0.9, confidence + 0.3)
    
    if confidence > 0.7:
        return {
            'suggested_type': 'datetime64[ns]',
            'confidence': confidence,
            'rationale': f'{match_rate:.1%} match datetime patterns'
        }
    
    return {'suggested_type': 'object', 'confidence': confidence * 0.5, 'rationale': 'Low datetime pattern match'}


def _check_boolean_type(str_series: pd.Series) -> Dict[str, Any]:
    """Check if column contains boolean values"""
    
    # Convert to lowercase for checking
    lower_series = str_series.str.lower().str.strip()
    
    boolean_values = {
        'true', 'false', 'yes', 'no', 'y', 'n', 
        '1', '0', 'on', 'off', 'enabled', 'disabled'
    }
    
    # Check how many values are boolean-like
    boolean_matches = lower_series.isin(boolean_values).sum()
    match_rate = boolean_matches / len(str_series)
    
    if match_rate > 0.8:
        return {
            'suggested_type': 'bool',
            'confidence': match_rate,
            'rationale': f'{match_rate:.1%} are boolean-like values'
        }
    
    return {'suggested_type': 'object', 'confidence': match_rate * 0.3, 'rationale': 'Low boolean pattern match'}


def _check_categorical_type(series: pd.Series) -> Dict[str, Any]:
    """Check if column should be categorical"""
    
    unique_count = len(series.unique())
    total_count = len(series)
    
    if total_count == 0:
        return {'suggested_type': 'object', 'confidence': 0.0, 'rationale': 'Empty series'}
    
    unique_ratio = unique_count / total_count
    
    # Categorical if low unique ratio and reasonable number of categories
    if unique_ratio < 0.1 and unique_count < 100:
        confidence = (0.1 - unique_ratio) / 0.1  # Higher confidence for lower ratios
        return {
            'suggested_type': 'category',
            'confidence': confidence,
            'rationale': f'{unique_count} unique values ({unique_ratio:.1%} of total)'
        }
    
    return {'suggested_type': 'object', 'confidence': 0.0, 'rationale': f'Too many unique values ({unique_count})'}


def _check_geographic_type(str_series: pd.Series) -> Dict[str, Any]:
    """Check if column contains geographic coordinates"""
    
    # Latitude/longitude patterns
    lat_pattern = r'^-?([0-8]?[0-9]|90)\.?\d*$'
    lon_pattern = r'^-?(1[0-7]\d|[0-9]?\d|180)\.?\d*$'
    
    # Check if values look like coordinates
    numeric_series = pd.to_numeric(str_series, errors='coerce')
    numeric_ratio = (~numeric_series.isnull()).sum() / len(str_series)
    
    if numeric_ratio > 0.8:
        # Check if values are in lat/lon ranges
        lat_like = ((numeric_series >= -90) & (numeric_series <= 90)).sum()
        lon_like = ((numeric_series >= -180) & (numeric_series <= 180)).sum()
        
        lat_ratio = lat_like / len(str_series)
        lon_ratio = lon_like / len(str_series)
        
        if lat_ratio > 0.9:
            return {
                'suggested_type': 'float64',
                'confidence': lat_ratio,
                'rationale': 'Values appear to be latitude coordinates'
            }
        elif lon_ratio > 0.9:
            return {
                'suggested_type': 'float64', 
                'confidence': lon_ratio,
                'rationale': 'Values appear to be longitude coordinates'
            }
    
    return {'suggested_type': 'object', 'confidence': 0.0, 'rationale': 'Not geographic coordinates'}


def _check_id_type(series: pd.Series, str_series: pd.Series) -> Dict[str, Any]:
    """Check if column is an identifier/ID column"""
    
    unique_ratio = len(series.unique()) / len(series)
    
    # High uniqueness suggests ID
    if unique_ratio > 0.95:
        
        # Check for ID-like patterns
        id_patterns = [
            r'^[A-Z]{2,4}\d+$',  # Like ABC123, ABCD1234
            r'^\d+$',  # Pure numeric IDs
            r'^[a-f0-9-]{36}$',  # UUIDs
            r'^[A-Z0-9]{6,}$',  # Mixed alphanumeric codes
        ]
        
        pattern_matches = 0
        for pattern in id_patterns:
            matches = str_series.str.match(pattern).sum()
            pattern_matches = max(pattern_matches, matches)
        
        pattern_ratio = pattern_matches / len(str_series)
        
        confidence = unique_ratio * 0.7 + pattern_ratio * 0.3
        
        if confidence > 0.8:
            return {
                'suggested_type': 'object',
                'confidence': confidence,
                'rationale': f'High uniqueness ({unique_ratio:.1%}) with ID-like patterns'
            }
    
    return {'suggested_type': 'object', 'confidence': 0.0, 'rationale': 'Not ID-like'}


def clean_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Clean and standardize column names"""
    
    original_columns = df.columns.tolist()
    mapping = standardize_column_names(original_columns)
    
    # Apply mapping
    df_cleaned = df.rename(columns=mapping)
    
    return df_cleaned, mapping


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> Tuple[np.ndarray, Dict[str, float]]:
    """Detect outliers using IQR method"""
    
    if not pd.api.types.is_numeric_dtype(series):
        return np.array([]), {}
    
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = (series < lower_bound) | (series > upper_bound)
    
    stats = {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outlier_count': outliers.sum(),
        'outlier_percentage': (outliers.sum() / len(series)) * 100
    }
    
    return outliers, stats


def suggest_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """Suggest optimal data types for DataFrame columns"""
    
    suggestions = {}
    
    for col in df.columns:
        current_dtype = df[col].dtype
        
        # Skip if already optimal
        if current_dtype in ['int64', 'float64', 'bool', 'datetime64[ns]', 'category']:
            suggestions[col] = str(current_dtype)
            continue
        
        # Analyze column and suggest type
        type_info = detect_column_types(df[[col]])
        suggestions[col] = type_info[col]['suggested_type']
    
    return suggestions


def safe_type_conversion(df: pd.DataFrame, type_mapping: Dict[str, str]) -> Tuple[pd.DataFrame, List[str]]:
    """Safely convert column types with error handling"""
    
    df_converted = df.copy()
    conversion_errors = []
    
    for col, target_type in type_mapping.items():
        if col not in df_converted.columns:
            continue
        
        try:
            if target_type == 'datetime64[ns]':
                df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
            elif target_type in ['int64', 'float64']:
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                if target_type == 'int64':
                    # Only convert to int if no NaN values after conversion
                    if df_converted[col].isnull().sum() == 0:
                        df_converted[col] = df_converted[col].astype('int64')
            elif target_type == 'bool':
                # Custom boolean conversion
                df_converted[col] = df_converted[col].map({
                    'true': True, 'false': False, 'yes': True, 'no': False,
                    'y': True, 'n': False, '1': True, '0': False,
                    'on': True, 'off': False, 'enabled': True, 'disabled': False,
                    True: True, False: False, 1: True, 0: False
                })
            elif target_type == 'category':
                df_converted[col] = df_converted[col].astype('category')
                
        except Exception as e:
            conversion_errors.append(f"Failed to convert {col} to {target_type}: {str(e)}")
    
    return df_converted, conversion_errors
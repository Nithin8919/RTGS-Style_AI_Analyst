"""
RTGS AI Analyst - Data Helper Utilities
Complete implementation of data processing utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import chardet
import re
from pathlib import Path
import logging
from datetime import datetime
import zipfile
import gzip

logger = logging.getLogger(__name__)

def detect_encoding(file_path: str) -> str:
    """Detect file encoding using chardet"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(50000)  # Read first 50KB
            result = chardet.detect(raw_data)
            confidence = result.get('confidence', 0.0)
            encoding = result.get('encoding', 'utf-8')
            
            logger.debug(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
            
            # Fallback for low confidence
            if confidence < 0.7:
                logger.warning(f"Low confidence encoding detection, falling back to utf-8")
                return 'utf-8'
                
            return encoding or 'utf-8'
    except Exception as e:
        logger.error(f"Encoding detection failed: {e}")
        return 'utf-8'

def detect_separator(file_path: str, encoding: str = 'utf-8') -> str:
    """Detect CSV separator by analyzing first few lines"""
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            # Read first 5 lines for analysis
            lines = [f.readline().strip() for _ in range(5)]
            lines = [line for line in lines if line]  # Remove empty lines
            
        if not lines:
            return ','
            
        # Common separators to test
        separators = [',', ';', '\t', '|', ':']
        sep_scores = {}
        
        for sep in separators:
            # Count occurrences across all lines
            counts = [line.count(sep) for line in lines]
            
            # Good separator should have:
            # 1. Consistent count across lines
            # 2. At least 1 occurrence per line
            if len(set(counts)) == 1 and counts[0] > 0:
                sep_scores[sep] = counts[0]
            elif all(c > 0 for c in counts):
                # Accept if all lines have at least one occurrence
                sep_scores[sep] = min(counts)
                
        if sep_scores:
            best_sep = max(sep_scores.keys(), key=sep_scores.get)
            logger.debug(f"Detected separator: '{best_sep}' with {sep_scores[best_sep]} fields")
            return best_sep
        
        # Fallback: use comma
        logger.warning("Could not reliably detect separator, using comma")
        return ','
        
    except Exception as e:
        logger.error(f"Separator detection failed: {e}")
        return ','

def estimate_row_count(file_path: str, encoding: str = 'utf-8') -> int:
    """Estimate total rows in file efficiently"""
    try:
        file_size = Path(file_path).stat().st_size
        
        # For small files, count exactly
        if file_size < 1_000_000:  # 1MB
            with open(file_path, 'r', encoding=encoding) as f:
                return sum(1 for _ in f) - 1  # Subtract header
        
        # For large files, estimate based on sample
        sample_size = min(100_000, file_size // 10)  # Sample 10% or 100KB
        
        with open(file_path, 'rb') as f:
            sample = f.read(sample_size)
            lines_in_sample = sample.count(b'\n')
            
        if lines_in_sample == 0:
            return 0
            
        # Estimate based on sample
        estimated = int((file_size / sample_size) * lines_in_sample) - 1
        logger.debug(f"Estimated {estimated:,} rows in file ({file_size:,} bytes)")
        return max(0, estimated)
        
    except Exception as e:
        logger.error(f"Row count estimation failed: {e}")
        return 0

def detect_column_types(df: pd.DataFrame, sample_size: int = 1000) -> Dict[str, Dict]:
    """
    Detect column types with confidence scores
    Returns detailed type information for each column
    """
    results = {}
    
    # Work with sample for large datasets
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df
    
    for col in df.columns:
        series = sample_df[col].dropna()
        
        if len(series) == 0:
            results[col] = {
                'type': 'unknown',
                'confidence': 0.0,
                'null_frac': 1.0,
                'unique_frac': 0.0,
                'sample_values': []
            }
            continue
        
        # Calculate basic stats
        null_frac = df[col].isnull().sum() / len(df)
        unique_frac = len(series.unique()) / len(series) if len(series) > 0 else 0
        sample_values = series.head(10).tolist()
        
        # Type detection with confidence scoring
        type_scores = {}
        
        # 1. Numeric detection
        type_scores['numeric'] = _detect_numeric(series)
        
        # 2. Integer detection  
        type_scores['integer'] = _detect_integer(series)
        
        # 3. Float detection
        type_scores['float'] = _detect_float(series)
        
        # 4. Date detection
        type_scores['datetime'] = _detect_datetime(series)
        
        # 5. Boolean detection
        type_scores['boolean'] = _detect_boolean(series)
        
        # 6. Categorical detection
        type_scores['categorical'] = _detect_categorical(series, unique_frac)
        
        # 7. Geographic detection
        type_scores['geographic'] = _detect_geographic(series)
        
        # 8. ID detection
        type_scores['id'] = _detect_id(series, unique_frac)
        
        # 9. Currency detection
        type_scores['currency'] = _detect_currency(series)
        
        # 10. Default to text
        type_scores['text'] = 0.3  # Base score for text
        
        # Get best type
        best_type = max(type_scores.keys(), key=type_scores.get)
        best_confidence = type_scores[best_type]
        
        # If confidence is low, mark as ambiguous
        if best_confidence < 0.6:
            best_type = 'text'  # Default to text for ambiguous cases
        
        results[col] = {
            'type': best_type,
            'confidence': best_confidence,
            'type_scores': type_scores,
            'null_frac': null_frac,
            'unique_frac': unique_frac,
            'sample_values': sample_values,
            'stats': _calculate_column_stats(df[col], best_type)
        }
    
    return results

def _detect_numeric(series: pd.Series) -> float:
    """Detect if column is numeric"""
    try:
        numeric_series = pd.to_numeric(series, errors='coerce')
        success_rate = numeric_series.notna().sum() / len(series)
        return success_rate
    except:
        return 0.0

def _detect_integer(series: pd.Series) -> float:
    """Detect if column is integer"""
    try:
        # First check if numeric
        numeric_series = pd.to_numeric(series, errors='coerce')
        numeric_rate = numeric_series.notna().sum() / len(series)
        
        if numeric_rate < 0.8:
            return 0.0
        
        # Check if values are integers
        valid_numeric = numeric_series.dropna()
        if len(valid_numeric) == 0:
            return 0.0
            
        integer_rate = (valid_numeric == valid_numeric.astype(int)).sum() / len(valid_numeric)
        return numeric_rate * integer_rate
    except:
        return 0.0

def _detect_float(series: pd.Series) -> float:
    """Detect if column is float"""
    try:
        numeric_series = pd.to_numeric(series, errors='coerce')
        numeric_rate = numeric_series.notna().sum() / len(series)
        
        if numeric_rate < 0.8:
            return 0.0
        
        # Check if values have decimal parts
        valid_numeric = numeric_series.dropna()
        if len(valid_numeric) == 0:
            return 0.0
            
        has_decimal = (valid_numeric != valid_numeric.astype(int)).sum() / len(valid_numeric)
        return numeric_rate * has_decimal
    except:
        return 0.0

def _detect_datetime(series: pd.Series) -> float:
    """Detect if column contains dates"""
    # Common date formats to try
    date_formats = [
        '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S',
        '%d-%m-%Y', '%Y/%m/%d', '%d.%m.%Y', '%Y', '%m/%Y',
        '%d-%b-%Y', '%d %B %Y', '%B %d, %Y'
    ]
    
    max_success_rate = 0.0
    
    for fmt in date_formats:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors='coerce')
            success_rate = parsed.notna().sum() / len(series)
            max_success_rate = max(max_success_rate, success_rate)
        except:
            continue
    
    # Also try general datetime parsing
    try:
        parsed = pd.to_datetime(series, errors='coerce')
        success_rate = parsed.notna().sum() / len(series)
        max_success_rate = max(max_success_rate, success_rate)
    except:
        pass
    
    return max_success_rate

def _detect_boolean(series: pd.Series) -> float:
    """Detect if column contains boolean values"""
    try:
        unique_vals = set(series.astype(str).str.lower().str.strip().unique())
        
        # Remove NaN-like values
        unique_vals.discard('nan')
        unique_vals.discard('none')
        unique_vals.discard('')
        
        # Boolean value sets
        bool_sets = [
            {'true', 'false'},
            {'1', '0'},
            {'yes', 'no'},
            {'y', 'n'},
            {'t', 'f'},
            {'1.0', '0.0'}
        ]
        
        for bool_set in bool_sets:
            if unique_vals.issubset(bool_set) and len(unique_vals) >= 1:
                return 1.0
        
        return 0.0
    except:
        return 0.0

def _detect_categorical(series: pd.Series, unique_frac: float) -> float:
    """Detect if column is categorical"""
    try:
        # Categorical if few unique values and mostly text
        if unique_frac < 0.1 and len(series.unique()) > 1:
            # Check if values are string-like
            is_string = series.astype(str).apply(lambda x: isinstance(x, str)).mean()
            return min(1.0, (1.0 - unique_frac) * is_string)
        return 0.0
    except:
        return 0.0

def _detect_geographic(series: pd.Series) -> float:
    """Detect if column contains geographic data"""
    try:
        # Look for geographic keywords
        geo_keywords = [
            'district', 'state', 'city', 'village', 'mandal', 'tehsil',
            'block', 'ward', 'constituency', 'pincode', 'zip'
        ]
        
        sample_str = ' '.join(series.astype(str).str.lower().head(20))
        
        keyword_score = sum(1 for keyword in geo_keywords if keyword in sample_str)
        
        # Check for coordinate patterns (lat, lon)
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
            if numeric_series.notna().sum() / len(series) > 0.8:
                # Check if values are in lat/lon range
                valid_coords = numeric_series.dropna()
                if len(valid_coords) > 0:
                    lat_like = ((valid_coords >= -90) & (valid_coords <= 90)).mean()
                    lon_like = ((valid_coords >= -180) & (valid_coords <= 180)).mean()
                    if lat_like > 0.8 or lon_like > 0.8:
                        return 0.9
        except:
            pass
        
        return min(0.8, keyword_score * 0.3)
    except:
        return 0.0

def _detect_id(series: pd.Series, unique_frac: float) -> float:
    """Detect if column is an ID field"""
    try:
        # High uniqueness
        if unique_frac < 0.95:
            return 0.0
        
        # Check for ID patterns
        sample_str = series.astype(str).head(10)
        
        # Pattern scoring
        patterns = [
            r'^\d+$',  # Pure numbers
            r'^[A-Z]+\d+$',  # Letters followed by numbers
            r'^\d+[A-Z]+\d*$',  # Numbers with letters
            r'^[A-Z0-9]+$'  # Alphanumeric
        ]
        
        pattern_score = 0
        for pattern in patterns:
            matches = sample_str.str.match(pattern, na=False).sum()
            pattern_score = max(pattern_score, matches / len(sample_str))
        
        return unique_frac * pattern_score
    except:
        return 0.0

def _detect_currency(series: pd.Series) -> float:
    """Detect if column contains currency values"""
    try:
        sample_str = series.astype(str).head(20)
        
        # Look for currency symbols and patterns
        currency_patterns = [
            r'₹',  # Rupee symbol
            r'\$',  # Dollar symbol
            r'rs\.?',  # Rs.
            r'inr',  # INR
            r'crore',  # Crore
            r'lakh',  # Lakh
        ]
        
        pattern_score = 0
        for pattern in currency_patterns:
            matches = sample_str.str.contains(pattern, case=False, na=False).sum()
            pattern_score = max(pattern_score, matches / len(sample_str))
        
        # Also check if numeric after removing currency symbols
        try:
            cleaned = series.astype(str).str.replace(r'[₹$,\s]', '', regex=True)
            numeric_rate = pd.to_numeric(cleaned, errors='coerce').notna().sum() / len(series)
            pattern_score = max(pattern_score, numeric_rate)
        except:
            pass
        
        return pattern_score
    except:
        return 0.0

def _calculate_column_stats(series: pd.Series, col_type: str) -> Dict:
    """Calculate type-specific statistics for column"""
    stats = {}
    
    try:
        if col_type in ['numeric', 'integer', 'float']:
            numeric_series = pd.to_numeric(series, errors='coerce')
            valid_series = numeric_series.dropna()
            
            if len(valid_series) > 0:
                stats.update({
                    'min': float(valid_series.min()),
                    'max': float(valid_series.max()),
                    'mean': float(valid_series.mean()),
                    'median': float(valid_series.median()),
                    'std': float(valid_series.std()),
                    'q25': float(valid_series.quantile(0.25)),
                    'q75': float(valid_series.quantile(0.75))
                })
        
        elif col_type == 'datetime':
            date_series = pd.to_datetime(series, errors='coerce')
            valid_dates = date_series.dropna()
            
            if len(valid_dates) > 0:
                stats.update({
                    'min_date': valid_dates.min().isoformat(),
                    'max_date': valid_dates.max().isoformat(),
                    'date_range_days': (valid_dates.max() - valid_dates.min()).days
                })
        
        elif col_type in ['categorical', 'text']:
            valid_series = series.dropna()
            value_counts = valid_series.value_counts()
            
            stats.update({
                'top_values': value_counts.head(5).to_dict(),
                'unique_count': len(value_counts),
                'most_common': value_counts.index[0] if len(value_counts) > 0 else None
            })
    
    except Exception as e:
        logger.warning(f"Failed to calculate stats for column type {col_type}: {e}")
    
    return stats

def standardize_column_names(columns: List[str]) -> Dict[str, str]:
    """Standardize column names to snake_case with intelligent mapping"""
    mapping = {}
    
    for col in columns:
        original = col
        
        # Clean the column name
        # Remove special characters except underscore
        standardized = re.sub(r'[^\w\s]', '', col)
        
        # Replace spaces with underscores
        standardized = re.sub(r'\s+', '_', standardized)
        
        # Convert to lowercase
        standardized = standardized.lower()
        
        # Remove multiple underscores
        standardized = re.sub(r'_+', '_', standardized)
        
        # Remove leading/trailing underscores
        standardized = standardized.strip('_')
        
        # Handle empty names
        if not standardized:
            standardized = f"column_{hash(original) % 1000}"
        
        # Handle numeric-only names
        if standardized.isdigit():
            standardized = f"col_{standardized}"
        
        mapping[original] = standardized
    
    return mapping

def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Detect outliers using IQR method"""
    try:
        # Convert to numeric
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        if numeric_series.isna().all():
            return pd.Series([False] * len(series), index=series.index)
        
        Q1 = numeric_series.quantile(0.25)
        Q3 = numeric_series.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:  # No variance
            return pd.Series([False] * len(series), index=series.index)
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = (numeric_series < lower_bound) | (numeric_series > upper_bound)
        
        # Handle NaN values
        outliers = outliers.fillna(False)
        
        return outliers
        
    except Exception as e:
        logger.error(f"Outlier detection failed: {e}")
        return pd.Series([False] * len(series), index=series.index)

def safe_type_conversion(series: pd.Series, target_type: str) -> pd.Series:
    """Safely convert series to target type with error handling"""
    try:
        if target_type == 'numeric':
            return pd.to_numeric(series, errors='coerce')
            
        elif target_type == 'integer':
            numeric = pd.to_numeric(series, errors='coerce')
            return numeric.astype('Int64')  # Nullable integer type
            
        elif target_type == 'float':
            return pd.to_numeric(series, errors='coerce').astype(float)
            
        elif target_type == 'datetime':
            return pd.to_datetime(series, errors='coerce')
            
        elif target_type == 'boolean':
            # Handle various boolean representations
            bool_map = {
                'true': True, 'false': False,
                '1': True, '0': False,
                'yes': True, 'no': False,
                'y': True, 'n': False,
                't': True, 'f': False,
                '1.0': True, '0.0': False
            }
            
            return series.astype(str).str.lower().str.strip().map(bool_map)
            
        elif target_type == 'categorical':
            return series.astype('category')
            
        else:  # Default to string
            return series.astype(str)
            
    except Exception as e:
        logger.warning(f"Type conversion to {target_type} failed: {e}")
        return series  # Return original series if conversion fails

def load_dataset_robust(file_path: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Robustly load dataset with automatic format detection and error handling
    Returns tuple of (dataframe, metadata)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Detect file info
    encoding = detect_encoding(str(file_path))
    file_info = {
        'file_path': str(file_path),
        'file_size': file_path.stat().st_size,
        'encoding': encoding,
        'format': file_path.suffix.lower()
    }
    
    try:
        if file_path.suffix.lower() == '.csv':
            separator = detect_separator(str(file_path), encoding)
            file_info['separator'] = separator
            
            # Load with detected parameters
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                sep=separator,
                low_memory=False,
                **kwargs
            )
            
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, **kwargs)
            
        elif file_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(file_path, **kwargs)
            
        elif file_path.suffix.lower() == '.json':
            df = pd.read_json(file_path, **kwargs)
            
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Add metadata
        file_info.update({
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        })
        
        logger.info(f"Successfully loaded dataset: {len(df):,} rows × {len(df.columns)} columns")
        
        return df, file_info
        
    except Exception as e:
        logger.error(f"Failed to load dataset {file_path}: {e}")
        raise

def create_sample_dataset(df: pd.DataFrame, 
                         sample_rows: int = 500, 
                         method: str = 'stratified') -> pd.DataFrame:
    """Create a representative sample of the dataset"""
    try:
        if len(df) <= sample_rows:
            return df.copy()
        
        if method == 'random':
            return df.sample(n=sample_rows, random_state=42)
        
        elif method == 'stratified':
            # Try to stratify by categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_cols) > 0:
                # Use first categorical column for stratification
                strat_col = categorical_cols[0]
                try:
                    return df.groupby(strat_col, group_keys=False).apply(
                        lambda x: x.sample(min(len(x), max(1, sample_rows // len(df[strat_col].unique()))), 
                                         random_state=42)
                    ).head(sample_rows)
                except:
                    # Fallback to random sampling
                    return df.sample(n=sample_rows, random_state=42)
            else:
                return df.sample(n=sample_rows, random_state=42)
        
        elif method == 'systematic':
            step = len(df) // sample_rows
            indices = list(range(0, len(df), step))[:sample_rows]
            return df.iloc[indices]
        
        else:
            raise ValueError(f"Unknown sampling method: {method}")
            
    except Exception as e:
        logger.error(f"Sampling failed: {e}")
        return df.head(sample_rows)  # Fallback to head
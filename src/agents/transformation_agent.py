"""
RTGS AI Analyst - Transformation Agent (Complete Implementation)
Handles feature engineering, time features, per-capita calculations, and spatial joins
"""

import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from src.utils.logging import get_agent_logger, TransformLogger

class TransformationAgent:
    """Agent responsible for feature engineering and data transformation"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = get_agent_logger("transformation")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract feature engineering config
        self.feature_config = self.config.get('feature_engineering', {})
        
        # Transformation controls from config
        self.transformation_controls = self.feature_config.get('transformation_controls', {
            'enable_categorical_encoding': True,
            'enable_numeric_scaling': True,
            'enable_skew_handling': True,
            'enable_datetime_features': True,
            'enable_interaction_features': True,
            'enable_missing_handling': True,
            'auto_detect_requirements': True  # If True, only apply needed transformations
        })
        
        # Initialize transformation tracking
        self.transformations = []
        self.encoders = {}
        self.scalers = {}
    
    def reset_transformation_state(self):
        """Reset transformation tracking for new run"""
        self.transformations = []
        self.encoders = {}
        self.scalers = {}
    
    def analyze_data_requirements(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data to determine which transformations are needed"""
        analysis = {
            'needs_categorical_encoding': False,
            'needs_numeric_scaling': False,
            'needs_skew_handling': False,
            'needs_datetime_features': False,
            'needs_interaction_features': False,
            'needs_missing_handling': False,
            'categorical_cols': [],
            'skewed_cols': [],
            'datetime_cols': [],
            'high_variance_cols': [],
            'missing_value_cols': []
        }
        
        # Get thresholds from config
        cat_config = self.transformation_controls.get('categorical_encoding', {})
        num_config = self.transformation_controls.get('numeric_scaling', {})
        skew_config = self.transformation_controls.get('skew_handling', {})
        
        min_cardinality = cat_config.get('min_cardinality', 2)
        max_cardinality = cat_config.get('max_cardinality', 50)
        variance_threshold = num_config.get('variance_threshold', 1000)
        range_threshold = num_config.get('range_threshold', 10000)
        skewness_threshold = skew_config.get('skewness_threshold', 2.0)
        min_values = skew_config.get('min_values', 3)
        
        # Check for categorical variables that need encoding
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if min_cardinality <= unique_count <= max_cardinality:
                analysis['needs_categorical_encoding'] = True
                analysis['categorical_cols'].append(col)
        
        # Check for numeric variables that need scaling
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if not col.endswith('_encoded'):  # Skip already encoded columns
                std_val = df[col].std()
                if std_val > variance_threshold or (df[col].max() - df[col].min()) > range_threshold:
                    analysis['needs_numeric_scaling'] = True
                    analysis['high_variance_cols'].append(col)
        
        # Check for skewed distributions
        for col in numeric_cols:
            if not col.endswith(('_encoded', '_std', '_norm')):
                if df[col].notna().sum() >= min_values:
                    skewness = abs(df[col].skew())
                    if skewness > skewness_threshold:
                        analysis['needs_skew_handling'] = True
                        analysis['skewed_cols'].append(col)
        
        # Check for datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            analysis['needs_datetime_features'] = True
            analysis['datetime_cols'] = list(datetime_cols)
        
        # Check for interaction opportunities
        if len(numeric_cols) >= 2:
            analysis['needs_interaction_features'] = True
        
        # Check for missing values in transformed columns
        transformed_cols = [col for col in df.columns 
                          if col.endswith(('_encoded', '_std', '_norm', '_log')) or 
                             '_ratio' in col or any(dt_feature in col for dt_feature in ['_year', '_month', '_day'])]
        
        for col in transformed_cols:
            if df[col].isnull().sum() > 0:
                analysis['needs_missing_handling'] = True
                analysis['missing_value_cols'].append(col)
        
        return analysis
        
    async def process(self, state) -> Any:
        """Main transformation processing pipeline with robust error handling"""
        self.logger.info("Starting data transformation process")
        
        # Reset transformation state for new run
        self.reset_transformation_state()
        
        try:
            # Initialize transform logger
            log_dir = Path(state.run_manifest['artifacts_paths']['logs_dir'])
            log_dir.mkdir(parents=True, exist_ok=True)
            
            transform_logger = TransformLogger(
                log_file=log_dir / "transform_log.jsonl",
                run_id=state.run_manifest['run_id']
            )
            
            # Get data with fallback hierarchy
            input_data = None
            data_source = None
            
            # Try to get data in order of preference
            if hasattr(state, 'cleaned_data') and state.cleaned_data is not None:
                input_data = state.cleaned_data
                data_source = "cleaned_data"
                self.logger.info("Using cleaned data for transformation")
            elif hasattr(state, 'standardized_data') and state.standardized_data is not None:
                input_data = state.standardized_data
                data_source = "standardized_data"
                self.logger.warning("Cleaned data not available, using standardized data")
            elif hasattr(state, 'raw_data') and state.raw_data is not None:
                input_data = state.raw_data
                data_source = "raw_data"
                self.logger.warning("Neither cleaned nor standardized data available, using raw data")
            else:
                raise ValueError("No data available for transformation (checked: cleaned_data, standardized_data, raw_data)")
            
            # Validate input data
            if input_data is None or len(input_data) == 0:
                raise ValueError(f"Input data from {data_source} is empty or None")
            
            self.logger.info(f"Starting transformation with {data_source}: {len(input_data)} rows, {len(input_data.columns)} columns")
            
            # Create working copy
            transformed_data = input_data.copy()
            
            # Track transformations
            transformation_log = []
            
            # Apply transformations with individual error handling
            try:
                self.logger.info("Creating time features...")
                transformed_data, time_log = await self._create_time_features(transformed_data, transform_logger)
                transformation_log.extend(time_log)
            except Exception as e:
                self.logger.warning(f"Time features creation failed: {str(e)}")
                if hasattr(state, 'warnings'):
                    state.warnings.append(f"Time features creation failed: {str(e)}")
            
            try:
                self.logger.info("Creating per-capita metrics...")
                transformed_data, per_capita_log = await self._create_per_capita_metrics(transformed_data, transform_logger)
                transformation_log.extend(per_capita_log)
            except Exception as e:
                self.logger.warning(f"Per-capita metrics creation failed: {str(e)}")
                if hasattr(state, 'warnings'):
                    state.warnings.append(f"Per-capita metrics creation failed: {str(e)}")
            
            try:
                self.logger.info("Creating ratio features...")
                transformed_data, ratio_log = await self._create_ratio_features(transformed_data, transform_logger)
                transformation_log.extend(ratio_log)
            except Exception as e:
                self.logger.warning(f"Ratio features creation failed: {str(e)}")
                if hasattr(state, 'warnings'):
                    state.warnings.append(f"Ratio features creation failed: {str(e)}")
            
            try:
                self.logger.info("Creating trend features...")
                transformed_data, trend_log = await self._create_trend_features(transformed_data, transform_logger)
                transformation_log.extend(trend_log)
            except Exception as e:
                self.logger.warning(f"Trend features creation failed: {str(e)}")
                if hasattr(state, 'warnings'):
                    state.warnings.append(f"Trend features creation failed: {str(e)}")
            
            try:
                self.logger.info("Creating aggregation features...")
                transformed_data, agg_log = await self._create_aggregation_features(transformed_data, transform_logger)
                transformation_log.extend(agg_log)
            except Exception as e:
                self.logger.warning(f"Aggregation features creation failed: {str(e)}")
                if hasattr(state, 'warnings'):
                    state.warnings.append(f"Aggregation features creation failed: {str(e)}")
            
            # Analyze data requirements for conditional transformations
            data_analysis = self.analyze_data_requirements(transformed_data)
            self.logger.info(f"Data analysis completed: {data_analysis}")
            
            # Apply robust data-agnostic transformations conditionally
            if (self.transformation_controls['enable_categorical_encoding'] and 
                (not self.transformation_controls['auto_detect_requirements'] or data_analysis['needs_categorical_encoding'])):
                try:
                    self.logger.info(f"Encoding categorical variables: {data_analysis['categorical_cols']}")
                    transformed_data = self.encode_categorical_variables(transformed_data)
                    transformation_log.append({
                        'feature_type': 'categorical_encoding',
                        'description': f"Applied categorical variable encoding to {len(data_analysis['categorical_cols'])} columns",
                        'transformations_applied': len(self.transformations)
                    })
                except Exception as e:
                    self.logger.warning(f"Categorical encoding failed: {str(e)}")
                    if hasattr(state, 'warnings'):
                        state.warnings.append(f"Categorical encoding failed: {str(e)}")
            else:
                self.logger.info("Skipping categorical encoding - not needed or disabled")
            
            if (self.transformation_controls['enable_numeric_scaling'] and 
                (not self.transformation_controls['auto_detect_requirements'] or data_analysis['needs_numeric_scaling'])):
                try:
                    self.logger.info(f"Scaling numeric variables: {data_analysis['high_variance_cols']}")
                    transformed_data = self.scale_numeric_variables(transformed_data)
                    transformation_log.append({
                        'feature_type': 'numeric_scaling',
                        'description': f"Applied standard scaling to {len(data_analysis['high_variance_cols'])} high-variance columns",
                        'transformations_applied': len(self.transformations)
                    })
                except Exception as e:
                    self.logger.warning(f"Numeric scaling failed: {str(e)}")
                    if hasattr(state, 'warnings'):
                        state.warnings.append(f"Numeric scaling failed: {str(e)}")
            else:
                self.logger.info("Skipping numeric scaling - not needed or disabled")
            
            if (self.transformation_controls['enable_skew_handling'] and 
                (not self.transformation_controls['auto_detect_requirements'] or data_analysis['needs_skew_handling'])):
                try:
                    self.logger.info(f"Handling skewed distributions: {data_analysis['skewed_cols']}")
                    transformed_data = self.handle_skewed_distributions(transformed_data)
                    transformation_log.append({
                        'feature_type': 'skew_handling',
                        'description': f"Applied log transformations to {len(data_analysis['skewed_cols'])} skewed columns",
                        'transformations_applied': len(self.transformations)
                    })
                except Exception as e:
                    self.logger.warning(f"Skew handling failed: {str(e)}")
                    if hasattr(state, 'warnings'):
                        state.warnings.append(f"Skew handling failed: {str(e)}")
            else:
                self.logger.info("Skipping skew handling - not needed or disabled")
            
            if (self.transformation_controls['enable_datetime_features'] and 
                (not self.transformation_controls['auto_detect_requirements'] or data_analysis['needs_datetime_features'])):
                try:
                    self.logger.info(f"Creating datetime features: {data_analysis['datetime_cols']}")
                    transformed_data = self.create_datetime_features(transformed_data)
                    transformation_log.append({
                        'feature_type': 'datetime_features',
                        'description': f"Created datetime features from {len(data_analysis['datetime_cols'])} datetime columns",
                        'transformations_applied': len(self.transformations)
                    })
                except Exception as e:
                    self.logger.warning(f"Datetime features creation failed: {str(e)}")
                    if hasattr(state, 'warnings'):
                        state.warnings.append(f"Datetime features creation failed: {str(e)}")
            else:
                self.logger.info("Skipping datetime features - not needed or disabled")
            
            if (self.transformation_controls['enable_interaction_features'] and 
                (not self.transformation_controls['auto_detect_requirements'] or data_analysis['needs_interaction_features'])):
                try:
                    self.logger.info("Creating interaction features...")
                    transformed_data = self.create_interaction_features(transformed_data)
                    transformation_log.append({
                        'feature_type': 'interaction_features',
                        'description': "Created interaction features between numeric variables",
                        'transformations_applied': len(self.transformations)
                    })
                except Exception as e:
                    self.logger.warning(f"Interaction features creation failed: {str(e)}")
                    if hasattr(state, 'warnings'):
                        state.warnings.append(f"Interaction features creation failed: {str(e)}")
            else:
                self.logger.info("Skipping interaction features - not needed or disabled")
            
            if (self.transformation_controls['enable_missing_handling'] and 
                (not self.transformation_controls['auto_detect_requirements'] or data_analysis['needs_missing_handling'])):
                try:
                    self.logger.info(f"Handling missing values: {data_analysis['missing_value_cols']}")
                    transformed_data = self.handle_missing_after_transformation(transformed_data)
                    transformation_log.append({
                        'feature_type': 'missing_value_handling',
                        'description': f"Handled missing values in {len(data_analysis['missing_value_cols'])} transformed columns",
                        'transformations_applied': len(self.transformations)
                    })
                except Exception as e:
                    self.logger.warning(f"Missing value handling failed: {str(e)}")
                    if hasattr(state, 'warnings'):
                        state.warnings.append(f"Missing value handling failed: {str(e)}")
            else:
                self.logger.info("Skipping missing value handling - not needed or disabled")
            
            # Validate transformations
            try:
                self.logger.info("Validating transformations...")
                validation_results = self.validate_transformations(input_data, transformed_data)
                if not validation_results['is_valid']:
                    self.logger.warning(f"Transformation validation failed: {validation_results['issues']}")
                    if hasattr(state, 'warnings'):
                        state.warnings.extend(validation_results['issues'])
                if validation_results['warnings']:
                    self.logger.warning(f"Transformation warnings: {validation_results['warnings']}")
                    if hasattr(state, 'warnings'):
                        state.warnings.extend(validation_results['warnings'])
            except Exception as e:
                self.logger.warning(f"Transformation validation failed: {str(e)}")
                if hasattr(state, 'warnings'):
                    state.warnings.append(f"Transformation validation failed: {str(e)}")
            
            # Create feature catalog
            try:
                feature_catalog = await self._create_feature_catalog(transformed_data, input_data, transformation_log)
            except Exception as e:
                self.logger.warning(f"Feature catalog creation failed: {str(e)}")
                feature_catalog = {
                    "error": f"Feature catalog creation failed: {str(e)}",
                    "transformation_operations": len(transformation_log),
                    "final_shape": transformed_data.shape
                }
            
            # Ensure output directories exist
            data_dir = Path(state.run_manifest['artifacts_paths']['data_dir']) / "transformed"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            docs_dir = Path(state.run_manifest['artifacts_paths']['docs_dir'])
            docs_dir.mkdir(parents=True, exist_ok=True)
            
            # Save transformed data
            try:
                output_path = data_dir / f"{state.run_manifest['dataset_info']['dataset_name']}_transformed.csv"
                transformed_data.to_csv(output_path, index=False)
                self.logger.info(f"Saved transformed data to {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to save transformed data: {str(e)}")
                raise
            
            # Save feature catalog
            try:
                catalog_path = docs_dir / "feature_catalog.json"
                with open(catalog_path, 'w') as f:
                    json.dump(feature_catalog, f, indent=2, default=str)
                self.logger.info(f"Saved feature catalog to {catalog_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save feature catalog: {str(e)}")
            
            # Update state
            state.transformed_data = transformed_data
            state.feature_catalog = feature_catalog
            state.transformation_log = transformation_log
            state.data_source_used = data_source
            state.transformation_summary = self.get_transformation_summary()
            state.data_analysis = data_analysis
            
            # Initialize warnings list if it doesn't exist
            if not hasattr(state, 'warnings'):
                state.warnings = []
            
            # Count applied transformations
            applied_transformations = []
            if data_analysis['needs_categorical_encoding'] and self.transformation_controls['enable_categorical_encoding']:
                applied_transformations.append(f"categorical encoding ({len(data_analysis['categorical_cols'])} cols)")
            if data_analysis['needs_numeric_scaling'] and self.transformation_controls['enable_numeric_scaling']:
                applied_transformations.append(f"numeric scaling ({len(data_analysis['high_variance_cols'])} cols)")
            if data_analysis['needs_skew_handling'] and self.transformation_controls['enable_skew_handling']:
                applied_transformations.append(f"skew handling ({len(data_analysis['skewed_cols'])} cols)")
            if data_analysis['needs_datetime_features'] and self.transformation_controls['enable_datetime_features']:
                applied_transformations.append(f"datetime features ({len(data_analysis['datetime_cols'])} cols)")
            if data_analysis['needs_interaction_features'] and self.transformation_controls['enable_interaction_features']:
                applied_transformations.append("interaction features")
            if data_analysis['needs_missing_handling'] and self.transformation_controls['enable_missing_handling']:
                applied_transformations.append(f"missing value handling ({len(data_analysis['missing_value_cols'])} cols)")
            
            success_msg = f"Transformation completed: {len(transformed_data)} rows, {len(transformed_data.columns)} columns"
            feature_msg = f"Created {len(transformed_data.columns) - len(input_data.columns)} new features"
            transformation_msg = f"Applied {len(self.transformations)} transformations: {', '.join(applied_transformations) if applied_transformations else 'none needed'}"
            
            self.logger.info(success_msg)
            self.logger.info(feature_msg)
            self.logger.info(transformation_msg)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Transformation failed: {str(e)}")
            if not hasattr(state, 'errors'):
                state.errors = []
            state.errors.append(f"Transformation error: {str(e)}")
            
            # Try to preserve input data as transformed data
            try:
                if hasattr(state, 'cleaned_data') and state.cleaned_data is not None:
                    state.transformed_data = state.cleaned_data
                elif hasattr(state, 'standardized_data') and state.standardized_data is not None:
                    state.transformed_data = state.standardized_data
                elif hasattr(state, 'raw_data') and state.raw_data is not None:
                    state.transformed_data = state.raw_data
            except:
                pass
            
            return state
    
    async def _create_time_features(self, df: pd.DataFrame, transform_logger) -> Tuple[pd.DataFrame, List[Dict]]:
        """Create time-based features from date columns"""
        log_entries = []
        
        # Find datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Also try to parse string columns that might be dates
        for col in df.select_dtypes(include=['object']).columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month']):
                try:
                    parsed_dates = pd.to_datetime(df[col], errors='coerce')
                    if parsed_dates.notna().sum() / len(df) > 0.5:  # If >50% parse successfully
                        df[col] = parsed_dates
                        datetime_cols.append(col)
                except:
                    continue
        
        for col in datetime_cols:
            try:
                # Extract time components
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_quarter"] = df[col].dt.quarter
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                df[f"{col}_dayofyear"] = df[col].dt.dayofyear
                df[f"{col}_weekofyear"] = df[col].dt.isocalendar().week.astype('int64')
                
                # Create time flags
                df[f"{col}_is_weekend"] = df[col].dt.dayofweek >= 5
                df[f"{col}_is_month_start"] = df[col].dt.is_month_start
                df[f"{col}_is_month_end"] = df[col].dt.is_month_end
                df[f"{col}_is_quarter_start"] = df[col].dt.is_quarter_start
                df[f"{col}_is_quarter_end"] = df[col].dt.is_quarter_end
                
                # Log transformation
                transform_logger.log_transform(
                    agent="transformation",
                    action="create_time_features",
                    column=col,
                    rows_affected=len(df),
                    rule_id="time_features_v1",
                    rationale=f"Extracted time components from datetime column {col}",
                    confidence="high"
                )
                
                log_entries.append({
                    'feature_type': 'time_features',
                    'source_column': col,
                    'created_features': [f"{col}_year", f"{col}_month", f"{col}_quarter", f"{col}_day"],
                    'description': f"Time components extracted from {col}"
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to create time features from {col}: {e}")
        
        return df, log_entries
    
    async def _create_per_capita_metrics(self, df: pd.DataFrame, transform_logger) -> Tuple[pd.DataFrame, List[Dict]]:
        """Create per-capita and normalized metrics"""
        log_entries = []
        
        # Find potential denominator columns
        denominators = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in self.feature_config.get('per_capita_denominators', ['population'])):
                if df[col].dtype in ['int64', 'float64'] and df[col].sum() > 0:
                    denominators.append(col)
        
        # Find numeric columns that could be normalized
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for denom_col in denominators:
            for num_col in numeric_cols:
                if num_col != denom_col and not num_col.endswith('_per_capita') and not num_col.endswith('_normalized'):
                    try:
                        # Avoid division by zero
                        safe_denominator = df[denom_col].replace(0, np.nan)
                        
                        # Create per-capita metric
                        per_capita_col = f"{num_col}_per_{denom_col.replace('_', '')}"
                        df[per_capita_col] = df[num_col] / safe_denominator
                        
                        # Create per-1000 metric for common cases
                        if 'population' in denom_col.lower():
                            per_1000_col = f"{num_col}_per_1000"
                            df[per_1000_col] = (df[num_col] / safe_denominator) * 1000
                        
                        # Log transformation
                        transform_logger.log_transform(
                            agent="transformation",
                            action="create_per_capita",
                            column=f"{num_col}_per_{denom_col}",
                            rows_affected=len(df),
                            rule_id="per_capita_v1",
                            rationale=f"Created per-capita metric: {num_col} per {denom_col}",
                            parameters={'numerator': num_col, 'denominator': denom_col},
                            confidence="high"
                        )
                        
                        log_entries.append({
                            'feature_type': 'per_capita',
                            'source_columns': [num_col, denom_col],
                            'created_features': [per_capita_col],
                            'description': f"Per-capita calculation: {num_col} normalized by {denom_col}"
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to create per-capita for {num_col}/{denom_col}: {e}")
        
        return df, log_entries
    
    async def _create_ratio_features(self, df: pd.DataFrame, transform_logger) -> Tuple[pd.DataFrame, List[Dict]]:
        """Create ratio and percentage features"""
        log_entries = []
        
        # Get ratio pairs from config
        ratio_pairs = self.feature_config.get('ratio_pairs', [])
        
        # Find columns matching ratio pair patterns
        for pair in ratio_pairs:
            if len(pair) != 2:
                continue
                
            col1_pattern, col2_pattern = pair
            
            # Find matching columns
            col1_matches = [col for col in df.columns if col1_pattern.lower() in col.lower()]
            col2_matches = [col for col in df.columns if col2_pattern.lower() in col.lower()]
            
            for col1 in col1_matches:
                for col2 in col2_matches:
                    if col1 != col2 and df[col1].dtype in ['int64', 'float64'] and df[col2].dtype in ['int64', 'float64']:
                        try:
                            # Create ratio
                            ratio_col = f"{col1}_to_{col2}_ratio"
                            safe_denominator = df[col2].replace(0, np.nan)
                            df[ratio_col] = df[col1] / safe_denominator
                            
                            # Create percentage
                            if df[col1].sum() > 0 and df[col2].sum() > 0:
                                pct_col = f"{col1}_pct_of_total"
                                total = df[col1] + df[col2]
                                df[pct_col] = (df[col1] / total.replace(0, np.nan)) * 100
                            
                            # Log transformation
                            transform_logger.log_transform(
                                agent="transformation",
                                action="create_ratio",
                                column=f"{col1}/{col2}",
                                rows_affected=len(df),
                                rule_id="ratio_features_v1",
                                rationale=f"Created ratio: {col1} to {col2}",
                                parameters={'numerator': col1, 'denominator': col2},
                                confidence="high"
                            )
                            
                            log_entries.append({
                                'feature_type': 'ratio',
                                'source_columns': [col1, col2],
                                'created_features': [ratio_col],
                                'description': f"Ratio calculation: {col1} divided by {col2}"
                            })
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to create ratio for {col1}/{col2}: {e}")
        
        return df, log_entries
    
    async def _create_trend_features(self, df: pd.DataFrame, transform_logger) -> Tuple[pd.DataFrame, List[Dict]]:
        """Create trend and change features"""
        log_entries = []
        
        # Find time/sequence columns
        time_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['year', 'month', 'date', 'time']):
                time_cols.append(col)
        
        if not time_cols:
            return df, log_entries
        
        # Get numeric columns for trend analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create rolling averages
        trend_windows = self.feature_config.get('trend_windows', [3, 6, 12])
        
        for time_col in time_cols[:1]:  # Use first time column
            # Sort by time column
            df_sorted = df.sort_values(time_col)
            
            for num_col in numeric_cols[:5]:  # Limit to 5 columns
                try:
                    # Create rolling averages
                    for window in trend_windows:
                        if len(df) >= window:
                            rolling_col = f"{num_col}_rolling_{window}"
                            df[rolling_col] = df_sorted[num_col].rolling(window=window, min_periods=1).mean()
                    
                    # Create lag features
                    lag_col = f"{num_col}_lag_1"
                    df[lag_col] = df_sorted[num_col].shift(1)
                    
                    # Create change features
                    change_col = f"{num_col}_change"
                    df[change_col] = df_sorted[num_col].diff()
                    
                    # Create percentage change
                    pct_change_col = f"{num_col}_pct_change"
                    df[pct_change_col] = df_sorted[num_col].pct_change() * 100
                    
                    # Log transformation
                    transform_logger.log_transform(
                        agent="transformation",
                        action="create_trend_features",
                        column=num_col,
                        rows_affected=len(df),
                        rule_id="trend_features_v1",
                        rationale=f"Created trend features for {num_col} based on {time_col}",
                        parameters={'time_column': time_col, 'windows': trend_windows},
                        confidence="medium"
                    )
                    
                    created_features = [f"{num_col}_rolling_{w}" for w in trend_windows if len(df) >= w]
                    created_features.extend([lag_col, change_col, pct_change_col])
                    
                    log_entries.append({
                        'feature_type': 'trend',
                        'source_columns': [num_col, time_col],
                        'created_features': created_features,
                        'description': f"Trend analysis features for {num_col} over time"
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create trend features for {num_col}: {e}")
        
        return df, log_entries
    
    async def _create_aggregation_features(self, df: pd.DataFrame, transform_logger) -> Tuple[pd.DataFrame, List[Dict]]:
        """Create aggregation features grouped by categorical columns"""
        log_entries = []
        
        # Find categorical columns suitable for grouping
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter categorical columns with reasonable cardinality
        suitable_categorical = []
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 20:  # Between 2 and 20 unique values
                suitable_categorical.append(col)
        
        # Limit to prevent feature explosion
        suitable_categorical = suitable_categorical[:2]
        numeric_cols = numeric_cols[:3]
        
        for group_col in suitable_categorical:
            for metric_col in numeric_cols:
                try:
                    # Create group statistics
                    group_stats = df.groupby(group_col)[metric_col].agg(['mean', 'sum', 'std', 'count']).reset_index()
                    
                    # Rename columns
                    stat_cols = [f"{metric_col}_{group_col}_mean", f"{metric_col}_{group_col}_sum", 
                               f"{metric_col}_{group_col}_std", f"{metric_col}_{group_col}_count"]
                    group_stats.columns = [group_col] + stat_cols
                    
                    # Merge back to original dataframe
                    df = df.merge(group_stats, on=group_col, how='left')
                    
                    # Create group rank
                    rank_col = f"{metric_col}_{group_col}_rank"
                    df[rank_col] = df.groupby(group_col)[metric_col].rank(ascending=False)
                    
                    # Log transformation
                    transform_logger.log_transform(
                        agent="transformation",
                        action="create_group_aggregations",
                        column=f"{metric_col}_by_{group_col}",
                        rows_affected=len(df),
                        rule_id="group_agg_v1",
                        rationale=f"Created group statistics for {metric_col} grouped by {group_col}",
                        parameters={'group_column': group_col, 'metric_column': metric_col},
                        confidence="medium"
                    )
                    
                    created_features = stat_cols + [rank_col]
                    
                    log_entries.append({
                        'feature_type': 'aggregation',
                        'source_columns': [metric_col, group_col],
                        'created_features': created_features,
                        'description': f"Group aggregation statistics for {metric_col} by {group_col}"
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create aggregation features for {metric_col} by {group_col}: {e}")
        
        return df, log_entries
    
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables for analysis"""
        df_transformed = df.copy()
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Get thresholds from config
        cat_config = self.transformation_controls.get('categorical_encoding', {})
        min_cardinality = cat_config.get('min_cardinality', 2)
        max_cardinality = cat_config.get('max_cardinality', 50)
        
        self.logger.info(f"Encoding {len(categorical_cols)} categorical columns...")
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            
            # Skip if outside cardinality range
            if unique_count < min_cardinality or unique_count > max_cardinality:
                self.logger.warning(f"Skipping encoding for '{col}' - cardinality {unique_count} outside range [{min_cardinality}, {max_cardinality}]")
                continue
            
            # For binary categorical variables
            if unique_count == 2:
                # Simple binary encoding
                unique_values = df[col].dropna().unique()
                if len(unique_values) == 2:
                    df_transformed[f'{col}_encoded'] = df_transformed[col].map({
                        unique_values[0]: 0,
                        unique_values[1]: 1
                    })
                    self.transformations.append(f"Binary encoded '{col}' -> '{col}_encoded'")
                    self.logger.info(f"Binary encoded column '{col}'")
            
            # For low cardinality categorical variables (<=10 categories)
            elif unique_count <= 10:
                # One-hot encoding
                dummies = pd.get_dummies(df_transformed[col], prefix=col, dummy_na=True)
                df_transformed = pd.concat([df_transformed, dummies], axis=1)
                self.transformations.append(f"One-hot encoded '{col}' into {len(dummies.columns)} dummy variables")
                self.logger.info(f"One-hot encoded column '{col}' into {len(dummies.columns)} columns")
            
            # For medium cardinality (11-50 categories)
            else:
                # Label encoding
                le = LabelEncoder()
                non_null_mask = df_transformed[col].notna()
                df_transformed.loc[non_null_mask, f'{col}_encoded'] = le.fit_transform(
                    df_transformed.loc[non_null_mask, col]
                )
                self.encoders[col] = le
                self.transformations.append(f"Label encoded '{col}' -> '{col}_encoded'")
                self.logger.info(f"Label encoded column '{col}'")
        
        return df_transformed
    
    def scale_numeric_variables(self, df: pd.DataFrame, method: str = None) -> pd.DataFrame:
        """Scale numeric variables"""
        df_transformed = df.copy()
        
        # Get method from config if not provided
        if method is None:
            num_config = self.transformation_controls.get('numeric_scaling', {})
            method = num_config.get('method', 'standard')
        
        # Get numeric columns (excluding encoded columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if not col.endswith('_encoded')]
        
        if len(numeric_cols) == 0:
            self.logger.info("No numeric columns to scale")
            return df_transformed
        
        self.logger.info(f"Scaling {len(numeric_cols)} numeric columns using {method} scaling...")
        
        # Choose scaler
        if method == 'standard':
            scaler = StandardScaler()
            suffix = '_std'
        elif method == 'minmax':
            scaler = MinMaxScaler()
            suffix = '_norm'
        else:
            self.logger.warning(f"Unknown scaling method: {method}, using standard scaling")
            scaler = StandardScaler()
            suffix = '_std'
        
        # Apply scaling
        for col in numeric_cols:
            if df[col].notna().sum() > 0:  # Only scale if we have non-null values
                non_null_mask = df_transformed[col].notna()
                scaled_values = scaler.fit_transform(df_transformed.loc[non_null_mask, [col]])
                df_transformed.loc[non_null_mask, f'{col}{suffix}'] = scaled_values.ravel()
                
                self.scalers[col] = scaler
                self.transformations.append(f"Applied {method} scaling to '{col}' -> '{col}{suffix}'")
                self.logger.info(f"Scaled column '{col}' using {method} scaling")
        
        return df_transformed
    
    def handle_skewed_distributions(self, df: pd.DataFrame, skewness_threshold: float = None) -> pd.DataFrame:
        """Apply log transformation to highly skewed numeric variables"""
        df_transformed = df.copy()
        
        # Get thresholds from config if not provided
        if skewness_threshold is None:
            skew_config = self.transformation_controls.get('skew_handling', {})
            skewness_threshold = skew_config.get('skewness_threshold', 2.0)
            min_values = skew_config.get('min_values', 3)
        else:
            min_values = 3
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if not col.endswith(('_encoded', '_std', '_norm'))]
        
        self.logger.info(f"Checking skewness for {len(numeric_cols)} numeric columns...")
        
        for col in numeric_cols:
            if df[col].notna().sum() < min_values:  # Need at least min_values
                continue
                
            # Calculate skewness
            skewness = df[col].skew()
            
            if abs(skewness) > skewness_threshold:
                # Apply log transformation (add 1 to handle zeros)
                if df[col].min() >= 0:  # Only for non-negative values
                    df_transformed[f'{col}_log'] = np.log1p(df_transformed[col])
                    self.transformations.append(f"Applied log transformation to '{col}' (skewness: {skewness:.2f}) -> '{col}_log'")
                    self.logger.info(f"Applied log transformation to '{col}' (skewness: {skewness:.2f})")
                else:
                    self.logger.warning(f"Skipped log transformation for '{col}' - contains negative values")
        
        return df_transformed
    
    def create_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from datetime columns"""
        df_transformed = df.copy()
        
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        if len(datetime_cols) == 0:
            return df_transformed
        
        self.logger.info(f"Creating datetime features for {len(datetime_cols)} columns...")
        
        for col in datetime_cols:
            if df[col].notna().sum() == 0:
                continue
            
            # Extract common datetime features
            df_transformed[f'{col}_year'] = df_transformed[col].dt.year
            df_transformed[f'{col}_month'] = df_transformed[col].dt.month
            df_transformed[f'{col}_day'] = df_transformed[col].dt.day
            df_transformed[f'{col}_dayofweek'] = df_transformed[col].dt.dayofweek
            df_transformed[f'{col}_quarter'] = df_transformed[col].dt.quarter
            
            # Create binary features for common patterns
            df_transformed[f'{col}_is_weekend'] = df_transformed[f'{col}_dayofweek'].isin([5, 6]).astype(int)
            df_transformed[f'{col}_is_month_start'] = df_transformed[col].dt.is_month_start.astype(int)
            df_transformed[f'{col}_is_month_end'] = df_transformed[col].dt.is_month_end.astype(int)
            
            features_created = 8
            self.transformations.append(f"Created {features_created} datetime features from '{col}'")
            self.logger.info(f"Created {features_created} datetime features from '{col}'")
        
        return df_transformed
    
    def create_interaction_features(self, df: pd.DataFrame, max_interactions: int = None) -> pd.DataFrame:
        """Create simple interaction features between numeric variables"""
        df_transformed = df.copy()
        
        # Get max_interactions from config if not provided
        if max_interactions is None:
            interaction_config = self.transformation_controls.get('interaction_features', {})
            max_interactions = interaction_config.get('max_interactions', 5)
            min_numeric_cols = interaction_config.get('min_numeric_cols', 2)
        else:
            min_numeric_cols = 2
        
        # Get original numeric columns (not transformed ones)
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if not col.endswith(('_encoded', '_std', '_norm', '_log'))]
        
        if len(numeric_cols) < min_numeric_cols:
            self.logger.info(f"Not enough numeric columns for interaction features (need {min_numeric_cols}, have {len(numeric_cols)})")
            return df_transformed
        
        self.logger.info("Creating interaction features...")
        
        interactions_created = 0
        for i, col1 in enumerate(numeric_cols):
            if interactions_created >= max_interactions:
                break
            for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                if interactions_created >= max_interactions:
                    break
                
                # Create ratio feature
                col2_nonzero = df_transformed[col2] != 0
                if col2_nonzero.sum() > 0:
                    df_transformed[f'{col1}_{col2}_ratio'] = df_transformed[col1] / df_transformed[col2].where(col2_nonzero, np.nan)
                    interactions_created += 1
                    self.transformations.append(f"Created ratio feature '{col1}_{col2}_ratio'")
                
                if interactions_created >= max_interactions:
                    break
        
        self.logger.info(f"Created {interactions_created} interaction features")
        return df_transformed
    
    def handle_missing_after_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle any missing values created during transformation"""
        df_transformed = df.copy()
        
        # Check for new missing values in transformed columns
        transformed_cols = [col for col in df_transformed.columns 
                          if col.endswith(('_encoded', '_std', '_norm', '_log')) or 
                             '_ratio' in col or any(dt_feature in col for dt_feature in ['_year', '_month', '_day'])]
        
        for col in transformed_cols:
            missing_count = df_transformed[col].isnull().sum()
            if missing_count > 0:
                if pd.api.types.is_numeric_dtype(df_transformed[col]):
                    # Fill with median for numeric
                    median_val = df_transformed[col].median()
                    df_transformed[col] = df_transformed[col].fillna(median_val)
                    self.transformations.append(f"Filled {missing_count} missing values in '{col}' with median")
                else:
                    # Fill with mode for categorical
                    mode_val = df_transformed[col].mode().iloc[0] if not df_transformed[col].mode().empty else 0
                    df_transformed[col] = df_transformed[col].fillna(mode_val)
                    self.transformations.append(f"Filled {missing_count} missing values in '{col}' with mode")
        
        return df_transformed
    
    def validate_transformations(self, df_original: pd.DataFrame, df_transformed: pd.DataFrame) -> Dict[str, Any]:
        """Validate transformation results"""
        validation = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'summary': {}
        }
        
        # Check if we created too many columns
        original_cols = df_original.shape[1]
        transformed_cols = df_transformed.shape[1]
        new_cols = transformed_cols - original_cols
        
        validation['summary'] = {
            'original_columns': original_cols,
            'transformed_columns': transformed_cols,
            'new_columns_created': new_cols,
            'transformations_applied': len(self.transformations)
        }
        
        if new_cols > original_cols * 2:  # More than double the columns
            validation['warnings'].append(f"Created many new columns ({new_cols}), consider feature selection")
        
        # Check for infinite values
        numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(df_transformed[col]).any():
                validation['issues'].append(f"Column '{col}' contains infinite values")
                validation['is_valid'] = False
        
        # Check for extremely large values (potential overflow)
        for col in numeric_cols:
            if df_transformed[col].abs().max() > 1e10:
                validation['warnings'].append(f"Column '{col}' has very large values")
        
        return validation
    
    def get_transformation_summary(self) -> Dict[str, Any]:
        """Get summary of transformations applied"""
        return {
            'total_transformations': len(self.transformations),
            'encoders_created': len(self.encoders),
            'scalers_created': len(self.scalers),
            'transformation_details': self.transformations,
            'encoder_columns': list(self.encoders.keys()),
            'scaler_columns': list(self.scalers.keys())
        }
    
    async def _create_feature_catalog(self, transformed_data: pd.DataFrame, 
                                    original_data: pd.DataFrame, 
                                    transformation_log: List[Dict]) -> Dict[str, Any]:
        """Create comprehensive feature catalog"""
        
        # Count features by type
        feature_counts = {}
        all_created_features = []
        
        for entry in transformation_log:
            feature_type = entry.get('feature_type', 'unknown')
            created_features = entry.get('created_features', [])
            
            feature_counts[feature_type] = feature_counts.get(feature_type, 0) + len(created_features)
            all_created_features.extend(created_features)
        
        # Analyze feature importance (basic)
        feature_importance = {}
        for feature in all_created_features:
            if feature in transformed_data.columns:
                # Simple importance based on variance and non-null values
                if transformed_data[feature].dtype in ['int64', 'float64']:
                    variance_score = transformed_data[feature].var() if transformed_data[feature].var() > 0 else 0
                    completeness_score = transformed_data[feature].notna().mean()
                    feature_importance[feature] = variance_score * completeness_score
                else:
                    feature_importance[feature] = transformed_data[feature].notna().mean()
        
        # Create catalog
        catalog = {
            "catalog_metadata": {
                "creation_time": datetime.utcnow().isoformat(),
                "original_features": len(original_data.columns),
                "total_features": len(transformed_data.columns),
                "created_features": len(all_created_features),
                "transformation_operations": len(transformation_log)
            },
            "feature_summary": {
                "by_type": feature_counts,
                "total_by_type": {
                    "time_features": feature_counts.get('time_features', 0),
                    "per_capita_features": feature_counts.get('per_capita', 0),
                    "ratio_features": feature_counts.get('ratio', 0),
                    "trend_features": feature_counts.get('trend', 0),
                    "aggregation_features": feature_counts.get('aggregation', 0),
                    "categorical_encoding": feature_counts.get('categorical_encoding', 0),
                    "numeric_scaling": feature_counts.get('numeric_scaling', 0),
                    "skew_handling": feature_counts.get('skew_handling', 0),
                    "datetime_features": feature_counts.get('datetime_features', 0),
                    "interaction_features": feature_counts.get('interaction_features', 0),
                    "missing_value_handling": feature_counts.get('missing_value_handling', 0)
                }
            },
            "feature_importance": dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)),
            "transformation_details": transformation_log,
            "feature_list": {
                "original_features": list(original_data.columns),
                "created_features": all_created_features,
                "all_features": list(transformed_data.columns)
            },
            "data_quality": {
                "shape_original": original_data.shape,
                "shape_transformed": transformed_data.shape,
                "memory_usage_mb": transformed_data.memory_usage(deep=True).sum() / (1024 * 1024),
                "completeness_by_feature": {
                    col: transformed_data[col].notna().mean() 
                    for col in transformed_data.columns
                }
            },
            "recommendations": await self._generate_feature_recommendations(transformed_data, transformation_log)
        }
        
        return catalog
    
    async def _generate_feature_recommendations(self, df: pd.DataFrame, transformation_log: List[Dict]) -> List[str]:
        """Generate recommendations for additional feature engineering"""
        recommendations = []
        
        # Check for missing feature types
        feature_types = {entry.get('feature_type') for entry in transformation_log}
        
        if 'time_features' not in feature_types:
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                recommendations.append("Consider creating time-based features from date columns for temporal analysis")
        
        if 'per_capita' not in feature_types:
            pop_keywords = ['population', 'households', 'residents']
            pop_cols = [col for col in df.columns if any(kw in col.lower() for kw in pop_keywords)]
            if pop_cols:
                recommendations.append("Consider creating per-capita metrics using population denominators")
        
        # Check for high correlation features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            try:
                corr_matrix = df[numeric_cols].corr().abs()
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if corr_matrix.iloc[i, j] > 0.9:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
                
                if high_corr_pairs:
                    recommendations.append(f"Consider removing {len(high_corr_pairs)} highly correlated feature pairs to reduce redundancy")
            except:
                pass
        
        # Check for feature scaling needs
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        scale_needed = []
        for col in numeric_cols:
            if df[col].std() > 1000 or (df[col].max() - df[col].min()) > 10000:
                scale_needed.append(col)
        
        if scale_needed:
            recommendations.append(f"Consider feature scaling for {len(scale_needed)} columns with high variance")
        
        # Check for categorical encoding opportunities
        categorical_cols = df.select_dtypes(include=['object']).columns
        high_cardinality = [col for col in categorical_cols if df[col].nunique() > 50]
        if high_cardinality:
            recommendations.append(f"Consider encoding or grouping {len(high_cardinality)} high-cardinality categorical features")
        
        return recommendations
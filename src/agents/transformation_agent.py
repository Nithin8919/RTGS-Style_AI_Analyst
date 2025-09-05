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
        
    async def process(self, state) -> Any:
        """Main transformation processing pipeline"""
        self.logger.info("Starting data transformation process")
        
        try:
            # Initialize transform logger
            transform_logger = TransformLogger(
                log_file=Path(state.run_manifest['artifacts_paths']['logs_dir']) / "transform_log.jsonl",
                run_id=state.run_manifest['run_id']
            )
            
            # Get cleaned data
            cleaned_data = getattr(state, 'cleaned_data', state.standardized_data)
            if cleaned_data is None:
                raise ValueError("No cleaned data available for transformation")
            
            # Create working copy
            transformed_data = cleaned_data.copy()
            
            # Track transformations
            transformation_log = []
            
            # Apply transformations
            self.logger.info("Creating time features...")
            transformed_data, time_log = await self._create_time_features(transformed_data, transform_logger)
            transformation_log.extend(time_log)
            
            self.logger.info("Creating per-capita metrics...")
            transformed_data, per_capita_log = await self._create_per_capita_metrics(transformed_data, transform_logger)
            transformation_log.extend(per_capita_log)
            
            self.logger.info("Creating ratio features...")
            transformed_data, ratio_log = await self._create_ratio_features(transformed_data, transform_logger)
            transformation_log.extend(ratio_log)
            
            self.logger.info("Creating trend features...")
            transformed_data, trend_log = await self._create_trend_features(transformed_data, transform_logger)
            transformation_log.extend(trend_log)
            
            self.logger.info("Creating aggregation features...")
            transformed_data, agg_log = await self._create_aggregation_features(transformed_data, transform_logger)
            transformation_log.extend(agg_log)
            
            # Create feature catalog
            feature_catalog = await self._create_feature_catalog(transformed_data, cleaned_data, transformation_log)
            
            # Save transformed data
            output_dir = Path(state.run_manifest['artifacts_paths']['data_dir']) / "transformed"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"{state.run_manifest['dataset_info']['dataset_name']}_transformed.csv"
            transformed_data.to_csv(output_path, index=False)
            
            # Save feature catalog
            catalog_path = Path(state.run_manifest['artifacts_paths']['docs_dir']) / "feature_catalog.json"
            with open(catalog_path, 'w') as f:
                json.dump(feature_catalog, f, indent=2, default=str)
            
            # Update state
            state.transformed_data = transformed_data
            state.feature_catalog = feature_catalog
            state.transformation_log = transformation_log
            
            self.logger.info(f"Transformation completed: {len(transformed_data)} rows, {len(transformed_data.columns)} columns")
            self.logger.info(f"Created {len(transformed_data.columns) - len(cleaned_data.columns)} new features")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Transformation failed: {str(e)}")
            state.errors.append(f"Transformation error: {str(e)}")
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
                df[f"{col}_weekofyear"] = df[col].dt.isocalendar().week
                
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
                    "aggregation_features": feature_counts.get('aggregation', 0)
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
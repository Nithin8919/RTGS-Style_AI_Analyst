# Transformation & Feature Engineering Agent (3.5)
"""
RTGS AI Analyst - Transformation Agent
Handles feature engineering, derived columns, and data transformations
"""

import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re

from src.utils.logging import get_agent_logger, TransformLogger


class TransformationAgent:
    """Agent responsible for feature engineering and data transformation"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = get_agent_logger("transformation")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
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
            cleaned_data = getattr(state, 'cleaned_data', state.raw_data)
            if cleaned_data is None:
                raise ValueError("No cleaned data available for transformation")
            
            # Create working copy
            transformed_data = cleaned_data.copy()
            
            # Track transformation operations
            transformation_log = []
            
            # Apply feature engineering transformations
            transformed_data, time_features = await self._create_time_features(
                transformed_data, transform_logger
            )
            transformation_log.extend(time_features)
            
            transformed_data, derived_features = await self._create_derived_features(
                transformed_data, transform_logger, state.run_manifest
            )
            transformation_log.extend(derived_features)
            
            transformed_data, aggregation_features = await self._create_aggregation_features(
                transformed_data, transform_logger
            )
            transformation_log.extend(aggregation_features)
            
            transformed_data, categorical_features = await self._create_categorical_features(
                transformed_data, transform_logger
            )
            transformation_log.extend(categorical_features)
            
            # Create transformation summary
            transformation_summary = self._create_transformation_summary(
                cleaned_data, transformed_data, transformation_log
            )
            
            # Save transformed data
            transformed_path = Path(state.run_manifest['run_config']['output_dir']) / "data" / "transformed" / f"{state.run_manifest['dataset_info']['dataset_name']}_transformed.csv"
            transformed_data.to_csv(transformed_path, index=False)
            
            # Save transformation log
            log_path = Path(state.run_manifest['artifacts_paths']['docs_dir']) / "transformation_log.jsonl"
            with open(log_path, 'w') as f:
                for entry in transformation_log:
                    f.write(json.dumps(entry) + '\n')
            
            # Update state
            state.transformed_data = transformed_data
            state.transformation_log = transformation_log
            state.transformation_summary = transformation_summary
            state.transformed_path = str(transformed_path)
            
            self.logger.info(f"Transformation completed: {len(transformed_data)} rows, {len(transformed_data.columns)} columns")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Data transformation failed: {str(e)}")
            state.errors.append(f"Data transformation failed: {str(e)}")
            return state

    async def _create_time_features(self, df: pd.DataFrame, transform_logger: TransformLogger) -> Tuple[pd.DataFrame, List[Dict]]:
        """Create time-based features from date columns"""
        self.logger.info("Creating time-based features")
        
        df_time = df.copy()
        time_features = []
        
        # Identify date columns
        date_columns = []
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
                date_columns.append(col)
        
        for col in date_columns:
            try:
                # Ensure column is datetime
                if df_time[col].dtype != 'datetime64[ns]':
                    df_time[col] = pd.to_datetime(df_time[col], errors='coerce')
                
                # Extract time components
                df_time[f'{col}_year'] = df_time[col].dt.year
                df_time[f'{col}_month'] = df_time[col].dt.month
                df_time[f'{col}_quarter'] = df_time[col].dt.quarter
                df_time[f'{col}_day_of_week'] = df_time[col].dt.dayofweek
                
                # Log transformation
                transform_logger.log_transform(
                    agent="transformation",
                    action="create_time_features",
                    column=col,
                    rows_affected=len(df_time),
                    rule_id="time_features_v1",
                    rationale=f"Extracted year, month, quarter, day_of_week from {col}",
                    confidence="high"
                )
                
                time_features.append({
                    'base_column': col,
                    'derived_columns': [f'{col}_year', f'{col}_month', f'{col}_quarter', f'{col}_day_of_week'],
                    'transformation_type': 'time_decomposition',
                    'description': 'Extracted temporal components from date column'
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to create time features for {col}: {str(e)}")
        
        return df_time, time_features

    async def _create_derived_features(self, df: pd.DataFrame, transform_logger: TransformLogger, run_manifest: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
        """Create derived features based on domain and existing columns"""
        self.logger.info("Creating derived features")
        
        df_derived = df.copy()
        derived_features = []
        
        domain = run_manifest['dataset_info']['domain_hint']
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Per-capita calculations (if population-like columns exist)
        population_keywords = ['population', 'households', 'families', 'residents']
        population_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in population_keywords)]
        
        if population_cols and numeric_columns:
            pop_col = population_cols[0]  # Use first population column
            
            for metric_col in numeric_columns[:5]:  # Limit to first 5 numeric columns
                if metric_col != pop_col and df_derived[pop_col].sum() > 0:
                    try:
                        per_capita_col = f'{metric_col}_per_1000'
                        df_derived[per_capita_col] = (df_derived[metric_col] / df_derived[pop_col]) * 1000
                        
                        transform_logger.log_transform(
                            agent="transformation",
                            action="create_per_capita",
                            column=metric_col,
                            rows_affected=len(df_derived),
                            rule_id="per_capita_v1",
                            rationale=f"Created per-capita metric: {per_capita_col}",
                            parameters={'denominator': pop_col, 'multiplier': 1000},
                            confidence="high"
                        )
                        
                        derived_features.append({
                            'base_columns': [metric_col, pop_col],
                            'derived_column': per_capita_col,
                            'transformation_type': 'per_capita',
                            'description': f'Per-capita calculation: {metric_col} per 1000 {pop_col}'
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to create per-capita for {metric_col}: {str(e)}")
        
        # Growth rate calculations (if time series data exists)
        if len(numeric_columns) > 0:
            # Simple year-over-year growth (requires year column)
            year_cols = [col for col in df.columns if 'year' in col.lower()]
            if year_cols:
                year_col = year_cols[0]
                
                for metric_col in numeric_columns[:3]:  # Limit calculations
                    try:
                        # Calculate year-over-year growth
                        df_sorted = df_derived.sort_values(year_col)
                        growth_col = f'{metric_col}_yoy_growth'
                        df_derived[growth_col] = df_sorted.groupby(level=0)[metric_col].pct_change() * 100
                        
                        derived_features.append({
                            'base_columns': [metric_col, year_col],
                            'derived_column': growth_col,
                            'transformation_type': 'growth_rate',
                            'description': f'Year-over-year growth rate for {metric_col}'
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to create growth rate for {metric_col}: {str(e)}")
        
        return df_derived, derived_features

    async def _create_aggregation_features(self, df: pd.DataFrame, transform_logger: TransformLogger) -> Tuple[pd.DataFrame, List[Dict]]:
        """Create aggregation features grouped by categorical columns"""
        self.logger.info("Creating aggregation features")
        
        df_agg = df.copy()
        aggregation_features = []
        
        # Find categorical columns for grouping
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Limit to prevent explosion of features
        categorical_cols = categorical_cols[:2]  # Max 2 grouping columns
        numeric_cols = numeric_cols[:3]  # Max 3 numeric columns
        
        for group_col in categorical_cols:
            if df[group_col].nunique() < 50:  # Don't group by high-cardinality columns
                for metric_col in numeric_cols:
                    try:
                        # Create group-wise statistics
                        group_stats = df.groupby(group_col)[metric_col].agg(['mean', 'sum', 'count']).reset_index()
                        
                        # Merge back to original dataframe
                        merge_cols = [f'{metric_col}_{group_col}_mean', f'{metric_col}_{group_col}_sum', f'{metric_col}_{group_col}_count']
                        group_stats.columns = [group_col] + merge_cols
                        
                        df_agg = df_agg.merge(group_stats, on=group_col, how='left')
                        
                        transform_logger.log_transform(
                            agent="transformation",
                            action="create_group_aggregations",
                            column=f"{metric_col}_grouped_by_{group_col}",
                            rows_affected=len(df_agg),
                            rule_id="group_agg_v1",
                            rationale=f"Created group statistics for {metric_col} by {group_col}",
                            confidence="medium"
                        )
                        
                        aggregation_features.append({
                            'base_columns': [metric_col, group_col],
                            'derived_columns': merge_cols,
                            'transformation_type': 'group_aggregation',
                            'description': f'Group statistics for {metric_col} by {group_col}'
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to create aggregation for {metric_col} by {group_col}: {str(e)}")
        
        return df_agg, aggregation_features

    async def _create_categorical_features(self, df: pd.DataFrame, transform_logger: TransformLogger) -> Tuple[pd.DataFrame, List[Dict]]:
        """Create categorical features and buckets"""
        self.logger.info("Creating categorical features")
        
        df_cat = df.copy()
        categorical_features = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create quantile buckets for numeric columns
        for col in numeric_cols[:3]:  # Limit to first 3 columns
            try:
                if df[col].nunique() > 10:  # Only create buckets for columns with enough variance
                    # Create tertiles (3 buckets)
                    bucket_col = f'{col}_tertile'
                    df_cat[bucket_col] = pd.qcut(df_cat[col], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
                    
                    # Create quintiles (5 buckets)
                    quintile_col = f'{col}_quintile'
                    df_cat[quintile_col] = pd.qcut(df_cat[col], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
                    
                    transform_logger.log_transform(
                        agent="transformation",
                        action="create_quantile_buckets",
                        column=col,
                        rows_affected=len(df_cat),
                        rule_id="quantile_buckets_v1",
                        rationale=f"Created tertile and quintile buckets for {col}",
                        confidence="medium"
                    )
                    
                    categorical_features.append({
                        'base_column': col,
                        'derived_columns': [bucket_col, quintile_col],
                        'transformation_type': 'quantile_buckets',
                        'description': f'Quantile-based categorical buckets for {col}'
                    })
                    
            except Exception as e:
                self.logger.warning(f"Failed to create categorical features for {col}: {str(e)}")
        
        # Create binary flags for important thresholds
        for col in numeric_cols[:2]:
            try:
                if col not in df_cat.columns:
                    continue
                    
                # Create above/below median flag
                median_val = df_cat[col].median()
                flag_col = f'{col}_above_median'
                df_cat[flag_col] = (df_cat[col] > median_val).astype(int)
                
                categorical_features.append({
                    'base_column': col,
                    'derived_column': flag_col,
                    'transformation_type': 'binary_flag',
                    'description': f'Binary flag for {col} above median ({median_val:.2f})'
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to create binary flag for {col}: {str(e)}")
        
        return df_cat, categorical_features

    def _create_transformation_summary(self, original_df: pd.DataFrame, transformed_df: pd.DataFrame,
                                     transformation_log: List) -> Dict[str, Any]:
        """Create comprehensive transformation summary"""
        
        # Count transformations by type
        transformation_counts = {}
        derived_columns = []
        
        for entry in transformation_log:
            trans_type = entry.get('transformation_type', 'unknown')
            transformation_counts[trans_type] = transformation_counts.get(trans_type, 0) + 1
            
            # Collect derived columns
            if 'derived_column' in entry:
                derived_columns.append(entry['derived_column'])
            elif 'derived_columns' in entry:
                derived_columns.extend(entry['derived_columns'])
        
        summary = {
            "transformation_timestamp": datetime.utcnow().isoformat(),
            "original_shape": {
                "rows": len(original_df),
                "columns": len(original_df.columns)
            },
            "transformed_shape": {
                "rows": len(transformed_df),
                "columns": len(transformed_df.columns)
            },
            "features_added": {
                "total_new_columns": len(transformed_df.columns) - len(original_df.columns),
                "derived_columns": derived_columns,
                "transformation_types": transformation_counts
            },
            "transformation_summary": {
                "total_transformations": len(transformation_log),
                "time_features": transformation_counts.get('time_decomposition', 0),
                "per_capita_features": transformation_counts.get('per_capita', 0),
                "aggregation_features": transformation_counts.get('group_aggregation', 0),
                "categorical_features": transformation_counts.get('quantile_buckets', 0) + transformation_counts.get('binary_flag', 0)
            },
            "feature_catalog": transformation_log,
            "recommendations": self._generate_transformation_recommendations(transformation_log, transformed_df)
        }
        
        return summary

    def _generate_transformation_recommendations(self, transformation_log: List, df: pd.DataFrame) -> List[str]:
        """Generate recommendations for further transformations"""
        
        recommendations = []
        
        # Check if we have time features
        time_features = [entry for entry in transformation_log if entry.get('transformation_type') == 'time_decomposition']
        if not time_features:
            recommendations.append("Consider adding time-based features if temporal analysis is needed")
        
        # Check for normalization needs
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        high_variance_cols = []
        for col in numeric_cols:
            if df[col].std() > 1000:  # High variance indicator
                high_variance_cols.append(col)
        
        if high_variance_cols:
            recommendations.append(f"Consider normalizing {len(high_variance_cols)} columns with high variance")
        
        # Check for feature scaling
        if len(numeric_cols) > 5:
            recommendations.append("Consider feature scaling for machine learning applications")
        
        return recommendations

    async def generate_preview(self, state) -> pd.DataFrame:
        """Generate transformation preview without applying changes"""
        self.logger.info("Generating transformation preview")
        
        cleaned_data = getattr(state, 'cleaned_data', state.raw_data)
        
        # Preview what transformations would be applied
        preview_operations = []
        
        # Time features preview
        date_columns = [col for col in cleaned_data.columns if 'date' in col.lower()]
        for col in date_columns:
            preview_operations.append({
                'base_column': col,
                'transformation': 'TIME_DECOMPOSITION',
                'new_columns': f'{col}_year, {col}_month, {col}_quarter, {col}_day_of_week',
                'description': 'Extract temporal components'
            })
        
        # Per-capita features preview
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
        pop_cols = [col for col in cleaned_data.columns if 'population' in col.lower()]
        
        if pop_cols and numeric_cols:
            preview_operations.append({
                'base_column': f'{numeric_cols[0]} / {pop_cols[0]}',
                'transformation': 'PER_CAPITA',
                'new_columns': f'{numeric_cols[0]}_per_1000',
                'description': 'Create per-capita metrics'
            })
        
        # Categorical features preview
        for col in numeric_cols[:2]:
            preview_operations.append({
                'base_column': col,
                'transformation': 'QUANTILE_BUCKETS',
                'new_columns': f'{col}_tertile, {col}_quintile',
                'description': 'Create categorical buckets'
            })
        
        return pd.DataFrame(preview_operations)
# Cleaning Agent (3.4)
"""
RTGS AI Analyst - Cleaning Agent
Handles missing data imputation, duplicate removal, outlier detection, and data quality validation
"""

import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import warnings
from scipy import stats

from src.utils.logging import get_agent_logger, TransformLogger
from src.utils.data_helpers import detect_outliers_iqr

warnings.filterwarnings('ignore')


class CleaningAgent:
    """Agent responsible for data cleaning and quality validation"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = get_agent_logger("cleaning")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract thresholds from config
        self.thresholds = self.config['data_quality']
        
    async def process(self, state) -> Any:
        """Main cleaning processing pipeline"""
        self.logger.info("Starting data cleaning process")
        
        try:
            # Initialize transform logger
            transform_logger = TransformLogger(
                log_file=Path(state.run_manifest['artifacts_paths']['logs_dir']) / "transform_log.jsonl",
                run_id=state.run_manifest['run_id']
            )
            
            # Get standardized data
            standardized_data = getattr(state, 'standardized_data', state.raw_data)
            if standardized_data is None:
                raise ValueError("No standardized data available for cleaning")
            
            # Create working copy
            cleaned_data = standardized_data.copy()
            
            # Track cleaning operations
            cleaning_operations = {
                'columns_dropped': [],
                'missing_value_operations': [],
                'duplicate_operations': [],
                'outlier_operations': [],
                'quality_flags': []
            }
            
            # Step 1: Handle missing values
            cleaned_data, missing_ops = await self._handle_missing_values(cleaned_data, transform_logger)
            cleaning_operations['missing_value_operations'] = missing_ops
            
            # Step 2: Remove duplicates
            cleaned_data, duplicate_ops = await self._handle_duplicates(cleaned_data, transform_logger)
            cleaning_operations['duplicate_operations'] = duplicate_ops
            
            # Step 3: Detect and handle outliers
            cleaned_data, outlier_ops = await self._handle_outliers(cleaned_data, transform_logger)
            cleaning_operations['outlier_operations'] = outlier_ops
            
            # Step 4: Quality validation
            quality_report = await self._validate_data_quality(cleaned_data, standardized_data)
            cleaning_operations['quality_flags'] = quality_report['quality_flags']
            
            # Create comprehensive cleaning summary
            cleaning_summary = self._create_cleaning_summary(
                standardized_data, cleaned_data, cleaning_operations, quality_report
            )
            
            # Generate transforms preview (before/after examples)
            transforms_preview = self._generate_transforms_preview(
                standardized_data, cleaned_data, cleaning_operations
            )
            
            # Save cleaned data
            cleaned_path = Path(state.run_manifest['run_config']['output_dir']) / "data" / "cleaned" / f"{state.run_manifest['dataset_info']['dataset_name']}_cleaned.csv"
            cleaned_data.to_csv(cleaned_path, index=False)
            
            # Save cleaning summary
            summary_path = Path(state.run_manifest['artifacts_paths']['docs_dir']) / "cleaning_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(cleaning_summary, f, indent=2)
            
            # Save transforms preview
            preview_path = Path(state.run_manifest['artifacts_paths']['docs_dir']) / "transforms_preview.csv"
            transforms_preview.to_csv(preview_path, index=False)
            
            # Update state
            state.cleaned_data = cleaned_data
            state.cleaning_summary = cleaning_summary
            state.cleaning_operations = cleaning_operations
            state.data_quality_report = quality_report
            state.cleaned_path = str(cleaned_path)
            
            self.logger.info(f"Cleaning completed: {len(cleaned_data)} rows, {len(cleaned_data.columns)} columns")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Data cleaning failed: {str(e)}")
            state.errors.append(f"Data cleaning failed: {str(e)}")
            return state

    async def _handle_missing_values(self, df: pd.DataFrame, transform_logger: TransformLogger) -> Tuple[pd.DataFrame, List[Dict]]:
        """Handle missing values using various imputation strategies"""
        self.logger.info("Handling missing values")
        
        df_cleaned = df.copy()
        missing_operations = []
        
        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_fraction = null_count / len(df)
            
            if null_count == 0:
                continue  # No missing values
            
            operation = None
            
            # Decision logic based on thresholds
            if null_fraction > self.thresholds['drop_column_threshold']:
                # Drop column if too many missing values
                df_cleaned = df_cleaned.drop(columns=[col])
                operation = {
                    'action': 'drop_column',
                    'column': col,
                    'null_fraction': null_fraction,
                    'rationale': f'Column has {null_fraction:.1%} missing values (>{self.thresholds["drop_column_threshold"]:.1%} threshold)'
                }
                
                transform_logger.log_transform(
                    agent="cleaning",
                    action="drop_column",
                    column=col,
                    rows_affected=len(df),
                    rule_id="drop_high_missing_col_v1",
                    rationale=operation['rationale'],
                    parameters={'null_fraction': null_fraction, 'threshold': self.thresholds['drop_column_threshold']},
                    confidence="high"
                )
                
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Handle numeric columns
                if null_fraction < self.thresholds['num_median_impute_threshold']:
                    # Simple median imputation for low missing
                    median_value = df[col].median()
                    df_cleaned[col] = df_cleaned[col].fillna(median_value)
                    
                    operation = {
                        'action': 'median_impute',
                        'column': col,
                        'null_fraction': null_fraction,
                        'impute_value': median_value,
                        'rationale': f'Median imputation for {null_fraction:.1%} missing values'
                    }
                    
                    transform_logger.log_transform(
                        agent="cleaning",
                        action="median_impute",
                        column=col,
                        rows_affected=null_count,
                        rule_id="median_impute_v1",
                        rationale=operation['rationale'],
                        parameters={'impute_value': median_value, 'null_fraction': null_fraction},
                        confidence="high",
                        preview_before="NaN",
                        preview_after=str(median_value)
                    )
                    
                elif null_fraction < self.thresholds['group_impute_threshold']:
                    # Group-wise imputation if geographic columns exist
                    imputed = await self._group_wise_imputation(df_cleaned, col, transform_logger)
                    if imputed:
                        operation = {
                            'action': 'group_impute',
                            'column': col,
                            'null_fraction': null_fraction,
                            'rationale': f'Group-wise imputation for {null_fraction:.1%} missing values'
                        }
                    else:
                        # Fallback to median
                        median_value = df[col].median()
                        df_cleaned[col] = df_cleaned[col].fillna(median_value)
                        operation = {
                            'action': 'median_impute_fallback',
                            'column': col,
                            'null_fraction': null_fraction,
                            'impute_value': median_value,
                            'rationale': f'Median imputation (group imputation failed) for {null_fraction:.1%} missing'
                        }
                else:
                    # Flag high missing for manual review
                    operation = {
                        'action': 'flag_high_missing',
                        'column': col,
                        'null_fraction': null_fraction,
                        'rationale': f'High missing values ({null_fraction:.1%}) - flagged for review'
                    }
                    
            else:
                # Handle categorical/text columns
                if null_fraction < self.thresholds['group_impute_threshold']:
                    # Mode imputation or custom categorical handling
                    if df[col].dtype.name == 'category' or df[col].nunique() < 20:
                        # Use mode for categorical
                        mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else "UNKNOWN"
                        df_cleaned[col] = df_cleaned[col].fillna(mode_value)
                        
                        operation = {
                            'action': 'mode_impute',
                            'column': col,
                            'null_fraction': null_fraction,
                            'impute_value': mode_value,
                            'rationale': f'Mode imputation for categorical column'
                        }
                        
                        transform_logger.log_transform(
                            agent="cleaning",
                            action="mode_impute",
                            column=col,
                            rows_affected=null_count,
                            rule_id="mode_impute_v1",
                            rationale=operation['rationale'],
                            parameters={'impute_value': mode_value, 'null_fraction': null_fraction},
                            confidence="medium",
                            preview_before="NaN",
                            preview_after=str(mode_value)
                        )
                    else:
                        # Use "MISSING" placeholder for text
                        df_cleaned[col] = df_cleaned[col].fillna("MISSING")
                        
                        operation = {
                            'action': 'missing_placeholder',
                            'column': col,
                            'null_fraction': null_fraction,
                            'impute_value': "MISSING",
                            'rationale': f'Missing placeholder for text column'
                        }
                        
                        transform_logger.log_transform(
                            agent="cleaning",
                            action="missing_placeholder",
                            column=col,
                            rows_affected=null_count,
                            rule_id="missing_placeholder_v1",
                            rationale=operation['rationale'],
                            confidence="medium"
                        )
                else:
                    # Flag high missing for manual review
                    operation = {
                        'action': 'flag_high_missing',
                        'column': col,
                        'null_fraction': null_fraction,
                        'rationale': f'High missing values ({null_fraction:.1%}) in categorical column'
                    }
            
            if operation:
                missing_operations.append(operation)
        
        self.logger.info(f"Processed missing values: {len(missing_operations)} operations")
        return df_cleaned, missing_operations

    async def _group_wise_imputation(self, df: pd.DataFrame, col: str, transform_logger: TransformLogger) -> bool:
        """Attempt group-wise imputation using geographic or categorical grouping"""
        
        # Look for potential grouping columns (geographic, categorical)
        grouping_candidates = []
        
        for group_col in df.columns:
            if (group_col != col and 
                not pd.api.types.is_numeric_dtype(df[group_col]) and
                df[group_col].nunique() < len(df) * 0.5):  # Not too many unique values
                grouping_candidates.append(group_col)
        
        if not grouping_candidates:
            return False
        
        # Try group-wise median imputation with the first suitable grouping column
        group_col = grouping_candidates[0]
        
        try:
            # Calculate group medians
            group_medians = df.groupby(group_col)[col].median()
            
            # Fill missing values with group medians
            def fill_group_median(row):
                if pd.isnull(row[col]) and row[group_col] in group_medians:
                    return group_medians[row[group_col]]
                return row[col]
            
            df[col] = df.apply(fill_group_median, axis=1)
            
            transform_logger.log_transform(
                agent="cleaning",
                action="group_wise_impute",
                column=col,
                rows_affected=df[col].isnull().sum(),
                rule_id="group_impute_v1",
                rationale=f"Group-wise median imputation using {group_col}",
                parameters={'grouping_column': group_col},
                confidence="medium"
            )
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Group-wise imputation failed for {col}: {str(e)}")
            return False

    async def _handle_duplicates(self, df: pd.DataFrame, transform_logger: TransformLogger) -> Tuple[pd.DataFrame, List[Dict]]:
        """Handle duplicate rows"""
        self.logger.info("Handling duplicate rows")
        
        original_count = len(df)
        
        # Remove exact duplicates
        df_no_dupes = df.drop_duplicates()
        exact_dupes_removed = original_count - len(df_no_dupes)
        
        operations = []
        
        if exact_dupes_removed > 0:
            transform_logger.log_transform(
                agent="cleaning",
                action="remove_exact_duplicates",
                rows_affected=exact_dupes_removed,
                rule_id="remove_duplicates_v1",
                rationale=f"Removed {exact_dupes_removed} exact duplicate rows",
                parameters={'duplicates_removed': exact_dupes_removed},
                confidence="high"
            )
            
            operations.append({
                'action': 'remove_exact_duplicates',
                'rows_removed': exact_dupes_removed,
                'rationale': f'Removed {exact_dupes_removed} exact duplicate rows'
            })
        
        # Check for potential fuzzy duplicates (optional advanced feature)
        # This would require more sophisticated matching logic
        
        self.logger.info(f"Duplicate handling completed: removed {exact_dupes_removed} exact duplicates")
        
        return df_no_dupes, operations

    async def _handle_outliers(self, df: pd.DataFrame, transform_logger: TransformLogger) -> Tuple[pd.DataFrame, List[Dict]]:
        """Detect and handle outliers in numeric columns"""
        self.logger.info("Detecting and handling outliers")
        
        df_outliers_handled = df.copy()
        outlier_operations = []
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df[col].isnull().all():
                continue
            
            # Detect outliers using IQR method
            outliers_mask, outlier_stats = detect_outliers_iqr(
                df[col], 
                multiplier=self.thresholds['outlier_iqr_multiplier']
            )
            
            outlier_count = outliers_mask.sum() if isinstance(outliers_mask, pd.Series) else 0
            outlier_percentage = (outlier_count / len(df)) * 100
            
            if outlier_count > 0:
                # Decision: flag vs remove based on percentage and domain knowledge
                if outlier_percentage > self.thresholds['max_auto_drop_rows_percent'] * 100:
                    # Too many outliers - just flag them
                    df_outliers_handled[f'{col}_outlier_flag'] = outliers_mask
                    
                    transform_logger.log_transform(
                        agent="cleaning",
                        action="flag_outliers",
                        column=col,
                        rows_affected=outlier_count,
                        rule_id="flag_outliers_v1",
                        rationale=f"Flagged {outlier_count} outliers ({outlier_percentage:.1f}%) - too many to auto-remove",
                        parameters={
                            'outlier_count': int(outlier_count),
                            'outlier_percentage': outlier_percentage,
                            'iqr_multiplier': self.thresholds['outlier_iqr_multiplier'],
                            **outlier_stats
                        },
                        confidence="medium"
                    )
                    
                    outlier_operations.append({
                        'action': 'flag_outliers',
                        'column': col,
                        'outlier_count': int(outlier_count),
                        'outlier_percentage': outlier_percentage,
                        'outlier_bounds': {
                            'lower': outlier_stats['lower_bound'],
                            'upper': outlier_stats['upper_bound']
                        },
                        'rationale': f'Flagged {outlier_count} outliers for review'
                    })
                else:
                    # Remove outliers if count is reasonable
                    df_outliers_handled = df_outliers_handled[~outliers_mask]
                    
                    transform_logger.log_transform(
                        agent="cleaning",
                        action="remove_outliers",
                        column=col,
                        rows_affected=outlier_count,
                        rule_id="remove_outliers_v1", 
                        rationale=f"Removed {outlier_count} outliers ({outlier_percentage:.1f}%)",
                        parameters={
                            'outlier_count': int(outlier_count),
                            'outlier_percentage': outlier_percentage,
                            'iqr_multiplier': self.thresholds['outlier_iqr_multiplier'],
                            **outlier_stats
                        },
                        confidence="medium"
                    )
                    
                    outlier_operations.append({
                        'action': 'remove_outliers',
                        'column': col,
                        'outlier_count': int(outlier_count),
                        'outlier_percentage': outlier_percentage,
                        'outlier_bounds': {
                            'lower': outlier_stats['lower_bound'],
                            'upper': outlier_stats['upper_bound']
                        },
                        'rationale': f'Removed {outlier_count} outlier rows'
                    })
        
        self.logger.info(f"Outlier handling completed: processed {len(outlier_operations)} columns")
        
        return df_outliers_handled, outlier_operations

    async def _validate_data_quality(self, cleaned_df: pd.DataFrame, original_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and generate quality report"""
        self.logger.info("Validating data quality")
        
        # Calculate quality metrics
        quality_metrics = {
            'completeness': {
                'total_cells': len(cleaned_df) * len(cleaned_df.columns),
                'missing_cells': cleaned_df.isnull().sum().sum(),
                'completeness_rate': 1 - (cleaned_df.isnull().sum().sum() / (len(cleaned_df) * len(cleaned_df.columns)))
            },
            'consistency': {
                'columns_preserved': len(cleaned_df.columns),
                'rows_preserved': len(cleaned_df),
                'data_preservation_rate': len(cleaned_df) / len(original_df) if len(original_df) > 0 else 0
            }
        }
        
        # Quality flags
        quality_flags = []
        
        # Check for columns with high missing values
        high_missing_columns = []
        for col in cleaned_df.columns:
            null_fraction = cleaned_df[col].isnull().sum() / len(cleaned_df)
            if null_fraction > self.thresholds['high_missingness_threshold']:
                high_missing_columns.append({
                    'column': col,
                    'missing_fraction': null_fraction
                })
        
        if high_missing_columns:
            quality_flags.append({
                'type': 'high_missingness',
                'severity': 'warning',
                'description': f'{len(high_missing_columns)} columns still have high missing values',
                'details': high_missing_columns
            })
        
        # Check for excessive data loss
        data_loss_rate = 1 - (len(cleaned_df) / len(original_df))
        if data_loss_rate > 0.1:  # More than 10% data loss
            quality_flags.append({
                'type': 'excessive_data_loss',
                'severity': 'warning',
                'description': f'{data_loss_rate:.1%} of rows were removed during cleaning',
                'details': {'data_loss_rate': data_loss_rate}
            })
        
        # Check for low variance columns (potential data quality issues)
        low_variance_columns = []
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[col].var() == 0:  # No variance
                low_variance_columns.append(col)
        
        if low_variance_columns:
            quality_flags.append({
                'type': 'low_variance_columns',
                'severity': 'info',
                'description': f'{len(low_variance_columns)} columns have zero variance',
                'details': {'columns': low_variance_columns}
            })
        
        quality_report = {
            'validation_timestamp': datetime.utcnow().isoformat(),
            'quality_metrics': quality_metrics,
            'quality_flags': quality_flags,
            'overall_quality_score': self._calculate_quality_score(quality_metrics, quality_flags),
            'recommendations': self._generate_quality_recommendations(quality_flags, quality_metrics)
        }
        
        return quality_report

    def _calculate_quality_score(self, metrics: Dict, flags: List) -> int:
        """Calculate overall quality score (0-100)"""
        
        base_score = 100
        
        # Deduct for completeness issues
        completeness_rate = metrics['completeness']['completeness_rate']
        base_score -= (1 - completeness_rate) * 30  # Up to 30 points for missing data
        
        # Deduct for data preservation issues
        preservation_rate = metrics['consistency']['data_preservation_rate']
        base_score -= (1 - preservation_rate) * 20  # Up to 20 points for data loss
        
        # Deduct for quality flags
        for flag in flags:
            if flag['severity'] == 'warning':
                base_score -= 10
            elif flag['severity'] == 'error':
                base_score -= 20
            elif flag['severity'] == 'info':
                base_score -= 2
        
        return max(0, min(100, int(base_score)))

    def _generate_quality_recommendations(self, flags: List, metrics: Dict) -> List[str]:
        """Generate actionable quality recommendations"""
        
        recommendations = []
        
        for flag in flags:
            if flag['type'] == 'high_missingness':
                recommendations.append(
                    f"Consider additional data sources or imputation strategies for {len(flag['details'])} columns with high missing values"
                )
            elif flag['type'] == 'excessive_data_loss':
                recommendations.append(
                    "Review cleaning parameters - significant data loss may impact analysis validity"
                )
            elif flag['type'] == 'low_variance_columns':
                recommendations.append(
                    f"Review {len(flag['details']['columns'])} constant-value columns for relevance to analysis"
                )
        
        # General recommendations based on metrics
        completeness_rate = metrics['completeness']['completeness_rate']
        if completeness_rate < 0.9:
            recommendations.append(
                f"Overall completeness is {completeness_rate:.1%} - consider data collection improvements"
            )
        
        return recommendations

    def _create_cleaning_summary(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame,
                                operations: Dict, quality_report: Dict) -> Dict[str, Any]:
        """Create comprehensive cleaning summary"""
        
        summary = {
            "cleaning_timestamp": datetime.utcnow().isoformat(),
            "original_shape": {
                "rows": len(original_df),
                "columns": len(original_df.columns)
            },
            "cleaned_shape": {
                "rows": len(cleaned_df),
                "columns": len(cleaned_df.columns)
            },
            "data_changes": {
                "rows_removed": len(original_df) - len(cleaned_df),
                "columns_removed": len(original_df.columns) - len(cleaned_df.columns),
                "data_preservation_rate": len(cleaned_df) / len(original_df) if len(original_df) > 0 else 0
            },
            "operations_summary": {
                "missing_value_operations": len(operations['missing_value_operations']),
                "duplicate_operations": len(operations['duplicate_operations']),
                "outlier_operations": len(operations['outlier_operations']),
                "total_operations": (len(operations['missing_value_operations']) + 
                                   len(operations['duplicate_operations']) + 
                                   len(operations['outlier_operations']))
            },
            "detailed_operations": operations,
            "quality_assessment": quality_report,
            "missing_values_by_column": {
                col: {
                    "missing_count": int(cleaned_df[col].isnull().sum()),
                    "missing_percentage": float(cleaned_df[col].isnull().sum() / len(cleaned_df) * 100)
                }
                for col in cleaned_df.columns
                if cleaned_df[col].isnull().sum() > 0
            }
        }
        
        return summary

    def _generate_transforms_preview(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame,
                                   operations: Dict) -> pd.DataFrame:
        """Generate before/after preview of transformations"""
        
        # Create a sample of 5 rows to show transformations
        sample_size = min(5, len(original_df))
        sample_indices = original_df.head(sample_size).index
        
        preview_data = []
        
        # Show column-level changes
        for col in original_df.columns:
            if col in cleaned_df.columns:
                # Column preserved
                original_sample = original_df[col].head(sample_size).fillna("NULL").astype(str).tolist()
                cleaned_sample = cleaned_df[col].head(sample_size).fillna("NULL").astype(str).tolist()
                
                preview_data.append({
                    'column': col,
                    'transformation': 'preserved',
                    'before_sample': ', '.join(original_sample),
                    'after_sample': ', '.join(cleaned_sample),
                    'change_description': 'Column preserved with potential value changes'
                })
            else:
                # Column dropped
                original_sample = original_df[col].head(sample_size).fillna("NULL").astype(str).tolist()
                
                preview_data.append({
                    'column': col,
                    'transformation': 'dropped',
                    'before_sample': ', '.join(original_sample),
                    'after_sample': 'COLUMN_DROPPED',
                    'change_description': 'Column removed due to quality issues'
                })
        
        # Add information about new columns (like outlier flags)
        for col in cleaned_df.columns:
            if col not in original_df.columns:
                cleaned_sample = cleaned_df[col].head(sample_size).fillna("NULL").astype(str).tolist()
                
                preview_data.append({
                    'column': col,
                    'transformation': 'added',
                    'before_sample': 'COLUMN_ADDED',
                    'after_sample': ', '.join(cleaned_sample),
                    'change_description': 'New column added during cleaning (e.g., outlier flag)'
                })
        
        return pd.DataFrame(preview_data)

    async def generate_preview(self, state) -> pd.DataFrame:
        """Generate preview without actually applying transformations"""
        self.logger.info("Generating cleaning preview")
        
        # This method is called from the orchestrator for preview mode
        standardized_data = getattr(state, 'standardized_data', state.raw_data)
        
        # Simulate cleaning operations to show what would happen
        preview_operations = []
        
        for col in standardized_data.columns:
            null_count = standardized_data[col].isnull().sum()
            null_fraction = null_count / len(standardized_data)
            
            if null_fraction > self.thresholds['drop_column_threshold']:
                preview_operations.append({
                    'column': col,
                    'operation': 'DROP_COLUMN',
                    'reason': f'{null_fraction:.1%} missing values',
                    'rows_affected': len(standardized_data)
                })
            elif null_count > 0:
                if pd.api.types.is_numeric_dtype(standardized_data[col]):
                    preview_operations.append({
                        'column': col,
                        'operation': 'MEDIAN_IMPUTE',
                        'reason': f'Fill {null_count} missing numeric values',
                        'rows_affected': null_count
                    })
                else:
                    preview_operations.append({
                        'column': col,
                        'operation': 'MODE_IMPUTE',
                        'reason': f'Fill {null_count} missing categorical values',
                        'rows_affected': null_count
                    })
        
        # Check for duplicates
        duplicate_count = standardized_data.duplicated().sum()
        if duplicate_count > 0:
            preview_operations.append({
                'column': 'ALL_COLUMNS',
                'operation': 'REMOVE_DUPLICATES',
                'reason': f'{duplicate_count} exact duplicate rows found',
                'rows_affected': duplicate_count
            })
        
        # Convert to DataFrame for preview
        preview_df = pd.DataFrame(preview_operations)
        
        return preview_df
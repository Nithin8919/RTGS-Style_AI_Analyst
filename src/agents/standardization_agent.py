# Standardization Agent (3.3)
"""
RTGS AI Analyst - Standardization Agent
Applies canonical naming and standardizes data types, units, and encodings
"""
import pandas as pd
import numpy as np
import json
import yaml
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import warnings

from src.utils.logging import get_agent_logger, TransformLogger
from src.utils.data_helpers import safe_type_conversion, standardize_column_names

warnings.filterwarnings('ignore')


class StandardizationAgent:
    """Agent responsible for data standardization and column normalization"""
    
    def __init__(self):
        self.logger = get_agent_logger("standardization")
        
        # Load configuration
        try:
            with open("config.yaml", 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning("Config file not found, using defaults")
            self.config = {}
    
    async def process(self, state) -> Any:
        """Main standardization processing pipeline"""
        self.logger.info("Starting data standardization process")
        
        try:
            # Initialize transform logger
            transform_logger = TransformLogger(
                log_file=Path(state.run_manifest['artifacts_paths']['logs_dir']) / "transform_log.jsonl",
                run_id=state.run_manifest['run_id']
            )
            
            # Get data and schema info
            raw_data = state.raw_data
            canonical_mapping = getattr(state, 'canonical_mapping', {})
            
            if raw_data is None:
                raise ValueError("No raw data available for standardization")
            
            # Create working copy
            standardized_data = raw_data.copy()
            
            # Apply standardization steps
            standardized_data, column_rename_log = await self._standardize_column_names(
                standardized_data, canonical_mapping, transform_logger
            )
            
            standardized_data, type_conversion_log = await self._standardize_data_types(
                standardized_data, canonical_mapping, transform_logger
            )
            
            standardized_data, unit_normalization_log = await self._normalize_units(
                standardized_data, transform_logger
            )
            
            standardized_data, encoding_standardization_log = await self._standardize_encodings(
                standardized_data, transform_logger
            )
            
            # Create standardization summary
            standardization_summary = self._create_standardization_summary(
                raw_data, standardized_data, canonical_mapping,
                column_rename_log, type_conversion_log, 
                unit_normalization_log, encoding_standardization_log
            )
            
            # Save standardized data
            standardized_path = Path(state.run_manifest['run_config']['output_dir']) / "data" / "standardized" / f"{state.run_manifest['dataset_info']['dataset_name']}_standardized.csv"
            standardized_data.to_csv(standardized_path, index=False)
            
            # Save standardization summary
            summary_path = Path(state.run_manifest['artifacts_paths']['docs_dir']) / "standardization_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(standardization_summary, f, indent=2)
            
            # Update state
            state.standardized_data = standardized_data
            state.standardization_summary = standardization_summary
            state.standardized_path = str(standardized_path)
            
            self.logger.info(f"Standardization completed: {len(standardized_data)} rows, {len(standardized_data.columns)} columns")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Standardization failed: {str(e)}")
            state.errors.append(f"Standardization failed: {str(e)}")
            return state

    async def _standardize_column_names(self, df: pd.DataFrame, canonical_mapping: Dict, transform_logger: TransformLogger) -> Tuple[pd.DataFrame, List[Dict]]:
        """Apply canonical column names"""
        self.logger.info("Standardizing column names")
        
        rename_mapping = {}
        rename_log = []
        
        for original_col in df.columns:
            if original_col in canonical_mapping:
                canonical_name = canonical_mapping[original_col]['canonical_name']
                confidence = canonical_mapping[original_col]['confidence']
                
                # Only rename if different and high enough confidence
                if (canonical_name != original_col and 
                    confidence > self.config['data_quality']['low_confidence_type_threshold']):
                    
                    # Ensure no naming conflicts
                    final_canonical_name = self._resolve_naming_conflicts(canonical_name, df.columns, rename_mapping)
                    
                    rename_mapping[original_col] = final_canonical_name
                    
                    # Log the rename operation
                    transform_logger.log_transform(
                        agent="standardization",
                        action="rename_column",
                        column=original_col,
                        rows_affected=len(df),
                        rule_id="column_rename_v1",
                        rationale=f"Renamed to canonical name: {final_canonical_name}",
                        parameters={
                            "original_name": original_col,
                            "canonical_name": final_canonical_name,
                            "confidence": confidence
                        },
                        confidence="high" if confidence > 0.8 else "medium",
                        preview_before=original_col,
                        preview_after=final_canonical_name
                    )
                    
                    rename_log.append({
                        "original": original_col,
                        "canonical": final_canonical_name,
                        "confidence": confidence,
                        "rationale": canonical_mapping[original_col].get('rationale', 'Schema inference')
                    })
        
        # Apply renames
        df_renamed = df.rename(columns=rename_mapping)
        
        self.logger.info(f"Renamed {len(rename_mapping)} columns")
        
        return df_renamed, rename_log

    def _resolve_naming_conflicts(self, canonical_name: str, existing_columns: List[str], current_mapping: Dict) -> str:
        """Resolve naming conflicts by adding suffixes"""
        
        # Check if name already exists in original columns or current mapping
        used_names = set(existing_columns) | set(current_mapping.values())
        
        if canonical_name not in used_names:
            return canonical_name
        
        # Add numeric suffix to resolve conflict
        counter = 1
        while f"{canonical_name}_{counter}" in used_names:
            counter += 1
        
        return f"{canonical_name}_{counter}"

    async def _standardize_data_types(self, df: pd.DataFrame, canonical_mapping: Dict, transform_logger: TransformLogger) -> Tuple[pd.DataFrame, List[Dict]]:
        """Standardize data types based on schema inference"""
        self.logger.info("Standardizing data types")
        
        type_mapping = {}
        conversion_log = []
        
        # Build type mapping from canonical mapping
        for col in df.columns:
            # Find original column name for this (possibly renamed) column
            original_col = None
            for orig, canon_info in canonical_mapping.items():
                if canon_info['canonical_name'] == col or orig == col:
                    original_col = orig
                    break
            
            if original_col and original_col in canonical_mapping:
                suggested_type = canonical_mapping[original_col]['suggested_type']
                confidence = canonical_mapping[original_col]['confidence']
                
                # Only convert if high confidence and type is different
                current_type = str(df[col].dtype)
                if (suggested_type != current_type and 
                    confidence > self.config['data_quality']['low_confidence_type_threshold']):
                    
                    type_mapping[col] = suggested_type
        
        # Apply type conversions safely
        df_converted, conversion_errors = safe_type_conversion(df, type_mapping)
        
        # Log type conversions
        for col, target_type in type_mapping.items():
            success = col not in [err.split()[3] for err in conversion_errors if 'convert' in err]
            
            if success:
                # Find original column info
                original_col = None
                for orig, canon_info in canonical_mapping.items():
                    if canon_info['canonical_name'] == col or orig == col:
                        original_col = orig
                        break
                
                original_type = str(df[col].dtype)
                final_type = str(df_converted[col].dtype)
                
                transform_logger.log_transform(
                    agent="standardization",
                    action="convert_data_type",
                    column=col,
                    rows_affected=len(df),
                    rule_id="type_conversion_v1",
                    rationale=f"Converted from {original_type} to {final_type}",
                    parameters={
                        "original_type": original_type,
                        "target_type": target_type,
                        "final_type": final_type
                    },
                    confidence="high",
                    preview_before=str(df[col].dtype),
                    preview_after=str(df_converted[col].dtype)
                )
                
                conversion_log.append({
                    "column": col,
                    "original_type": original_type,
                    "target_type": target_type,
                    "final_type": final_type,
                    "success": True
                })
        
        # Log conversion errors
        for error in conversion_errors:
            self.logger.warning(f"Type conversion error: {error}")
            # Extract column name from error message (basic parsing)
            if "convert" in error:
                col_name = error.split()[3] if len(error.split()) > 3 else "unknown"
                conversion_log.append({
                    "column": col_name,
                    "success": False,
                    "error": error
                })
        
        self.logger.info(f"Converted types for {len(type_mapping)} columns, {len(conversion_errors)} errors")
        
        return df_converted, conversion_log

    async def _normalize_units(self, df: pd.DataFrame, transform_logger: TransformLogger) -> Tuple[pd.DataFrame, List[Dict]]:
        """Normalize units and handle common unit conversions"""
        self.logger.info("Normalizing units")
        
        df_normalized = df.copy()
        unit_log = []
        
        # Unit normalization patterns
        unit_patterns = [
            # Currency patterns
            {
                'pattern': r'.*(?:rs|rupees|inr).*',
                'action': 'currency_normalization',
                'description': 'Standardize currency columns'
            },
            # Percentage patterns  
            {
                'pattern': r'.*(?:percent|%|pct).*',
                'action': 'percentage_normalization',
                'description': 'Convert percentages to decimal'
            },
            # Date patterns
            {
                'pattern': r'.*(?:date|time).*',
                'action': 'date_normalization', 
                'description': 'Standardize date formats'
            }
        ]
        
        for col in df_normalized.columns:
            col_lower = col.lower()
            
            # Apply unit normalizations
            for pattern_info in unit_patterns:
                if re.match(pattern_info['pattern'], col_lower):
                    
                    if pattern_info['action'] == 'currency_normalization':
                        df_normalized, normalized = self._normalize_currency_column(df_normalized, col)
                        if normalized:
                            transform_logger.log_transform(
                                agent="standardization",
                                action="normalize_currency",
                                column=col,
                                rows_affected=len(df_normalized),
                                rule_id="currency_norm_v1",
                                rationale="Removed currency symbols and standardized format",
                                confidence="high"
                            )
                            unit_log.append({
                                "column": col,
                                "action": "currency_normalization",
                                "description": "Removed currency symbols and standardized to numeric"
                            })
                    
                    elif pattern_info['action'] == 'percentage_normalization':
                        df_normalized, normalized = self._normalize_percentage_column(df_normalized, col)
                        if normalized:
                            transform_logger.log_transform(
                                agent="standardization", 
                                action="normalize_percentage",
                                column=col,
                                rows_affected=len(df_normalized),
                                rule_id="percentage_norm_v1",
                                rationale="Converted percentage strings to decimal values",
                                confidence="high"
                            )
                            unit_log.append({
                                "column": col,
                                "action": "percentage_normalization", 
                                "description": "Converted percentages to decimal format"
                            })
                    
                    elif pattern_info['action'] == 'date_normalization':
                        df_normalized, normalized = self._normalize_date_column(df_normalized, col)
                        if normalized:
                            transform_logger.log_transform(
                                agent="standardization",
                                action="normalize_dates",
                                column=col, 
                                rows_affected=len(df_normalized),
                                rule_id="date_norm_v1",
                                rationale="Standardized date format to ISO format",
                                confidence="high"
                            )
                            unit_log.append({
                                "column": col,
                                "action": "date_normalization",
                                "description": "Standardized to ISO date format"
                            })
        
        self.logger.info(f"Applied unit normalization to {len(unit_log)} columns")
        
        return df_normalized, unit_log

    def _normalize_currency_column(self, df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, bool]:
        """Normalize currency values by removing symbols and converting to numeric"""
        try:
            if df[col].dtype == 'object':
                # Remove common currency symbols
                df[col] = df[col].astype(str).str.replace(r'[₹$,\s]', '', regex=True)
                df[col] = df[col].str.replace(r'[^\d.]', '', regex=True)
                
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                return df, True
        except Exception as e:
            self.logger.warning(f"Failed to normalize currency column {col}: {str(e)}")
        
        return df, False

    def _normalize_percentage_column(self, df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, bool]:
        """Normalize percentage values to decimal format"""
        try:
            if df[col].dtype == 'object':
                # Remove % symbol and convert to decimal
                df[col] = df[col].astype(str).str.replace('%', '')
                df[col] = pd.to_numeric(df[col], errors='coerce') / 100
                return df, True
            elif pd.api.types.is_numeric_dtype(df[col]):
                # If numeric and values > 1, assume they're percentages that need conversion
                if df[col].max() > 1 and df[col].max() <= 100:
                    df[col] = df[col] / 100
                    return df, True
        except Exception as e:
            self.logger.warning(f"Failed to normalize percentage column {col}: {str(e)}")
        
        return df, False

    def _normalize_date_column(self, df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, bool]:
        """Normalize date formats to ISO standard"""
        try:
            if df[col].dtype == 'object' or 'datetime' not in str(df[col].dtype):
                # Try to parse as datetime
                df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                return df, True
        except Exception as e:
            self.logger.warning(f"Failed to normalize date column {col}: {str(e)}")
        
        return df, False

    async def _standardize_encodings(self, df: pd.DataFrame, transform_logger: TransformLogger) -> Tuple[pd.DataFrame, List[Dict]]:
        """Standardize text encodings and common values"""
        self.logger.info("Standardizing encodings")
        
        df_encoded = df.copy()
        encoding_log = []
        
        # Text standardization patterns
        text_standardizations = [
            # Boolean standardizations
            {
                'pattern': r'(?i)^(yes|y|true|1|on|enabled)$',
                'replacement': True,
                'type': 'boolean'
            },
            {
                'pattern': r'(?i)^(no|n|false|0|off|disabled)$', 
                'replacement': False,
                'type': 'boolean'
            },
            # Common text cleanups
            {
                'pattern': r'^\s+|\s+$',  # Leading/trailing whitespace
                'replacement': '',
                'type': 'whitespace'
            },
            {
                'pattern': r'\s+',  # Multiple spaces
                'replacement': ' ',
                'type': 'whitespace'
            }
        ]
        
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object':
                original_values = df_encoded[col].copy()
                
                # Apply text standardizations
                for std in text_standardizations:
                    if std['type'] == 'boolean':
                        # Check if column has boolean-like values
                        sample_values = df_encoded[col].dropna().astype(str).str.lower().unique()
                        boolean_pattern = r'(?i)^(yes|y|true|1|on|enabled|no|n|false|0|off|disabled)$'
                        
                        if any(re.match(boolean_pattern, val) for val in sample_values):
                            # Convert boolean-like strings
                            df_encoded[col] = df_encoded[col].astype(str).str.replace(
                                std['pattern'], str(std['replacement']), regex=True
                            )
                    
                    elif std['type'] == 'whitespace':
                        # Clean whitespace
                        df_encoded[col] = df_encoded[col].astype(str).str.replace(
                            std['pattern'], std['replacement'], regex=True
                        )
                
                # Check if any changes were made
                if not original_values.equals(df_encoded[col]):
                    transform_logger.log_transform(
                        agent="standardization",
                        action="standardize_encoding",
                        column=col,
                        rows_affected=len(df_encoded),
                        rule_id="encoding_std_v1",
                        rationale="Standardized text encodings and boolean values",
                        confidence="high"
                    )
                    
                    encoding_log.append({
                        "column": col,
                        "action": "text_standardization",
                        "description": "Cleaned whitespace and standardized boolean values"
                    })
        
        self.logger.info(f"Applied encoding standardization to {len(encoding_log)} columns")
        
        return df_encoded, encoding_log

    def _create_standardization_summary(self, original_df: pd.DataFrame, standardized_df: pd.DataFrame,
                                      canonical_mapping: Dict, column_rename_log: List,
                                      type_conversion_log: List, unit_normalization_log: List,
                                      encoding_standardization_log: List) -> Dict[str, Any]:
        """Create comprehensive standardization summary"""
        
        summary = {
            "standardization_timestamp": datetime.utcnow().isoformat(),
            "original_shape": {
                "rows": len(original_df),
                "columns": len(original_df.columns)
            },
            "standardized_shape": {
                "rows": len(standardized_df),
                "columns": len(standardized_df.columns)
            },
            
            "transformations_applied": {
                "column_renames": len(column_rename_log),
                "type_conversions": len([log for log in type_conversion_log if log.get('success', False)]),
                "unit_normalizations": len(unit_normalization_log),
                "encoding_standardizations": len(encoding_standardization_log)
            },
            
            "column_mapping": {
                "renames": {log['original']: log['canonical'] for log in column_rename_log},
                "type_changes": {
                    log['column']: {
                        'from': log['original_type'],
                        'to': log['final_type']
                    }
                    for log in type_conversion_log if log.get('success', False)
                }
            },
            
            "quality_metrics": {
                "successful_type_conversions": len([log for log in type_conversion_log if log.get('success', False)]),
                "failed_type_conversions": len([log for log in type_conversion_log if not log.get('success', True)]),
                "columns_with_unit_normalization": len(unit_normalization_log),
                "columns_with_encoding_fixes": len(encoding_standardization_log)
            },
            
            "final_schema": {
                "columns": list(standardized_df.columns),
                "dtypes": {col: str(dtype) for col, dtype in standardized_df.dtypes.items()}
            },
            
            "recommendations": self._generate_standardization_recommendations(
                type_conversion_log, unit_normalization_log, canonical_mapping
            )
        }
        
        return summary

    def _generate_standardization_recommendations(self, type_conversion_log: List, 
                                                unit_normalization_log: List,
                                                canonical_mapping: Dict) -> List[str]:
        """Generate recommendations based on standardization results"""
        
        recommendations = []
        
        # Check for failed type conversions
        failed_conversions = [log for log in type_conversion_log if not log.get('success', True)]
        if failed_conversions:
            recommendations.append(
                f"Review {len(failed_conversions)} columns with failed type conversions"
            )
        
        # Check for columns that might need manual review
        low_confidence_renames = [
            col for col, info in canonical_mapping.items()
            if info['confidence'] < 0.7
        ]
        if low_confidence_renames:
            recommendations.append(
                f"Manually review {len(low_confidence_renames)} column names with low confidence"
            )
        
        # Check for potential data quality issues
        if len(unit_normalization_log) == 0:
            recommendations.append(
                "No unit normalizations applied - verify if currency/percentage columns need standardization"
            )
        
        return recommendations
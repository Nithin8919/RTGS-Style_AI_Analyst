# Schema Inference & Canonicalization Agent (3.2)
"""
RTGS AI Analyst - Schema Inference Agent
Handles type detection, canonical naming, and schema mapping with LLM assistance
"""

import pandas as pd
import numpy as np
import json
import yaml
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from src.utils.logging import get_agent_logger, TransformLogger
from src.utils.data_helpers import detect_column_types, standardize_column_names


class SchemaInferenceAgent:
    """Agent responsible for schema inference and canonical naming"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = get_agent_logger("schema_inference")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize LLM for canonical naming
        self.llm = ChatOpenAI(
            model=self.config['openai']['model'],
            temperature=self.config['openai']['temperature'],
            max_tokens=self.config['openai']['max_tokens']
        )
        
        # Load column aliases from config
        self.column_aliases = self._load_column_aliases()
        
    def _load_column_aliases(self) -> Dict[str, str]:
        """Load column alias mappings from configuration"""
        aliases = {}
        
        for alias_group in self.config['standardization']['column_aliases']:
            canonical = alias_group['canonical']
            for alias in alias_group['aliases']:
                aliases[alias.lower()] = canonical
                
        return aliases
        
    async def process(self, state) -> Any:
        """Main schema inference processing pipeline"""
        self.logger.info("Starting schema inference process")
        
        try:
            # Initialize transform logger
            transform_logger = TransformLogger(
                log_file=Path(state.run_manifest['artifacts_paths']['logs_dir']) / "transform_log.jsonl",
                run_id=state.run_manifest['run_id']
            )
            
            # Use sample data for inference
            sample_data = getattr(state, 'sample_data', state.raw_data)
            if sample_data is None:
                raise ValueError("No data available for schema inference")
            
            # Detect column types with confidence scores
            type_detection_results = await self._detect_column_types(sample_data)
            
            # Generate canonical names
            canonical_mapping = await self._generate_canonical_names(
                sample_data, 
                type_detection_results,
                state.run_manifest
            )
            
            # Create comprehensive schema info
            schema_info = await self._create_schema_info(
                sample_data,
                type_detection_results,
                canonical_mapping,
                state
            )
            
            # Save schema mapping
            schema_mapping_path = Path(state.run_manifest['artifacts_paths']['docs_dir']) / "schema_mapping.json"
            with open(schema_mapping_path, 'w') as f:
                json.dump(canonical_mapping, f, indent=2)
            
            # Log schema inference actions
            for original_col, canonical_info in canonical_mapping.items():
                transform_logger.log_transform(
                    agent="schema_inference",
                    action="infer_type_and_name",
                    column=original_col,
                    rows_affected=len(sample_data),
                    rule_id="schema_inference_v1",
                    rationale=f"Inferred as {canonical_info['suggested_type']}, canonical name: {canonical_info['canonical_name']}",
                    parameters={
                        "original_type": str(sample_data[original_col].dtype),
                        "suggested_type": canonical_info['suggested_type'],
                        "confidence": canonical_info['confidence']
                    },
                    confidence=canonical_info['confidence_level']
                )
            
            # Update state
            state.schema_info = schema_info
            state.canonical_mapping = canonical_mapping
            state.type_detection_results = type_detection_results
            
            self.logger.info(f"Schema inference completed: {len(canonical_mapping)} columns processed")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Schema inference failed: {str(e)}")
            state.errors.append(f"Schema inference failed: {str(e)}")
            return state

    async def _detect_column_types(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Detect column types with enhanced confidence scoring"""
        self.logger.info("Detecting column types")
        
        # Use the enhanced type detection from data_helpers
        type_results = detect_column_types(df)
        
        # Enhance with domain-specific logic
        domain_hint = getattr(self, 'domain_hint', 'unknown')
        
        for col, type_info in type_results.items():
            # Rename 'type' to 'suggested_type' for consistency
            if 'type' in type_info and 'suggested_type' not in type_info:
                type_info['suggested_type'] = type_info['type']
            
            # Add domain-specific hints
            type_info = self._add_domain_specific_hints(col, type_info, domain_hint)
            type_results[col] = type_info
        
        return type_results

    def _add_domain_specific_hints(self, column_name: str, type_info: Dict, domain: str) -> Dict[str, Any]:
        """Add domain-specific type hints"""
        
        col_lower = column_name.lower()
        
        # Geographic columns
        geo_keywords = ['district', 'mandal', 'village', 'zone', 'area', 'region', 'location']
        if any(keyword in col_lower for keyword in geo_keywords):
            if type_info['suggested_type'] == 'object':
                type_info['domain_hint'] = 'geographic'
                type_info['confidence'] = min(0.9, type_info['confidence'] + 0.1)
        
        # Temporal columns
        time_keywords = ['date', 'time', 'year', 'month', 'quarter', 'period']
        if any(keyword in col_lower for keyword in time_keywords):
            if type_info['suggested_type'] in ['object', 'datetime64[ns]']:
                type_info['domain_hint'] = 'temporal'
                type_info['confidence'] = min(0.9, type_info['confidence'] + 0.1)
        
        # Metric columns
        metric_keywords = ['count', 'total', 'amount', 'value', 'rate', 'percent', 'number']
        if any(keyword in col_lower for keyword in metric_keywords):
            if type_info['suggested_type'] in ['int64', 'float64']:
                type_info['domain_hint'] = 'metric'
                type_info['confidence'] = min(0.9, type_info['confidence'] + 0.1)
        
        # ID columns
        id_keywords = ['id', 'code', 'number', 'ref', 'key']
        if any(keyword in col_lower for keyword in id_keywords):
            type_info['domain_hint'] = 'identifier'
        
        return type_info

    async def _generate_canonical_names(self, df: pd.DataFrame, type_results: Dict, run_manifest: Dict) -> Dict[str, Dict[str, Any]]:
        """Generate canonical names using rule-based approach + LLM fallback"""
        self.logger.info("Generating canonical column names")
        
        canonical_mapping = {}
        domain_hint = run_manifest['dataset_info']['domain_hint']
        
        for col in df.columns:
            type_info = type_results[col]
            
            # First try rule-based canonical naming
            canonical_name = self._get_canonical_name_rules(col, type_info, domain_hint)
            confidence_level = "high" if canonical_name != col else "medium"
            
            # If low confidence and ambiguous, use LLM
            if (type_info['confidence'] < self.config['data_quality']['llm_fallback_confidence_range'][1] and 
                type_info['confidence'] > self.config['data_quality']['llm_fallback_confidence_range'][0]):
                
                # Use LLM for canonical naming
                llm_result = await self._get_llm_canonical_name(col, type_info, domain_hint, df[col])
                if llm_result['confidence'] > type_info['confidence']:
                    canonical_name = llm_result['canonical_name']
                    type_info['suggested_type'] = llm_result.get('suggested_type', type_info['suggested_type'])
                    confidence_level = "medium"
            
            canonical_mapping[col] = {
                'canonical_name': canonical_name,
                'original_name': col,
                'suggested_type': type_info['suggested_type'],
                'confidence': type_info['confidence'],
                'confidence_level': confidence_level,
                'rationale': type_info.get('rationale', 'Rule-based inference'),
                'domain_hint': type_info.get('domain_hint', 'none'),
                'sample_values': type_info.get('sample_values', [])
            }
        
        return canonical_mapping

    def _get_canonical_name_rules(self, column_name: str, type_info: Dict, domain: str) -> str:
        """Apply rule-based canonical naming"""
        
        col_lower = column_name.lower().strip()
        
        # Check direct aliases first
        if col_lower in self.column_aliases:
            return self.column_aliases[col_lower]
        
        # Pattern-based matching
        canonical = self._apply_naming_patterns(col_lower, domain)
        
        if canonical != col_lower:
            return canonical
        
        # Default: clean the column name
        cleaned = re.sub(r'[^\w\s]', '', column_name)
        cleaned = re.sub(r'\s+', '_', cleaned)
        cleaned = cleaned.lower().strip('_')
        
        return cleaned if cleaned else column_name

    def _apply_naming_patterns(self, col_lower: str, domain: str) -> str:
        """Apply pattern-based canonical naming"""
        
        # Common patterns across domains
        patterns = {
            # Geographic patterns
            r'.*dist.*name.*|.*district.*': 'district',
            r'.*mandal.*name.*|.*sub.*dist.*': 'mandal', 
            r'.*village.*name.*|.*gram.*': 'village',
            
            # Temporal patterns
            r'.*reg.*date.*|.*registration.*date.*': 'registration_date',
            r'.*created.*date.*|.*entry.*date.*': 'created_date',
            r'.*fin.*year.*|.*financial.*year.*': 'financial_year',
            
            # Metric patterns
            r'.*total.*count.*|.*count.*total.*': 'total_count',
            r'.*amount.*rs.*|.*amount.*inr.*': 'amount_inr',
            r'.*per.*capita.*|.*per.*1000.*': 'per_capita',
            
            # Common ID patterns
            r'.*application.*id.*|.*app.*id.*': 'application_id',
            r'.*beneficiary.*id.*|.*ben.*id.*': 'beneficiary_id',
        }
        
        # Domain-specific patterns
        if domain == 'transport':
            patterns.update({
                r'.*vehicle.*reg.*|.*registration.*no.*': 'vehicle_registration',
                r'.*license.*no.*|.*dl.*no.*': 'license_number',
                r'.*vehicle.*type.*|.*category.*': 'vehicle_category',
            })
        elif domain == 'health':
            patterns.update({
                r'.*patient.*id.*|.*patient.*no.*': 'patient_id',
                r'.*hospital.*name.*|.*facility.*': 'facility_name',
                r'.*diagnosis.*|.*disease.*': 'diagnosis',
            })
        elif domain == 'education':
            patterns.update({
                r'.*student.*id.*|.*enrollment.*': 'student_id',
                r'.*school.*name.*|.*institution.*': 'school_name',
                r'.*class.*|.*grade.*|.*std.*': 'class_grade',
            })
        
        # Apply patterns
        for pattern, canonical in patterns.items():
            if re.match(pattern, col_lower):
                return canonical
        
        return col_lower

    async def _get_llm_canonical_name(self, column_name: str, type_info: Dict, domain: str, sample_data: pd.Series) -> Dict[str, Any]:
        """Use LLM to generate canonical name and type for ambiguous columns"""
        
        self.logger.info(f"Using LLM for canonical naming: {column_name}")
        
        # Prepare sample values (safe, no PII)
        sample_values = sample_data.dropna().head(10).astype(str).tolist()
        
        # Create prompt
        system_prompt = """You are an expert data analyst helping to standardize government dataset column names.

Your task is to suggest a canonical (standard) column name and data type for the given column.

Guidelines:
- Use snake_case format (lowercase with underscores)
- Be descriptive but concise
- Use common government data conventions
- Consider the domain context
- Suggest appropriate data type

Respond with a JSON object containing:
{
  "canonical_name": "suggested_standardized_name", 
  "suggested_type": "pandas_dtype",
  "confidence": 0.8,
  "rationale": "explanation of reasoning"
}"""

        user_prompt = f"""Column to analyze:
- Original name: "{column_name}"
- Domain: {domain}
- Current detected type: {type_info['suggested_type']}
- Type confidence: {type_info['confidence']:.2f}
- Sample values: {sample_values[:5]}
- Null fraction: {type_info.get('null_fraction', 0):.2f}
- Unique fraction: {type_info.get('unique_fraction', 0):.2f}

Please suggest a canonical name and confirm/correct the data type."""
        
        try:
            # Call LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse JSON response
            parser = JsonOutputParser()
            result = parser.parse(response.content)
            
            # Validate and clean the response
            result['canonical_name'] = self._clean_canonical_name(result.get('canonical_name', column_name))
            result['confidence'] = min(1.0, max(0.0, result.get('confidence', 0.5)))
            
            return result
            
        except Exception as e:
            self.logger.warning(f"LLM canonical naming failed for {column_name}: {str(e)}")
            # Fallback to rule-based
            return {
                'canonical_name': self._clean_canonical_name(column_name),
                'suggested_type': type_info['suggested_type'],
                'confidence': 0.5,
                'rationale': 'LLM failed, used rule-based fallback'
            }

    def _clean_canonical_name(self, name: str) -> str:
        """Clean and validate canonical name"""
        # Remove special characters, convert to snake_case
        cleaned = re.sub(r'[^\w\s]', '', name)
        cleaned = re.sub(r'\s+', '_', cleaned)
        cleaned = cleaned.lower().strip('_')
        
        # Ensure valid Python identifier
        if not cleaned or cleaned[0].isdigit():
            cleaned = f"col_{cleaned}"
        
        return cleaned

    async def _create_schema_info(self, df: pd.DataFrame, type_results: Dict, canonical_mapping: Dict, state) -> Dict[str, Any]:
        """Create comprehensive schema information"""
        
        # Calculate schema-level statistics
        total_columns = len(df.columns)
        high_confidence_columns = sum(1 for info in canonical_mapping.values() if info['confidence'] > 0.8)
        
        # Group columns by type
        type_distribution = {}
        for info in canonical_mapping.values():
            suggested_type = info['suggested_type']
            type_distribution[suggested_type] = type_distribution.get(suggested_type, 0) + 1
        
        # Identify problematic columns
        low_confidence_columns = [
            col for col, info in canonical_mapping.items() 
            if info['confidence'] < self.config['data_quality']['low_confidence_type_threshold']
        ]
        
        # Create comprehensive schema info
        schema_info = {
            "dataset_name": state.run_manifest['dataset_info']['dataset_name'],
            "inference_timestamp": datetime.utcnow().isoformat(),
            "total_columns": total_columns,
            "high_confidence_columns": high_confidence_columns,
            "confidence_rate": high_confidence_columns / total_columns if total_columns > 0 else 0,
            
            "type_distribution": type_distribution,
            "canonical_mapping": canonical_mapping,
            
            "quality_flags": {
                "low_confidence_columns": low_confidence_columns,
                "columns_needing_review": [
                    col for col, info in canonical_mapping.items()
                    if info['confidence_level'] == 'medium'
                ]
            },
            
            "recommendations": self._generate_schema_recommendations(canonical_mapping, type_results),
            
            "column_statistics": {
                col: {
                    "null_fraction": type_results[col].get('null_fraction', 0),
                    "unique_fraction": type_results[col].get('unique_fraction', 0),
                    "sample_values": type_results[col].get('sample_values', [])[:5]
                }
                for col in df.columns
            }
        }
        
        return schema_info

    def _generate_schema_recommendations(self, canonical_mapping: Dict, type_results: Dict) -> List[str]:
        """Generate actionable schema recommendations"""
        
        recommendations = []
        
        # Check for low confidence columns
        low_conf_columns = [
            col for col, info in canonical_mapping.items() 
            if info['confidence'] < 0.6
        ]
        
        if low_conf_columns:
            recommendations.append(
                f"Review {len(low_conf_columns)} columns with low type confidence: {', '.join(low_conf_columns[:3])}"
            )
        
        # Check for high nullability
        high_null_columns = [
            col for col, info in type_results.items()
            if info.get('null_fraction', 0) > 0.5
        ]
        
        if high_null_columns:
            recommendations.append(
                f"Consider dropping {len(high_null_columns)} columns with >50% missing values"
            )
        
        # Check for potential ID columns
        id_columns = [
            col for col, info in canonical_mapping.items()
            if info.get('domain_hint') == 'identifier'
        ]
        
        if id_columns:
            recommendations.append(
                f"Identified {len(id_columns)} potential ID columns - verify uniqueness requirements"
            )
        
        # Type conversion recommendations
        datetime_candidates = [
            col for col, info in canonical_mapping.items()
            if info['suggested_type'] == 'datetime64[ns]' and info['confidence'] > 0.7
        ]
        
        if datetime_candidates:
            recommendations.append(
                f"Convert {len(datetime_candidates)} columns to datetime format"
            )
        
        return recommendations
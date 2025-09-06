"""
RTGS AI Analyst - Validator Agent
Runs data quality validation gates and calculates confidence scores
"""

import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from src.utils.logging import get_agent_logger, TransformLogger


class ValidatorAgent:
    """Agent responsible for data quality validation and confidence assessment"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = get_agent_logger("validator")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load validation gates
        self.validation_gates = self.config.get('validation_gates', {})
        
    def json_safe_converter(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return str(obj)
    
    def deep_convert_numpy_types(self, obj):
        """Recursively convert numpy types in nested data structures"""
        if isinstance(obj, dict):
            return {key: self.deep_convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.deep_convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.deep_convert_numpy_types(item) for item in obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
        
    async def process(self, state) -> Any:
        """Main validation processing pipeline"""
        self.logger.info("Starting data validation process")
        
        try:
            # Initialize transform logger
            transform_logger = TransformLogger(
                log_file=Path(state.run_manifest['artifacts_paths']['logs_dir']) / "transform_log.jsonl",
                run_id=state.run_manifest['run_id']
            )
            
            # Get transformed data
            transformed_data = getattr(state, 'transformed_data', state.cleaned_data)
            if transformed_data is None:
                raise ValueError("No transformed data available for validation")
            
            # Run validation gates
            validation_results = await self._run_validation_gates(transformed_data, state)
            
            # Calculate confidence scores
            confidence_assessment = await self._calculate_confidence_scores(
                transformed_data, state, validation_results
            )
            
            # Generate remediation recommendations
            recommendations = await self._generate_remediation_recommendations(
                validation_results, confidence_assessment, state
            )
            
            # Create comprehensive validation report
            validation_report = self._create_validation_report(
                validation_results, confidence_assessment, recommendations, state
            )
            
            # Log validation results
            for gate_name, result in validation_results['quality_gates'].items():
                transform_logger.log_quality_check(
                    check_name=gate_name,
                    passed=result['passed'],
                    details=result['details'],
                    remediation=result.get('remediation')
                )
            
            # Save validation report - convert numpy types first
            report_path = Path(state.run_manifest['artifacts_paths']['docs_dir']) / "run_validation_report.json"
            safe_validation_report = self.deep_convert_numpy_types(validation_report)
            with open(report_path, 'w') as f:
                json.dump(safe_validation_report, f, indent=2)
            
            # Update state
            state.validation_results = validation_results
            state.confidence_assessment = confidence_assessment
            state.validation_report = validation_report
            state.quality_score = confidence_assessment['overall_score']
            
            # Determine if pipeline should continue
            critical_failures = [
                gate for gate, result in validation_results['quality_gates'].items()
                if not result['passed'] and result['severity'] == 'critical'
            ]
            
            if critical_failures:
                state.errors.append(f"Critical validation failures: {critical_failures}")
                self.logger.error(f"Critical validation failures: {critical_failures}")
            
            self.logger.info(f"Validation completed: Quality score {confidence_assessment['overall_score']}/100")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            state.errors.append(f"Data validation failed: {str(e)}")
            return state

    async def _run_validation_gates(self, df: pd.DataFrame, state) -> Dict[str, Any]:
        """Run all configured validation gates"""
        self.logger.info("Running validation gates")
        
        gate_results = {}
        
        # Critical gates
        critical_gates = self.validation_gates.get('critical', [])
        for gate in critical_gates:
            result = await self._run_single_gate(df, gate, 'critical', state)
            gate_results[gate['name']] = result
        
        # Warning gates
        warning_gates = self.validation_gates.get('warning', [])
        for gate in warning_gates:
            result = await self._run_single_gate(df, gate, 'warning', state)
            gate_results[gate['name']] = result
        
        # Calculate gate summary
        total_gates = len(gate_results)
        passed_gates = sum(1 for result in gate_results.values() if result['passed'])
        failed_gates = total_gates - passed_gates
        
        gate_summary = {
            'total_gates': total_gates,
            'passed_gates': passed_gates,
            'failed_gates': failed_gates,
            'pass_rate': passed_gates / total_gates if total_gates > 0 else 1.0
        }
        
        return {
            'quality_gates': gate_results,
            'gate_summary': gate_summary,
            'validation_timestamp': datetime.utcnow().isoformat()
        }

    async def _run_single_gate(self, df: pd.DataFrame, gate_config: Dict, severity: str, state) -> Dict[str, Any]:
        """Run a single validation gate"""
        
        gate_name = gate_config['name']
        gate_description = gate_config['description']
        threshold = gate_config['threshold']
        
        self.logger.debug(f"Running gate: {gate_name}")
        
        try:
            if gate_name == 'sufficient_data':
                # Check if dataset has minimum required rows
                actual_rows = len(df)
                passed = bool(actual_rows >= threshold)
                
                return {
                    'passed': passed,
                    'severity': severity,
                    'description': gate_description,
                    'threshold': threshold,
                    'actual_value': actual_rows,
                    'details': f'Dataset has {actual_rows} rows (threshold: {threshold})',
                    'remediation': 'Collect more data or reduce analysis scope' if not passed else None
                }
                
            elif gate_name == 'key_columns_present':
                # Check if essential columns are not entirely missing
                columns_with_data = sum(1 for col in df.columns if df[col].notnull().any())
                total_columns = len(df.columns)
                data_presence_rate = columns_with_data / total_columns if total_columns > 0 else 0
                
                passed = bool(data_presence_rate >= threshold)
                
                return {
                    'passed': passed,
                    'severity': severity,
                    'description': gate_description,
                    'threshold': threshold,
                    'actual_value': data_presence_rate,
                    'details': f'{columns_with_data}/{total_columns} columns have data ({data_presence_rate:.1%})',
                    'remediation': 'Review data collection process for empty columns' if not passed else None
                }
                
            elif gate_name == 'high_missingness':
                # Check for columns with excessive missing values
                high_missing_columns = []
                for col in df.columns:
                    null_fraction = df[col].isnull().sum() / len(df)
                    if null_fraction > threshold:
                        high_missing_columns.append({
                            'column': col,
                            'missing_fraction': null_fraction
                        })
                
                passed = bool(len(high_missing_columns) == 0)
                
                return {
                    'passed': passed,
                    'severity': severity,
                    'description': gate_description,
                    'threshold': threshold,
                    'actual_value': len(high_missing_columns),
                    'details': f'{len(high_missing_columns)} columns exceed {threshold:.1%} missing threshold',
                    'high_missing_columns': high_missing_columns,
                    'remediation': f'Review data quality for {len(high_missing_columns)} columns with high missingness' if not passed else None
                }
                
            elif gate_name == 'reasonable_duplicates':
                # Check duplicate rate
                duplicate_count = df.duplicated().sum()
                duplicate_rate = duplicate_count / len(df) if len(df) > 0 else 0
                
                passed = bool(duplicate_rate <= threshold)
                
                return {
                    'passed': passed,
                    'severity': severity,
                    'description': gate_description,
                    'threshold': threshold,
                    'actual_value': duplicate_rate,
                    'details': f'{duplicate_count} duplicates ({duplicate_rate:.1%} rate)',
                    'remediation': 'Review data collection for systematic duplication' if not passed else None
                }
                
            else:
                # Unknown gate - skip
                return {
                    'passed': True,
                    'severity': severity,
                    'description': f'Unknown gate: {gate_name}',
                    'details': 'Gate not implemented',
                    'remediation': None
                }
                
        except Exception as e:
            self.logger.error(f"Gate {gate_name} failed with error: {str(e)}")
            return {
                'passed': False,
                'severity': severity,
                'description': gate_description,
                'details': f'Gate execution failed: {str(e)}',
                'error': str(e),
                'remediation': 'Review gate configuration and data format'
            }

    async def _calculate_confidence_scores(self, df: pd.DataFrame, state, validation_results: Dict) -> Dict[str, Any]:
        """Calculate comprehensive confidence scores"""
        self.logger.info("Calculating confidence scores")
        
        confidence_scores = {}
        
        # Data completeness score (0-100)
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        completeness_score = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0
        confidence_scores['data_completeness'] = completeness_score
        
        # Data consistency score (based on type conversions and standardization)
        consistency_score = 100
        standardization_summary = getattr(state, 'standardization_summary', {})
        
        # Reduce score for failed type conversions
        failed_conversions = standardization_summary.get('quality_metrics', {}).get('failed_type_conversions', 0)
        if failed_conversions > 0:
            consistency_score -= min(20, failed_conversions * 5)
        
        confidence_scores['data_consistency'] = max(0, consistency_score)
        
        # Transformation quality score
        transformation_summary = getattr(state, 'transformation_summary', {})
        transformation_count = transformation_summary.get('transformation_summary', {}).get('total_transformations', 0)
        
        # Higher score for more successful transformations
        transformation_score = min(100, 70 + (transformation_count * 5))
        confidence_scores['transformation_quality'] = transformation_score
        
        # Validation gate score
        gate_pass_rate = validation_results['gate_summary']['pass_rate']
        gate_score = gate_pass_rate * 100
        confidence_scores['validation_gates'] = gate_score
        
        # Schema inference quality
        schema_info = getattr(state, 'schema_info', {})
        schema_confidence = schema_info.get('confidence_rate', 0.5) * 100
        confidence_scores['schema_quality'] = schema_confidence
        
        # Calculate weighted overall score
        weights = {
            'data_completeness': 0.25,
            'data_consistency': 0.20,
            'transformation_quality': 0.15,
            'validation_gates': 0.25,
            'schema_quality': 0.15
        }
        
        overall_score = sum(
            confidence_scores[component] * weights[component]
            for component in weights.keys()
        )
        
        # Apply penalties for critical issues
        critical_failures = [
            gate for gate, result in validation_results['quality_gates'].items()
            if not result['passed'] and result['severity'] == 'critical'
        ]
        
        if critical_failures:
            overall_score *= 0.5  # 50% penalty for critical failures
        
        # Ensure score is in valid range
        overall_score = max(0, min(100, overall_score))
        
        # Determine confidence level
        if overall_score >= 85:
            confidence_level = "HIGH"
        elif overall_score >= 70:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        return {
            'component_scores': confidence_scores,
            'overall_score': round(overall_score, 1),
            'confidence_level': confidence_level,
            'score_breakdown': {
                'weights_used': weights,
                'penalties_applied': len(critical_failures) > 0,
                'critical_failures': len(critical_failures)
            },
            'assessment_timestamp': datetime.utcnow().isoformat()
        }

    async def _generate_remediation_recommendations(self, validation_results: Dict, 
                                                  confidence_assessment: Dict, state) -> List[Dict[str, Any]]:
        """Generate actionable remediation recommendations"""
        self.logger.info("Generating remediation recommendations")
        
        recommendations = []
        
        # Recommendations based on failed gates
        for gate_name, result in validation_results['quality_gates'].items():
            if not result['passed'] and result.get('remediation'):
                recommendations.append({
                    'type': 'validation_failure',
                    'priority': 'high' if result['severity'] == 'critical' else 'medium',
                    'gate': gate_name,
                    'issue': result['details'],
                    'recommendation': result['remediation'],
                    'estimated_effort': 'medium'
                })
        
        # Recommendations based on confidence scores
        component_scores = confidence_assessment['component_scores']
        
        if component_scores['data_completeness'] < 80:
            recommendations.append({
                'type': 'data_quality',
                'priority': 'high',
                'issue': f"Data completeness is low ({component_scores['data_completeness']:.1f}%)",
                'recommendation': 'Review data collection processes and consider imputation strategies',
                'estimated_effort': 'high'
            })
        
        if component_scores['data_consistency'] < 70:
            recommendations.append({
                'type': 'data_quality',
                'priority': 'medium',
                'issue': f"Data consistency issues detected ({component_scores['data_consistency']:.1f}%)",
                'recommendation': 'Review standardization rules and type conversion errors',
                'estimated_effort': 'medium'
            })
        
        if component_scores['schema_quality'] < 60:
            recommendations.append({
                'type': 'schema_inference',
                'priority': 'medium',
                'issue': f"Low schema inference confidence ({component_scores['schema_quality']:.1f}%)",
                'recommendation': 'Manually review column types and canonical names',
                'estimated_effort': 'low'
            })
        
        # Recommendations based on overall confidence
        overall_score = confidence_assessment['overall_score']
        
        if overall_score < 50:
            recommendations.append({
                'type': 'overall_quality',
                'priority': 'critical',
                'issue': f"Overall data quality is very low ({overall_score:.1f}/100)",
                'recommendation': 'Consider collecting additional/better quality data before proceeding with analysis',
                'estimated_effort': 'high'
            })
        elif overall_score < 70:
            recommendations.append({
                'type': 'overall_quality',
                'priority': 'high',
                'issue': f"Overall data quality needs improvement ({overall_score:.1f}/100)",
                'recommendation': 'Address high-priority data quality issues before finalizing analysis',
                'estimated_effort': 'medium'
            })
        
        # Sort recommendations by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))
        
        return recommendations

    def _create_validation_report(self, validation_results: Dict, confidence_assessment: Dict,
                                recommendations: List, state) -> Dict[str, Any]:
        """Create comprehensive validation report"""
        
        # Extract key metrics from state
        original_rows = len(getattr(state, 'raw_data', pd.DataFrame()))
        final_rows = len(getattr(state, 'transformed_data', pd.DataFrame()))
        
        cleaning_summary = getattr(state, 'cleaning_summary', {})
        transformation_summary = getattr(state, 'transformation_summary', {})
        
        report = {
            "validation_timestamp": datetime.utcnow().isoformat(),
            "run_id": state.run_manifest['run_id'],
            "dataset_name": state.run_manifest['dataset_info']['dataset_name'],
            
            "executive_summary": {
                "overall_confidence": confidence_assessment['confidence_level'],
                "quality_score": confidence_assessment['overall_score'],
                "critical_issues": len([r for r in recommendations if r['priority'] == 'critical']),
                "data_preservation_rate": (final_rows / original_rows * 100) if original_rows > 0 else 0,
                "recommendation_summary": f"{len(recommendations)} recommendations generated"
            },
            
            "validation_results": validation_results,
            "confidence_assessment": confidence_assessment,
            "recommendations": recommendations,
            
            "data_pipeline_summary": {
                "original_data_shape": {
                    "rows": original_rows,
                    "columns": len(getattr(state, 'raw_data', pd.DataFrame()).columns)
                },
                "final_data_shape": {
                    "rows": final_rows,
                    "columns": len(getattr(state, 'transformed_data', pd.DataFrame()).columns)
                },
                "operations_performed": {
                    "cleaning_operations": cleaning_summary.get('operations_summary', {}).get('total_operations', 0),
                    "transformations_applied": transformation_summary.get('transformation_summary', {}).get('total_transformations', 0),
                    "features_created": transformation_summary.get('features_added', {}).get('total_new_columns', 0)
                }
            },
            
            "quality_metrics": {
                "data_completeness": confidence_assessment['component_scores']['data_completeness'],
                "schema_inference_quality": confidence_assessment['component_scores']['schema_quality'],
                "validation_gate_pass_rate": validation_results['gate_summary']['pass_rate'] * 100,
                "overall_quality_score": confidence_assessment['overall_score']
            },
            
            "next_steps": {
                "proceed_to_analysis": confidence_assessment['overall_score'] >= 50,
                "requires_manual_review": len([r for r in recommendations if r['priority'] in ['critical', 'high']]) > 0,
                "estimated_remediation_effort": self._estimate_total_effort(recommendations)
            }
        }
        
        return report

    def _estimate_total_effort(self, recommendations: List) -> str:
        """Estimate total effort required for remediation"""
        
        effort_weights = {'low': 1, 'medium': 3, 'high': 5}
        total_effort = sum(effort_weights.get(rec.get('estimated_effort', 'medium'), 3) for rec in recommendations)
        
        if total_effort <= 3:
            return 'low'
        elif total_effort <= 8:
            return 'medium'
        else:
            return 'high'
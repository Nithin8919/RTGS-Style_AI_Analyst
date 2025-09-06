"""
RTGS AI Analyst - Flow Controller
Orchestrates the multi-agent data analysis pipeline
"""

import asyncio
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import traceback

from src.utils.logging import get_logger
from src.agents.ingestion_agent import IngestionAgent
from src.agents.schema_inference_agent import SchemaInferenceAgent
from src.agents.standardization_agent import StandardizationAgent
from src.agents.cleaning_agent import CleaningAgent
from src.agents.transformation_agent import TransformationAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.insight_agent import InsightAgent
from src.agents.report_agent import ReportAgent
from src.agents.observability_agent import ObservabilityAgent


class PipelineState:
    """State object passed between agents"""
    
    def __init__(self, dataset_path: str, config: Dict):
        self.dataset_path = dataset_path
        self.config = config
        self.errors = []
        self.warnings = []
        
        # Create run manifest
        run_id = str(uuid.uuid4())
        dataset_name = Path(dataset_path).stem
        self.run_manifest = {
            'run_id': run_id,
            'start_time': datetime.utcnow().isoformat(),
            'dataset_path': dataset_path,
            'dataset_info': {
                'dataset_name': dataset_name,
                'source_path': dataset_path,
                'domain_hint': config.get('domain', 'unknown'),
                'scope': config.get('scope', 'unspecified'),
                'description': config.get('description', 'No description provided')
            },
            'run_config': {
                'sample_rows': config.get('sample_rows', 500)
            },
            'artifacts_paths': {}
        }
        
        # Setup artifacts paths after run_manifest is created
        self.run_manifest['artifacts_paths'] = self._setup_artifacts_paths()
        
        # Data at different stages
        self.raw_data = None
        self.standardized_data = None
        self.cleaned_data = None
        self.transformed_data = None
        self.analysis_results = None
        self.insights = None
        self.reports = None
        
        # Quality and confidence scores
        self.quality_score = 0
        self.confidence_score = 0
    
    def _setup_artifacts_paths(self) -> Dict[str, str]:
        """Setup artifact directory paths"""
        base_dir = Path("artifacts")
        run_id = self.run_manifest['run_id'][:8]  # Short run ID
        
        paths = {
            'logs_dir': str(base_dir / "logs" / f"run_{run_id}"),
            'plots_dir': str(base_dir / "plots" / f"run_{run_id}"),
            'reports_dir': str(base_dir / "reports" / f"run_{run_id}"),
            'docs_dir': str(base_dir / "docs" / f"run_{run_id}"),
            'quick_start_dir': str(base_dir / "quick_start" / f"run_{run_id}")
        }
        
        # Create directories
        for path in paths.values():
            Path(path).mkdir(parents=True, exist_ok=True)
        
        return paths


class RTGSOrchestrator:
    """Main orchestrator for the RTGS AI Analyst pipeline"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = get_logger("orchestrator")
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            self.config = {}
        
        # Initialize agents
        self.agents = {
            'ingestion': IngestionAgent(),
            'schema': SchemaInferenceAgent(config_path),
            'standardization': StandardizationAgent(),
            'cleaning': CleaningAgent(),
            'transformation': TransformationAgent(),
            'analysis': AnalysisAgent(),
            'insight': InsightAgent(),
            'report': ReportAgent(),
            'observability': ObservabilityAgent()
        }
        
        # Pipeline configuration
        self.pipeline_config = self.config.get('pipeline', {})
        self.max_retries = self.pipeline_config.get('max_retries', 2)
        self.continue_on_error = self.pipeline_config.get('continue_on_error', True)
    
    async def run_pipeline(self, dataset_path: str, domain_hint: str = None) -> Dict[str, Any]:
        """Run the complete data analysis pipeline"""
        
        self.logger.info(f"Starting RTGS AI Analyst pipeline for: {dataset_path}")
        
        # Initialize state
        state = PipelineState(dataset_path, self.config)
        
        if domain_hint:
            state.run_manifest['dataset_info']['domain_hint'] = domain_hint
        
        try:
            # Define pipeline stages
            pipeline_stages = [
                ('ingestion', 'Data Ingestion'),
                ('schema', 'Schema Inference'),
                ('standardization', 'Data Standardization'),
                ('cleaning', 'Data Cleaning'),
                ('transformation', 'Feature Engineering'),
                ('analysis', 'Statistical Analysis'),
                ('insight', 'Insight Generation'),
                ('report', 'Report Assembly'),
                ('observability', 'Observability Finalization')
            ]
            
            # Execute pipeline stages
            for agent_name, stage_name in pipeline_stages:
                self.logger.info(f"Executing stage: {stage_name}")
                
                start_time = datetime.utcnow()
                try:
                    agent = self.agents[agent_name]
                    state = await self._execute_agent_with_retry(agent, state, agent_name)
                    
                    execution_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    # Log execution to observability
                    await self.agents['observability'].log_agent_execution(
                        agent_name, start_time, execution_time, 'success', state
                    )
                    
                    self.logger.info(f"Completed stage: {stage_name} in {execution_time:.2f}s")
                    
                except Exception as e:
                    execution_time = (datetime.utcnow() - start_time).total_seconds()
                    error_msg = f"Stage {stage_name} failed: {str(e)}"
                    self.logger.error(error_msg)
                    
                    # Log execution to observability
                    await self.agents['observability'].log_agent_execution(
                        agent_name, start_time, execution_time, 'error', state, str(e)
                    )
                    
                    state.errors.append(error_msg)
                    
                    if not self.continue_on_error:
                        raise RuntimeError(f"Pipeline stopped due to error in {stage_name}: {str(e)}")
            
            # Finalize run
            state.run_manifest['end_time'] = datetime.utcnow().isoformat()
            state.run_manifest['status'] = 'completed' if len(state.errors) == 0 else 'completed_with_errors'
            
            self.logger.info(f"Pipeline completed with {len(state.errors)} errors")
            
            return self._create_pipeline_summary(state)
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            state.errors.append(f"Pipeline failure: {str(e)}")
            state.run_manifest['status'] = 'failed'
            state.run_manifest['end_time'] = datetime.utcnow().isoformat()
            
            return self._create_pipeline_summary(state)
    
    async def _execute_agent_with_retry(self, agent, state, agent_name: str):
        """Execute an agent with retry logic"""
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if hasattr(agent, 'process'):
                    return await agent.process(state)
                else:
                    self.logger.warning(f"Agent {agent_name} does not have process method, skipping")
                    return state
                    
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    self.logger.warning(f"Agent {agent_name} failed (attempt {attempt + 1}), retrying: {str(e)}")
                    await asyncio.sleep(1)  # Brief delay before retry
                else:
                    self.logger.error(f"Agent {agent_name} failed after {self.max_retries + 1} attempts")
        
        # If we get here, all retries failed
        raise last_exception
    
    def _create_pipeline_summary(self, state: PipelineState) -> Dict[str, Any]:
        """Create a summary of the pipeline execution"""
        
        summary = {
            'run_manifest': state.run_manifest,
            'pipeline_status': state.run_manifest.get('status', 'unknown'),
            'errors_count': len(state.errors),
            'warnings_count': len(state.warnings),
            'errors': state.errors,
            'warnings': state.warnings,
            'data_summary': {
                'raw_data_available': state.raw_data is not None,
                'standardized_data_available': state.standardized_data is not None,
                'cleaned_data_available': state.cleaned_data is not None,
                'transformed_data_available': state.transformed_data is not None,
                'analysis_results_available': state.analysis_results is not None,
                'insights_available': state.insights is not None,
                'reports_available': state.reports is not None
            },
            'quality_metrics': {
                'overall_quality_score': state.quality_score,
                'overall_confidence_score': state.confidence_score
            },
            'artifacts_paths': state.run_manifest['artifacts_paths']
        }
        
        return summary
    
    async def validate_prerequisites(self) -> Dict[str, Any]:
        """Validate that all prerequisites are met"""
        
        validation_results = {
            'status': 'passed',
            'checks': {}
        }
        
        # Check if config file exists
        config_exists = Path("config.yaml").exists()
        validation_results['checks']['config_file'] = {
            'status': 'passed' if config_exists else 'warning',
            'message': 'Config file exists' if config_exists else 'Config file not found, using defaults'
        }
        
        # Check if required directories exist
        required_dirs = ['data/raw', 'artifacts']
        for dir_path in required_dirs:
            dir_exists = Path(dir_path).exists()
            validation_results['checks'][f'directory_{dir_path.replace("/", "_")}'] = {
                'status': 'passed' if dir_exists else 'failed',
                'message': f'Directory {dir_path} exists' if dir_exists else f'Directory {dir_path} missing'
            }
            
            if not dir_exists:
                validation_results['status'] = 'failed'
        
        # Check agent initialization
        for agent_name, agent in self.agents.items():
            try:
                # Try to access agent logger (basic functionality check)
                if hasattr(agent, 'logger'):
                    validation_results['checks'][f'agent_{agent_name}'] = {
                        'status': 'passed',
                        'message': f'{agent_name} agent initialized successfully'
                    }
                else:
                    validation_results['checks'][f'agent_{agent_name}'] = {
                        'status': 'warning',
                        'message': f'{agent_name} agent missing logger attribute'
                    }
            except Exception as e:
                validation_results['checks'][f'agent_{agent_name}'] = {
                    'status': 'failed',
                    'message': f'{agent_name} agent initialization failed: {str(e)}'
                }
                validation_results['status'] = 'failed'
        
        return validation_results


# Convenience function for backward compatibility
async def run_rtgs_pipeline(dataset_path: str, domain_hint: str = None, config_path: str = "config.yaml") -> Dict[str, Any]:
    """Convenience function to run the RTGS pipeline"""
    orchestrator = RTGSOrchestrator(config_path)
    return await orchestrator.run_pipeline(dataset_path, domain_hint)

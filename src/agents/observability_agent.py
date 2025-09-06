"""
RTGS AI Analyst - Observability Agent (3.11)
Handles monitoring, tracing, and observability for the entire pipeline
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import traceback
import psutil
import time
import hashlib

from src.utils.logging import get_agent_logger

class ObservabilityAgent:
    """Agent responsible for monitoring, tracing, and observability"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = get_agent_logger("observability")
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            self.config = {}
        
        # Observability configuration
        self.observability_config = self.config.get('observability', {})
        self.langsmith_enabled = self.observability_config.get('langsmith_enabled', False)
        self.trace_sampling_rate = self.observability_config.get('trace_sampling_rate', 1.0)
        
        # Initialize tracing components
        self.trace_logs = []
        self.performance_metrics = {}
        self.error_tracking = []
        
    async def process(self, state) -> Any:
        """Main observability processing - finalize run tracking"""
        self.logger.info("Finalizing observability and tracing")
        
        try:
            # Collect final run statistics
            run_summary = await self._create_run_summary(state)
            
            # Generate performance report
            performance_report = await self._generate_performance_report(state)
            
            # Create trace summary
            trace_summary = await self._create_trace_summary(state)
            
            # Save observability data
            await self._save_observability_data(state, run_summary, performance_report, trace_summary)
            
            # Export to LangSmith if enabled
            if self.langsmith_enabled:
                await self._export_to_langsmith(state, run_summary)
            
            # Update state with observability info
            state.observability_summary = {
                'run_summary': run_summary,
                'performance_report': performance_report,
                'trace_summary': trace_summary,
                'observability_timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.info("Observability finalization completed")
            return state
            
        except Exception as e:
            self.logger.error(f"Observability finalization failed: {str(e)}")
            state.errors.append(f"Observability error: {str(e)}")
            return state
    
    async def log_agent_execution(self, agent_name: str, start_time: datetime, 
                                 execution_time: float, status: str, state: Any, 
                                 error: str = None) -> None:
        """Log agent execution for tracing"""
        
        trace_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'agent_name': agent_name,
            'start_time': start_time.isoformat(),
            'execution_time_seconds': execution_time,
            'status': status,
            'run_id': getattr(state, 'run_manifest', {}).get('run_id', 'unknown'),
            'memory_usage_mb': self._get_memory_usage(),
            'cpu_usage_percent': self._get_cpu_usage()
        }
        
        if error:
            trace_entry['error'] = error
        
        self.trace_logs.append(trace_entry)
        self.logger.debug(f"Logged execution for {agent_name}: {status} in {execution_time:.2f}s")
    
    async def finalize_run(self, state) -> Any:
        """Alias for process method to match expected interface"""
        return await self.process(state)
    
    async def _create_run_summary(self, state) -> Dict[str, Any]:
        """Create comprehensive run summary"""
        return {
            'run_metadata': {
                'run_id': getattr(state, 'run_manifest', {}).get('run_id', 'unknown'),
                'end_time': datetime.utcnow().isoformat(),
                'pipeline_status': 'completed' if len(getattr(state, 'errors', [])) == 0 else 'completed_with_errors'
            },
            'execution_summary': {
                'total_agents_executed': len(self.performance_metrics),
                'total_errors': len(self.error_tracking),
                'success_rate': self._calculate_success_rate()
            }
        }
    
    async def _generate_performance_report(self, state) -> Dict[str, Any]:
        """Generate detailed performance report"""
        return {
            'agent_performance': self.performance_metrics.copy(),
            'execution_timeline': [
                {
                    'timestamp': log.get('timestamp'),
                    'agent': log.get('agent_name'),
                    'status': log.get('status')
                }
                for log in self.trace_logs if isinstance(log, dict)
            ]
        }
    
    async def _create_trace_summary(self, state) -> Dict[str, Any]:
        """Create trace summary for debugging"""
        return {
            'total_traces': len(self.trace_logs),
            'agent_execution_order': [
                log.get('agent_name') for log in self.trace_logs 
                if isinstance(log, dict) and 'agent_name' in log
            ]
        }
    
    async def _save_observability_data(self, state, run_summary: Dict, 
                                      performance_report: Dict, trace_summary: Dict) -> None:
        """Save observability data to files"""
        try:
            obs_dir = Path(state.run_manifest['artifacts_paths']['logs_dir']) / "observability"
            obs_dir.mkdir(parents=True, exist_ok=True)
            
            with open(obs_dir / "run_summary.json", 'w') as f:
                json.dump(run_summary, f, indent=2, default=str)
            
            self.logger.info(f"Observability data saved to {obs_dir}")
        except Exception as e:
            self.logger.error(f"Failed to save observability data: {e}")
    
    async def _export_to_langsmith(self, state, run_summary: Dict) -> None:
        """Export traces to LangSmith if enabled"""
        self.logger.info("LangSmith export enabled - would export trace data")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall pipeline success rate"""
        if not self.performance_metrics:
            return 100.0
        
        total_executions = sum(metrics['total_executions'] for metrics in self.performance_metrics.values())
        total_successes = sum(metrics['success_count'] for metrics in self.performance_metrics.values())
        
        return (total_successes / total_executions * 100) if total_executions > 0 else 100.0

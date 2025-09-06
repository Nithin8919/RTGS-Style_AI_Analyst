"""
RTGS AI Analyst - Flow Controller
LangGraph-based orchestrator for multi-agent pipeline
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import traceback

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from src.utils.logging import get_logger
from src.agents.ingestion_agent import IngestionAgent
from src.agents.schema_inference_agent import SchemaInferenceAgent
from src.agents.standardization_agent import StandardizationAgent
from src.agents.cleaning_agent import CleaningAgent
from src.agents.transformation_agent import TransformationAgent
from src.agents.validator_agent import ValidatorAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.insight_agent import InsightAgent
from src.agents.report_agent import EnhancedReportAgent
from src.agents.memory_agent import MemoryAgent
from src.agents.observability_agent import ObservabilityAgent


@dataclass
class PipelineState:
    """State object passed between agents in the pipeline"""
    run_manifest: Dict[str, Any]
    dataset_path: Optional[str] = None
    raw_data: Optional[Any] = None
    standardized_data: Optional[Any] = None
    cleaned_data: Optional[Any] = None
    transformed_data: Optional[Any] = None
    schema_info: Optional[Dict] = None
    cleaning_summary: Optional[Dict] = None
    analysis_results: Optional[Dict] = None
    insights: Optional[Dict] = None
    reports: Optional[Dict] = None
    errors: List[str] = None
    current_agent: str = "none"
    confidence_score: float = 0.0
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class RTGSOrchestrator:
    """Main orchestrator using LangGraph for agent coordination"""
    
    def __init__(self, run_manifest: Dict[str, Any]):
        self.logger = get_logger(__name__)
        self.run_manifest = run_manifest
        self.state = PipelineState(run_manifest=run_manifest)
        
        # Initialize agents
        self._initialize_agents()
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
        
    def _initialize_agents(self):
        """Initialize all pipeline agents"""
        self.agents = {
            'ingestion': IngestionAgent(),
            'schema_inference': SchemaInferenceAgent(),
            'standardization': StandardizationAgent(),
            'cleaning': CleaningAgent(),
            'transformation': TransformationAgent(),
            'validator': ValidatorAgent(),
            'analysis': AnalysisAgent(),
            'insight': InsightAgent(),
            'report': EnhancedReportAgent(),
            'memory': MemoryAgent(),
            'observability': ObservabilityAgent()
        }
        
        self.logger.info(f"Initialized {len(self.agents)} agents")

    def _build_workflow(self) -> CompiledStateGraph:
        """Build LangGraph workflow defining agent execution order"""
        
        # Create state graph
        workflow = StateGraph(PipelineState)
        
        # Add agent nodes
        workflow.add_node("ingestion", self._execute_ingestion)
        workflow.add_node("schema_inference", self._execute_schema_inference)
        workflow.add_node("standardization", self._execute_standardization)
        workflow.add_node("cleaning", self._execute_cleaning)
        workflow.add_node("transformation", self._execute_transformation)
        workflow.add_node("validation", self._execute_validation)
        workflow.add_node("analysis", self._execute_analysis)
        workflow.add_node("insight_generation", self._execute_insight_generation)
        workflow.add_node("report_assembly", self._execute_report_assembly)
        workflow.add_node("memory_update", self._execute_memory_update)
        workflow.add_node("observability", self._execute_observability)
        
        # Define linear flow with conditional branches
        workflow.add_edge(START, "ingestion")
        workflow.add_edge("ingestion", "schema_inference")
        workflow.add_edge("schema_inference", "standardization")
        workflow.add_edge("standardization", "cleaning")
        workflow.add_edge("cleaning", "transformation")
        workflow.add_edge("transformation", "validation")
        
        # Conditional flow after validation
        workflow.add_conditional_edges(
            "validation",
            self._should_continue_after_validation,
            {
                "continue": "analysis",
                "retry_cleaning": "cleaning",
                "fail": END
            }
        )
        
        workflow.add_edge("analysis", "insight_generation")
        workflow.add_edge("insight_generation", "report_assembly")
        workflow.add_edge("report_assembly", "memory_update")
        workflow.add_edge("memory_update", "observability")
        workflow.add_edge("observability", END)
        
        # Compile workflow
        return workflow.compile()

    async def _execute_agent(self, agent_name: str, agent_func, state: PipelineState) -> PipelineState:
        """Generic agent execution wrapper with error handling"""
        try:
            state.current_agent = agent_name
            self.logger.info(f"Executing agent: {agent_name}")
            
            # Record start time for observability
            start_time = datetime.utcnow()
            
            # Execute agent
            updated_state = await agent_func(state)
            
            # Record execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(f"Agent {agent_name} completed in {execution_time:.2f}s")
            
            # Log to observability
            await self.agents['observability'].log_agent_execution(
                agent_name, start_time, execution_time, "success", state
            )
            
            return updated_state
            
        except Exception as e:
            error_msg = f"Agent {agent_name} failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            state.errors.append(error_msg)
            
            # Log error to observability
            await self.agents['observability'].log_agent_execution(
                agent_name, start_time, 0, "error", state, error=str(e)
            )
            
            return state

    async def _execute_ingestion(self, state: PipelineState) -> PipelineState:
        """Execute data ingestion agent"""
        return await self._execute_agent(
            "ingestion",
            self.agents['ingestion'].process,
            state
        )

    async def _execute_schema_inference(self, state: PipelineState) -> PipelineState:
        """Execute schema inference agent"""
        return await self._execute_agent(
            "schema_inference", 
            self.agents['schema_inference'].process,
            state
        )

    async def _execute_standardization(self, state: PipelineState) -> PipelineState:
        """Execute standardization agent"""
        return await self._execute_agent(
            "standardization",
            self.agents['standardization'].process,
            state
        )

    async def _execute_cleaning(self, state: PipelineState) -> PipelineState:
        """Execute cleaning agent"""
        return await self._execute_agent(
            "cleaning",
            self.agents['cleaning'].process,
            state
        )

    async def _execute_transformation(self, state: PipelineState) -> PipelineState:
        """Execute transformation agent"""
        return await self._execute_agent(
            "transformation",
            self.agents['transformation'].process,
            state
        )

    async def _execute_validation(self, state: PipelineState) -> PipelineState:
        """Execute validation agent"""
        return await self._execute_agent(
            "validation",
            self.agents['validator'].process,
            state
        )

    async def _execute_analysis(self, state: PipelineState) -> PipelineState:
        """Execute analysis agent"""
        return await self._execute_agent(
            "analysis",
            self.agents['analysis'].process,
            state
        )

    async def _execute_insight_generation(self, state: PipelineState) -> PipelineState:
        """Execute insight generation agent"""
        return await self._execute_agent(
            "insight_generation",
            self.agents['insight'].process,
            state
        )

    async def _execute_report_assembly(self, state: PipelineState) -> PipelineState:
        """Execute report assembly agent"""
        return await self._execute_agent(
            "report_assembly",
            self.agents['report'].process,
            state
        )

    async def _execute_memory_update(self, state: PipelineState) -> PipelineState:
        """Execute memory update agent"""
        return await self._execute_agent(
            "memory_update",
            self.agents['memory'].process,
            state
        )

    async def _execute_observability(self, state: PipelineState) -> PipelineState:
        """Execute observability agent"""
        return await self._execute_agent(
            "observability",
            self.agents['observability'].finalize_run,
            state
        )

    def _should_continue_after_validation(self, state: PipelineState) -> str:
        """Conditional logic after validation - decides next step"""
        
        # Check if validation found critical issues
        validation_results = getattr(state, 'validation_results', {})
        quality_gates = validation_results.get('quality_gates', {})
        
        failed_gates = [gate for gate, passed in quality_gates.items() if not passed]
        
        if not failed_gates:
            # All quality gates passed - continue to analysis
            return "continue"
        
        # Check if issues are recoverable
        critical_failures = [gate for gate in failed_gates if 'critical' in gate.lower()]
        
        if critical_failures:
            # Critical failure - end pipeline
            self.logger.error(f"Critical quality gates failed: {critical_failures}")
            state.errors.append(f"Critical quality gates failed: {critical_failures}")
            return "fail"
        
        # Check if we can retry cleaning
        retry_count = getattr(state, 'cleaning_retry_count', 0)
        if retry_count < 2:  # Max 2 retries
            self.logger.warning(f"Retrying cleaning due to failed gates: {failed_gates}")
            state.cleaning_retry_count = retry_count + 1
            return "retry_cleaning"
        
        # Too many retries - continue with warnings
        self.logger.warning(f"Proceeding despite failed gates: {failed_gates}")
        state.confidence_score *= 0.7  # Reduce confidence
        return "continue"

    async def dry_run(self) -> Dict[str, Any]:
        """Execute dry run - ingestion and schema inference only"""
        self.logger.info("Starting dry run - ingestion and schema inference only")
        
        try:
            # Execute only first two agents
            state = await self._execute_ingestion(self.state)
            state = await self._execute_schema_inference(state)
            
            # Prepare dry run results
            results = {
                "mode": "dry-run",
                "status": "completed",
                "dataset_profile": state.schema_info,
                "errors": state.errors,
                "confidence_overall": "PENDING"
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Dry run failed: {str(e)}")
            return {
                "mode": "dry-run", 
                "status": "failed",
                "error": str(e),
                "confidence_overall": "LOW"
            }

    async def preview_run(self) -> Dict[str, Any]:
        """Execute preview run - through cleaning with transforms preview"""
        self.logger.info("Starting preview run - through cleaning with preview")
        
        try:
            # Execute through cleaning
            state = await self._execute_ingestion(self.state)
            state = await self._execute_schema_inference(state)
            state = await self._execute_standardization(state)
            state = await self._execute_cleaning(state)
            
            # Generate transforms preview without applying transformations
            preview_data = await self.agents['transformation'].generate_preview(state)
            
            # Save preview to file
            artifacts_dir = Path(self.run_manifest['artifacts_paths']['docs_dir'])
            preview_path = artifacts_dir / "transforms_preview.csv"
            preview_data.to_csv(preview_path, index=False)
            
            results = {
                "mode": "preview",
                "status": "completed", 
                "transforms_preview_path": str(preview_path),
                "cleaning_summary": state.cleaning_summary,
                "errors": state.errors,
                "confidence_overall": "PENDING"
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Preview run failed: {str(e)}")
            return {
                "mode": "preview",
                "status": "failed", 
                "error": str(e),
                "confidence_overall": "LOW"
            }

    async def full_run(self) -> Dict[str, Any]:
        """Execute complete pipeline"""
        self.logger.info("Starting full pipeline execution")
        
        try:
            # Set initial dataset path from manifest
            self.state.dataset_path = self.run_manifest['dataset_info']['source_path']
            
            # Execute complete workflow using LangGraph
            final_state = await self.workflow.ainvoke(
                self.state,
                config=RunnableConfig(
                    metadata={
                        "run_id": self.run_manifest['run_id'],
                        "pipeline": "rtgs_ai_analyst"
                    }
                )
            )
            
            # Handle case where LangGraph returns dict instead of PipelineState
            if isinstance(final_state, dict):
                # Convert dict back to PipelineState if needed
                state_obj = PipelineState(run_manifest=self.run_manifest)
                for key, value in final_state.items():
                    if hasattr(state_obj, key):
                        setattr(state_obj, key, value)
                final_state = state_obj
            
            # Determine overall confidence
            confidence = self._calculate_overall_confidence(final_state)
            
            # Extract key insights for CLI display
            key_insights = []
            if hasattr(final_state, 'insights') and final_state.insights:
                insights_data = final_state.insights.get('key_findings', [])
                key_insights = [finding['finding'] for finding in insights_data[:3]]
            
            # Collect pipeline statistics
            # Enhanced pipeline stats for data agnostic capabilities
            cleaned_data = getattr(final_state, 'cleaned_data', None)
            transformed_data = getattr(final_state, 'transformed_data', None)
            analysis_results = getattr(final_state, 'analysis_results', {})
            insights = getattr(final_state, 'insights', {})
            
            pipeline_stats = {
                "rows_processed": len(cleaned_data) if cleaned_data is not None else 0,
                "original_columns": len(getattr(final_state, 'raw_data', None).columns) if hasattr(final_state, 'raw_data') and final_state.raw_data is not None else 0,
                "cleaned_columns": len(cleaned_data.columns) if cleaned_data is not None else 0,
                "transformed_columns": len(transformed_data.columns) if transformed_data is not None else 0,
                "features_engineered": len(transformed_data.columns) - len(cleaned_data.columns) if cleaned_data is not None and transformed_data is not None else 0,
                "transformations_applied": len(getattr(final_state, 'transformation_log', [])),
                "quality_score": getattr(final_state, 'quality_score', 0),
                "statistical_tests_performed": len(analysis_results.get('hypothesis_tests', {}).get('group_comparisons', [])) if isinstance(analysis_results, dict) else 0,
                "correlations_identified": len(analysis_results.get('correlations', {}).get('significant_correlations', [])) if isinstance(analysis_results, dict) else 0,
                "insights_generated": len(insights.get('key_findings', [])) if isinstance(insights, dict) else 0,
                "recommendations_created": len(insights.get('policy_recommendations', [])) if isinstance(insights, dict) else 0,
                "execution_time_seconds": getattr(final_state, 'total_execution_time', 0)
            }
            
            results = {
                "mode": "full",
                "status": "completed" if not final_state.errors else "completed_with_warnings",
                "confidence_overall": confidence,
                "key_insights": key_insights,
                "pipeline_stats": pipeline_stats,
                "errors": final_state.errors,
                "artifacts_generated": self._get_generated_artifacts(),
                "final_state": {
                    "dataset_processed": final_state.dataset_path is not None,
                    "schema_inferred": final_state.schema_info is not None,
                    "data_cleaned": final_state.cleaned_data is not None,
                    "analysis_completed": final_state.analysis_results is not None,
                    "insights_generated": final_state.insights is not None,
                    "reports_created": final_state.reports is not None
                }
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Full pipeline failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                "mode": "full",
                "status": "failed",
                "error": str(e),
                "confidence_overall": "LOW",
                "errors": [str(e)]
            }

    def _calculate_overall_confidence(self, state: PipelineState) -> str:
        """Calculate overall confidence based on pipeline execution"""
        if state.errors:
            # Any errors reduce confidence
            if len(state.errors) > 3:
                return "LOW"
            return "MEDIUM"
        
        # Check quality score if available
        quality_score = getattr(state, 'quality_score', 100)
        
        if quality_score >= 85:
            return "HIGH"
        elif quality_score >= 70:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_generated_artifacts(self) -> List[str]:
        """Get list of artifacts that should be generated"""
        artifacts_dir = Path(self.run_manifest['artifacts_paths']['docs_dir'])
        
        expected_artifacts = [
            "run_manifest.json",
            "dataset_profile.json", 
            "schema_mapping.json",
            "cleaning_summary.json",
            "transformation_log.jsonl",
            "insights_executive.json"
        ]
        
        generated = []
        for artifact in expected_artifacts:
            if (artifacts_dir / artifact).exists():
                generated.append(artifact)
                
        return generated

    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline execution status"""
        return {
            "run_id": self.run_manifest['run_id'],
            "current_agent": self.state.current_agent,
            "errors": self.state.errors,
            "confidence_score": self.state.confidence_score,
            "pipeline_stage": self._get_pipeline_stage()
        }

    def _get_pipeline_stage(self) -> str:
        """Determine current pipeline stage based on state"""
        if self.state.reports:
            return "completed"
        elif self.state.insights:
            return "generating_reports"
        elif self.state.analysis_results:
            return "generating_insights"
        elif self.state.transformed_data is not None:
            return "analyzing"
        elif self.state.cleaned_data is not None:
            return "transforming"
        elif self.state.standardized_data is not None:
            return "cleaning"
        elif self.state.schema_info:
            return "standardizing"
        elif self.state.raw_data is not None:
            return "inferring_schema"
        else:
            return "ingesting"
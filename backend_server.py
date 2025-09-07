#!/usr/bin/env python3
"""
RTGS AI Analyst - Enhanced FastAPI Backend Server
Web API for frontend integration with comprehensive RTGS AI analysis pipeline
Matches CLI output structure and provides enhanced analysis results
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import yaml

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# Import RTGS components
from src.orchestrator.agent_router import RTGSOrchestrator
from src.agents.report_agent import EnhancedReportAgent, LLMAnalysisEngine
from src.utils.logging import setup_logging, get_logger

# Initialize FastAPI app
app = FastAPI(
    title="RTGS AI Analyst API",
    description="Enhanced Government Data Analysis Pipeline with LLM-powered AI Enhancement",
    version="2.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for state management
active_connections: List[WebSocket] = []
analysis_sessions: Dict[str, Dict[str, Any]] = {}
logger = get_logger(__name__)

# Enhanced Pydantic models for API
class AnalysisRequest(BaseModel):
    dataset_name: str
    domain: str = "auto"
    scope: str = "Regional Analysis"
    mode: str = "run"
    sample_rows: int = 500
    auto_approve: bool = True
    report_format: str = "pdf"
    interactive: bool = False

class EnhancedAnalysisConfig(BaseModel):
    business_questions: List[str] = []
    key_metrics: List[str] = []
    stakeholders: str = "Government officials"
    time_scope: str = "Not specified"
    geo_scope: str = "Not specified"
    analysis_focus: str = "balanced"
    description: str = "Government dataset for AI-enhanced policy analysis"

class EnhancedRunStatus(BaseModel):
    run_id: str
    status: str  # pending, running, completed, failed
    progress: int  # 0-100
    current_step: str
    phase: str  # data_processing, ai_analysis, finalizing
    message: str
    enhanced_summary: Optional[Dict[str, Any]] = None
    pipeline_stats: Optional[Dict[str, Any]] = None
    capabilities_demonstrated: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ComprehensiveResults(BaseModel):
    """Enhanced results structure matching CLI output"""
    run_manifest: Dict[str, Any]
    analysis_summary: Dict[str, Any]
    enhanced_capabilities: Dict[str, Any]
    ai_insights: Dict[str, Any]
    pipeline_results: Dict[str, Any]
    generated_outputs: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    next_steps: List[str]
    system_capabilities: Dict[str, Any]

# WebSocket connection manager
class EnhancedConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except:
            self.disconnect(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """Enhanced broadcast with structured data"""
        message_str = json.dumps(message)
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except:
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for conn in disconnected:
            self.disconnect(conn)

manager = EnhancedConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize the enhanced application"""
    setup_logging()
    logger.info("RTGS AI Analyst Enhanced API server starting up")
    
    # Create necessary directories
    directories = [
        "uploads", "artifacts", "data/raw", "data/standardized", 
        "data/cleaned", "data/transformed", "artifacts/reports",
        "artifacts/plots", "artifacts/docs", "artifacts/quick_start"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

@app.get("/")
async def root():
    """Enhanced root endpoint"""
    return {
        "message": "RTGS AI Analyst Enhanced API",
        "version": "2.0.0",
        "status": "running",
        "enhanced_features": {
            "ai_powered": True,
            "llm_engine": "claude-sonnet-4",
            "domain_detection": "automatic",
            "multi_agent_system": "langgraph_orchestrated",
            "comprehensive_visualizations": True
        },
        "endpoints": {
            "upload": "/upload",
            "analyze": "/analyze",
            "status": "/status/{run_id}",
            "results": "/results/{run_id}",
            "comprehensive_results": "/comprehensive/{run_id}",
            "download": "/download/{run_id}/{artifact_type}",
            "history": "/history",
            "websocket": "/ws",
            "capabilities": "/capabilities",
            "domain_detect": "/detect-domain"
        }
    }

@app.get("/capabilities")
async def get_capabilities():
    """Get system capabilities information"""
    return {
        "dual_mode_analysis": {
            "traditional_pipeline": "Data cleaning, transformation & statistical analysis",
            "llm_enhancement": "AI pattern recognition & policy recommendations",
            "multi_agent_system": "LangGraph orchestration with 8+ specialized agents",
            "domain_adaptive": "Automatically adapts to any government data domain"
        },
        "ai_capabilities": {
            "domain_detection": "AI-powered automatic domain identification",
            "pattern_recognition": "Advanced AI analysis of data relationships",
            "policy_insights": "Context-aware recommendations for government action",
            "multi_format_reports": "Technical analysis + Policy briefs + Interactive dashboards"
        },
        "supported_domains": [
            "transport", "health", "education", "economics", 
            "agriculture", "environment", "urban", "social", "auto"
        ],
        "output_formats": ["PDF", "HTML", "Markdown", "Interactive Dashboard"],
        "audience_types": ["Technical teams", "Policy makers", "Executives"]
    }

@app.post("/detect-domain")
async def detect_domain_endpoint(file: UploadFile = File(...)):
    """AI-powered domain detection endpoint"""
    try:
        # Save temporary file
        temp_path = f"uploads/temp_{uuid.uuid4()}.csv"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Initialize LLM engine for domain detection
        llm_engine = LLMAnalysisEngine()
        
        # Detect domain using AI
        detected_domain = await detect_domain_with_ai(temp_path, llm_engine)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return {
            "detected_domain": detected_domain,
            "confidence": "high" if detected_domain != "general" else "low",
            "message": f"AI detected domain: {detected_domain.title()}"
        }
        
    except Exception as e:
        logger.error(f"Domain detection failed: {str(e)}")
        return {
            "detected_domain": "general",
            "confidence": "low", 
            "error": f"Detection failed: {str(e)}"
        }

async def detect_domain_with_ai(dataset_path: str, llm_engine) -> str:
    """AI domain detection matching CLI implementation"""
    try:
        # Load sample for analysis
        df = pd.read_csv(dataset_path, nrows=100)
        
        # Prepare context for AI analysis
        column_info = {
            'columns': list(df.columns),
            'sample_data': {}
        }
        
        for col in df.columns[:10]:
            if df[col].dtype == 'object':
                unique_vals = df[col].dropna().unique()[:5]
                column_info['sample_data'][col] = [str(val) for val in unique_vals]
            else:
                stats = {
                    'min': float(df[col].min()) if pd.notnull(df[col].min()) else None,
                    'max': float(df[col].max()) if pd.notnull(df[col].max()) else None,
                    'mean': float(df[col].mean()) if pd.notnull(df[col].mean()) else None
                }
                column_info['sample_data'][col] = [f"Numeric: {stats}"]
        
        prompt = f"""You are an expert government data analyst. Analyze this dataset and determine the most likely domain.

DATASET ANALYSIS:
- Filename: {Path(dataset_path).name}
- Columns: {column_info['columns']}
- Sample Data: {json.dumps(column_info['sample_data'], indent=2)}

DOMAIN OPTIONS:
- health: Healthcare services, medical data, public health metrics
- education: Schools, students, learning outcomes, educational infrastructure
- transport: Roads, vehicles, traffic, public transportation, connectivity
- economics: Employment, business, GDP, economic indicators, finance
- agriculture: Farming, crops, livestock, rural development, food security
- environment: Pollution, climate, natural resources, sustainability
- urban: City planning, utilities, municipal services, urban development
- social: Demographics, welfare, social services, community programs

Based on the column names and data patterns, respond with just the domain name (one word) that best matches this dataset.

Domain:"""
        
        result = await llm_engine.call_llm(prompt, 100)
        detected_domain = result.strip().lower()
        
        valid_domains = ['health', 'education', 'transport', 'economics', 
                       'agriculture', 'environment', 'urban', 'social']
        
        return detected_domain if detected_domain in valid_domains else 'general'
        
    except Exception as e:
        logger.warning(f"AI domain detection failed: {e}")
        return 'general'

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Enhanced file upload with AI domain detection"""
    try:
        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        filename = f"{file_id}{file_extension}"
        file_path = f"uploads/{filename}"
        
        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Basic file validation and AI domain detection
        try:
            if file_extension == '.csv':
                df = pd.read_csv(file_path, nrows=5)
                full_df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, nrows=5)
                full_df = pd.read_excel(file_path)
            
            # AI domain detection
            llm_engine = LLMAnalysisEngine()
            detected_domain = await detect_domain_with_ai(file_path, llm_engine)
            
            file_info = {
                "file_id": file_id,
                "filename": file.filename,
                "file_path": file_path,
                "rows": len(full_df),
                "columns": list(df.columns),
                "detected_domain": detected_domain,
                "upload_time": datetime.now().isoformat(),
                "ai_analysis": {
                    "domain_confidence": "high" if detected_domain != "general" else "low",
                    "sample_analysis": f"Dataset contains {len(df.columns)} columns and {len(full_df)} rows"
                }
            }
            
            logger.info(f"File uploaded successfully with AI analysis: {file_info['filename']}")
            return file_info
            
        except Exception as e:
            # Clean up invalid file
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"Invalid file format: {str(e)}")
            
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/analyze")
async def start_analysis(
    file_id: str,
    config: EnhancedAnalysisConfig,
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Start enhanced analysis matching CLI capabilities"""
    try:
        
        # Find uploaded file
        upload_dir = Path("uploads")
        file_path = None
        for file in upload_dir.glob(f"{file_id}.*"):
            file_path = file
            break
        
        if not file_path or not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found for file_id: {file_id}")
        
        # Generate run ID matching CLI format
        run_id = f"rtgs-enhanced-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"
        
        # Create enhanced analysis session
        analysis_sessions[run_id] = {
            "run_id": run_id,
            "file_path": str(file_path),
            "filename": file_path.name,
            "config": config.dict(),
            "request": request.dict(),
            "status": "pending",
            "progress": 0,
            "current_step": "Initializing",
            "phase": "setup",
            "message": "Enhanced analysis queued",
            "start_time": datetime.now().isoformat(),
            "enhanced_summary": {},
            "pipeline_stats": {},
            "capabilities_demonstrated": {},
            "results": None,
            "error": None
        }
        
        # Start enhanced background analysis
        background_tasks.add_task(run_enhanced_analysis_pipeline, run_id)
        
        logger.info(f"Enhanced analysis started for run_id: {run_id}, file: {file_path.name}")
        return {"run_id": run_id, "status": "started", "enhanced": True}
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Enhanced analysis start failed: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "message": "Enhanced analysis failed to start",
            "error": str(e),
            "debug_info": {
                "received_data_type": type(analysis_data).__name__,
                "data_keys": list(analysis_data.keys()) if isinstance(analysis_data, dict) else "not_dict"
            }
        })

# Add a debug endpoint to help troubleshoot request format issues
@app.post("/debug/analyze")
async def debug_analyze_request(request_data: dict):
    """Debug endpoint to help troubleshoot request format issues"""
    return {
        "received_data": request_data,
        "data_type": type(request_data).__name__,
        "keys": list(request_data.keys()) if isinstance(request_data, dict) else "not_dict",
        "expected_format": {
            "file_id": "string - required",
            "config": {
                "business_questions": "list[str] - optional",
                "key_metrics": "list[str] - optional", 
                "stakeholders": "str - optional",
                "time_scope": "str - optional",
                "geo_scope": "str - optional",
                "analysis_focus": "str - optional",
                "description": "str - optional"
            },
            "request": {
                "dataset_name": "str - optional",
                "domain": "str - optional (default: auto)",
                "scope": "str - optional",
                "mode": "str - optional (default: run)",
                "sample_rows": "int - optional (default: 500)",
                "auto_approve": "bool - optional (default: true)",
                "report_format": "str - optional (default: pdf)",
                "interactive": "bool - optional (default: false)"
            }
        }
    }

# Alternative simplified endpoint for basic analysis
@app.post("/analyze/simple")
async def start_simple_analysis(
    file_id: str,
    domain: str = "auto",
    scope: str = "Regional Analysis",
    mode: str = "run",
    background_tasks: BackgroundTasks = None
):
    """Simplified analysis endpoint with minimal parameters"""
    try:
        # Create default config and request
        analysis_data = {
            "file_id": file_id,
            "config": {
                "business_questions": [],
                "key_metrics": [],
                "stakeholders": "Government officials",
                "description": "Government dataset for AI-enhanced policy analysis"
            },
            "request": {
                "dataset_name": f"dataset_{file_id}",
                "domain": domain,
                "scope": scope,
                "mode": mode,
                "sample_rows": 500,
                "auto_approve": True,
                "report_format": "pdf",
                "interactive": False
            }
        }
        
        # Call the main analysis endpoint
        return await start_analysis(analysis_data, background_tasks)
        
    except Exception as e:
        logger.error(f"Simple analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Simple analysis failed: {str(e)}")

async def run_enhanced_analysis_pipeline(run_id: str):
    """Run the complete enhanced RTGS analysis pipeline matching CLI output"""
    try:
        session = analysis_sessions[run_id]
        session["status"] = "running"
        session["phase"] = "data_processing"
        session["current_step"] = "Setting up enhanced pipeline"
        session["progress"] = 10
        
        # Broadcast initial progress
        await manager.broadcast({
            "run_id": run_id,
            "status": "running",
            "phase": "data_processing",
            "progress": 10,
            "current_step": "Setting up enhanced pipeline",
            "message": "🚀 RTGS AI ANALYST - Enhanced Pipeline Starting",
            "capabilities": {
                "llm_enhanced": True,
                "ai_powered": True,
                "domain_adaptive": True
            }
        })
        
        # Create enhanced run manifest
        run_manifest = create_enhanced_run_manifest(session)
        
        # Setup comprehensive output directories
        setup_enhanced_output_directories(run_manifest)
        
        session["progress"] = 20
        session["current_step"] = "Data Processing Pipeline"
        await manager.broadcast({
            "run_id": run_id,
            "progress": 20,
            "current_step": "🔄 Phase 1: Data Processing Pipeline (LangGraph)",
            "message": "Multi-agent orchestrator initialized"
        })
        
        # Initialize enhanced orchestrator
        orchestrator = RTGSOrchestrator(run_manifest)
        
        # Run traditional pipeline based on mode
        if session["request"]["mode"] == "dry-run":
            result = await orchestrator.dry_run()
        elif session["request"]["mode"] == "preview":
            result = await orchestrator.preview_run()
            if not session["request"]["auto_approve"]:
                result = await orchestrator.full_run()
        else:
            result = await orchestrator.full_run()
        
        # Update pipeline stats
        session["pipeline_stats"] = {
            "rows_processed": result.get("rows_processed", "N/A"),
            "cleaned_columns": result.get("cleaned_columns", "N/A"),
            "features_engineered": result.get("features_engineered", "N/A"),
            "statistical_tests_performed": result.get("statistical_tests_performed", "N/A")
        }
        
        session["progress"] = 60
        session["phase"] = "ai_analysis"
        session["current_step"] = "LLM-Enhanced Analysis"
        await manager.broadcast({
            "run_id": run_id,
            "progress": 60,
            "phase": "ai_analysis",
            "current_step": "🧠 Phase 2: LLM-Enhanced Analysis (Claude Sonnet 4)",
            "message": "Running AI pattern analysis..."
        })
        
        # Run enhanced analysis
        enhanced_report_agent = EnhancedReportAgent()
        
        # Create enhanced state for analysis
        enhanced_state = create_enhanced_state(run_manifest, result)
        enhanced_result = await enhanced_report_agent.process(enhanced_state)
        
        session["progress"] = 90
        session["phase"] = "finalizing"
        session["current_step"] = "Finalizing Enhanced Results"
        await manager.broadcast({
            "run_id": run_id,
            "progress": 90,
            "phase": "finalizing",
            "current_step": "💾 Phase 3: Finalizing Results",
            "message": "Generating comprehensive reports and finalizing results"
        })
        
        # Create comprehensive results matching CLI output
        comprehensive_results = create_comprehensive_results(
            run_manifest, result, enhanced_result, session
        )
        
        # Update session with enhanced results
        session["results"] = comprehensive_results
        session["enhanced_summary"] = getattr(enhanced_result, 'cli_summary', {})
        session["capabilities_demonstrated"] = {
            "domain_detection": run_manifest.get('run_config', {}).get('enable_domain_detection', False),
            "pattern_recognition": "Advanced AI analysis of data relationships",
            "policy_insights": "Context-aware recommendations for government action",
            "multi_format_reports": "Technical analysis + Policy briefs + Interactive dashboards"
        }
        
        session["status"] = "completed"
        session["progress"] = 100
        session["current_step"] = "Enhanced Analysis Completed"
        session["message"] = "🎯 RTGS AI ANALYST - ENHANCED ANALYSIS COMPLETED"
        
        # Final comprehensive broadcast
        await manager.broadcast({
            "run_id": run_id,
            "status": "completed",
            "progress": 100,
            "current_step": "🎯 Enhanced Analysis Completed",
            "message": "AI-powered insights ready for government decision-making!",
            "enhanced_summary": session["enhanced_summary"],
            "capabilities_demonstrated": session["capabilities_demonstrated"]
        })
        
        logger.info(f"Enhanced analysis completed for run_id: {run_id}")
        
    except Exception as e:
        logger.error(f"Enhanced analysis failed for run_id {run_id}: {str(e)}")
        session = analysis_sessions[run_id]
        session["status"] = "failed"
        session["error"] = str(e)
        session["message"] = f"Enhanced analysis failed: {str(e)}"
        
        await manager.broadcast({
            "run_id": run_id,
            "status": "failed",
            "error": str(e),
            "message": f"Enhanced analysis failed: {str(e)}"
        })

def create_enhanced_run_manifest(session: Dict[str, Any]) -> Dict[str, Any]:
    """Create enhanced run manifest matching CLI structure"""
    run_id = session["run_id"]
    file_path = session["file_path"]
    config = session["config"]
    request = session["request"]
    
    # Parse dataset context if available
    dataset_context = config
    
    manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "dataset_info": {
            "source_path": file_path,
            "dataset_name": Path(file_path).stem,
            "domain_hint": request["domain"],
            "scope": request["scope"],
            "description": config.get("description", "Web-uploaded dataset for AI-enhanced analysis")
        },
        "user_context": {
            "business_questions": config["business_questions"],
            "key_metrics": config["key_metrics"],
            "stakeholders": config["stakeholders"],
            "time_scope": config["time_scope"],
            "geo_scope": config["geo_scope"]
        },
        "run_config": {
            "mode": request["mode"],
            "sample_rows": request["sample_rows"],
            "auto_approve": request["auto_approve"],
            "output_dir": f"artifacts/{run_id}",
            "report_format": request["report_format"],
            "llm_enhanced": True,
            "enable_domain_detection": request["domain"] == "auto"
        },
        "agent_version_tags": {
            "orchestrator": "v1.0",
            "ingestion": "v1.0", 
            "schema": "v1.0",
            "cleaning": "v1.0",
            "analysis": "v1.0",
            "insight": "v1.0",
            "enhanced_report": "v2.0-llm",
            "llm_engine": "claude-sonnet-4"
        },
        "llm_analysis": {
            "enabled": True,
            "domain_detection": "auto" if request["domain"] == "auto" else "manual",
            "analysis_depth": "comprehensive",
            "policy_focus": True
        },
        "artifacts_paths": {},
        "confidence_overall": "PENDING",
        "notes": []
    }
    
    return manifest

def setup_enhanced_output_directories(run_manifest: Dict[str, Any]) -> None:
    """Create comprehensive output directory structure matching CLI"""
    base_dir = Path(run_manifest["run_config"]["output_dir"])
    
    directories = [
        base_dir / "artifacts" / "logs",
        base_dir / "artifacts" / "reports", 
        base_dir / "artifacts" / "plots" / "interactive",
        base_dir / "artifacts" / "plots" / "static",
        base_dir / "artifacts" / "docs",
        base_dir / "artifacts" / "quick_start",
        base_dir / "artifacts" / "llm_analysis",
        base_dir / "data" / "raw",
        base_dir / "data" / "standardized", 
        base_dir / "data" / "cleaned",
        base_dir / "data" / "transformed"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        
    # Update manifest with comprehensive artifact paths
    artifacts_base = base_dir / "artifacts"
    data_base = base_dir / "data"
    run_manifest["artifacts_paths"] = {
        "logs_dir": str(artifacts_base / "logs"),
        "reports_dir": str(artifacts_base / "reports"),
        "plots_dir": str(artifacts_base / "plots"),
        "docs_dir": str(artifacts_base / "docs"),
        "quick_start_dir": str(artifacts_base / "quick_start"),
        "llm_analysis_dir": str(artifacts_base / "llm_analysis"),
        "data_dir": str(data_base),
        "run_manifest": str(artifacts_base / "docs" / "run_manifest.json")
    }

def create_enhanced_state(run_manifest: Dict[str, Any], traditional_results: Dict[str, Any]):
    """Create enhanced state for analysis matching CLI structure"""
    class EnhancedState:
        def __init__(self, manifest, traditional_results):
            self.run_manifest = manifest
            
            # Load processed data
            try:
                cleaned_data_path = Path(manifest['artifacts_paths']['data_dir']) / 'cleaned' / f"{manifest['dataset_info']['dataset_name']}_cleaned.csv"
                if cleaned_data_path.exists():
                    self.transformed_data = pd.read_csv(cleaned_data_path)
                else:
                    self.transformed_data = pd.read_csv(manifest['dataset_info']['source_path'])
            except:
                self.transformed_data = pd.DataFrame()
            
            self.analysis_results = traditional_results.get('analysis_results', {})
            self.insights = traditional_results.get('insights', {})
            self.errors = []
            self.warnings = []
    
    return EnhancedState(run_manifest, traditional_results)

def create_comprehensive_results(run_manifest: Dict[str, Any], 
                                traditional_results: Dict[str, Any],
                                enhanced_result, 
                                session: Dict[str, Any]) -> Dict[str, Any]:
    """Create comprehensive results structure matching CLI output format"""
    
    enhanced_summary = getattr(enhanced_result, 'cli_summary', {})
    llm_reports = getattr(enhanced_result, 'llm_enhanced_reports', {})
    
    return {
        "run_manifest": run_manifest,
        "analysis_summary": {
            "dataset_info": run_manifest.get('dataset_info', {}),
            "confidence": enhanced_summary.get('confidence_badge', '🟡 MEDIUM'),
            "quality_score": enhanced_summary.get('quality_score', '75/100'),
            "patterns_found": enhanced_summary.get('findings_count', 0),
            "actions_identified": enhanced_summary.get('actions_count', 0)
        },
        "enhanced_capabilities": {
            "dual_mode_analysis": {
                "traditional_pipeline": "✅ Data cleaning, transformation & statistical analysis",
                "llm_enhancement": "🧠 AI pattern recognition & policy recommendations",
                "multi_agent_system": "📊 LangGraph orchestration with 8+ specialized agents",
                "domain_adaptive": "🎯 Automatically adapts to any government data domain"
            },
            "ai_capabilities_demonstrated": {
                "domain_detection": 'AI-powered' if run_manifest.get('run_config', {}).get('enable_domain_detection') else 'Manual',
                "pattern_recognition": "Advanced AI analysis of data relationships",
                "policy_insights": "Context-aware recommendations for government action",
                "multi_format_reports": "Technical analysis + Policy briefs + Interactive dashboards"
            }
        },
        "ai_insights": {
            "key_findings": enhanced_summary.get('key_findings', []),
            "priority_actions": enhanced_summary.get('priority_actions', []),
            "llm_powered": enhanced_summary.get('llm_powered', True)
        },
        "pipeline_results": {
            "data_processing": session.get("pipeline_stats", {}),
            "traditional_results": traditional_results,
            "confidence_overall": run_manifest.get('confidence_overall', 'PENDING')
        },
        "generated_outputs": {
            "traditional_analysis": {
                "technical_report": f"{run_manifest['artifacts_paths']['reports_dir']}/technical_report.md",
                "analysis_results": f"{run_manifest['artifacts_paths']['docs_dir']}/analysis_results.json"
            },
            "llm_enhanced_analysis": llm_reports,
            "quick_start": {
                "executive_summary": f"{run_manifest['artifacts_paths']['quick_start_dir']}/key_outputs_summary.html",
                "demo_script": f"{run_manifest['artifacts_paths']['quick_start_dir']}/demo_script.md"
            }
        },
        "quality_metrics": {
            "overall_confidence": enhanced_summary.get('confidence_badge', 'MEDIUM'),
            "analysis_depth": "comprehensive",
            "llm_enhancement": "enabled"
        },
        "next_steps": [
            "📊 Review AI insights in the policy dashboard",
            "🔍 Validate findings with domain experts",
            "📋 Prioritize actions based on urgency and feasibility",
            "🚀 Begin implementation of immediate actions",
            "📈 Set up monitoring for recommended KPIs"
        ],
        "system_capabilities": {
            "data_agnostic": "✅ Works with any government dataset structure",
            "domain_adaptive": "✅ Automatically provides sector-specific insights",
            "llm_enhanced": "✅ AI-powered pattern recognition and policy recommendations",
            "production_ready": "✅ Complete audit trail and quality assurance",
            "multi_audience": "✅ Reports for technical teams, policy makers, and executives"
        }
    }

@app.get("/status/{run_id}")
async def get_enhanced_status(run_id: str):
    """Get enhanced analysis status with comprehensive details"""
    if run_id not in analysis_sessions:
        raise HTTPException(status_code=404, detail="Run not found")
    
    session = analysis_sessions[run_id]
    return EnhancedRunStatus(**session)

@app.get("/results/{run_id}")
async def get_results(run_id: str):
    """Get analysis results (legacy endpoint)"""
    if run_id not in analysis_sessions:
        raise HTTPException(status_code=404, detail="Run not found")
    
    session = analysis_sessions[run_id]
    if session["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")
    
    return session["results"]

@app.get("/comprehensive/{run_id}")
async def get_comprehensive_results(run_id: str):
    """Get comprehensive analysis results matching CLI output structure"""
    if run_id not in analysis_sessions:
        raise HTTPException(status_code=404, detail="Run not found")
    
    session = analysis_sessions[run_id]
    if session["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")
    
    # Return comprehensive results in CLI-matching structure
    return ComprehensiveResults(**session["results"])

@app.get("/download/{run_id}/{artifact_type}")
async def download_artifact(run_id: str, artifact_type: str):
    """Download specific artifacts (PDFs, reports, etc.)"""
    if run_id not in analysis_sessions:
        raise HTTPException(status_code=404, detail="Run not found")
    
    session = analysis_sessions[run_id]
    if session["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")
    
    results = session["results"]
    generated_outputs = results.get("generated_outputs", {})
    
    # Map artifact types to file paths
    artifact_map = {
        "technical_pdf": generated_outputs.get("llm_enhanced_analysis", {}).get("technical_quality_pdf"),
        "policy_pdf": generated_outputs.get("llm_enhanced_analysis", {}).get("policy_focused_pdf"),
        "dashboard": generated_outputs.get("llm_enhanced_analysis", {}).get("interactive_dashboard"),
        "executive_summary": generated_outputs.get("quick_start", {}).get("executive_summary"),
        "technical_report": generated_outputs.get("traditional_analysis", {}).get("technical_report"),
        "manifest": session["results"]["run_manifest"]["artifacts_paths"]["run_manifest"]
    }
    
    if artifact_type not in artifact_map:
        raise HTTPException(status_code=400, detail=f"Unknown artifact type: {artifact_type}")
    
    file_path = artifact_map[artifact_type]
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_type}")
    
    # Determine content type
    content_types = {
        "technical_pdf": "application/pdf",
        "policy_pdf": "application/pdf", 
        "dashboard": "text/html",
        "executive_summary": "text/html",
        "technical_report": "text/markdown",
        "manifest": "application/json"
    }
    
    return FileResponse(
        file_path, 
        media_type=content_types.get(artifact_type, "application/octet-stream"),
        filename=f"{run_id}_{artifact_type}.{Path(file_path).suffix[1:]}"
    )

@app.get("/history")
async def get_enhanced_history():
    """Get enhanced analysis history with comprehensive details"""
    return {
        "sessions": [
            {
                "run_id": run_id,
                "filename": session.get("filename", session.get("file_path", "").split("/")[-1]),
                "dataset_name": session.get("results", {}).get("run_manifest", {}).get("dataset_info", {}).get("dataset_name", "Unknown"),
                "domain": session.get("results", {}).get("run_manifest", {}).get("dataset_info", {}).get("domain_hint", "General"),
                "status": session["status"],
                "start_time": session["start_time"],
                "progress": session["progress"],
                "enhanced": True,
                "ai_confidence": session.get("enhanced_summary", {}).get("confidence_badge", "MEDIUM"),
                "patterns_found": session.get("enhanced_summary", {}).get("findings_count", 0),
                "actions_identified": session.get("enhanced_summary", {}).get("actions_count", 0)
            }
            for run_id, session in analysis_sessions.items()
        ],
        "total_analyses": len(analysis_sessions),
        "enhanced_features": True
    }

@app.get("/summary/{run_id}")
async def get_executive_summary(run_id: str):
    """Get executive summary matching CLI display format"""
    if run_id not in analysis_sessions:
        raise HTTPException(status_code=404, detail="Run not found")
    
    session = analysis_sessions[run_id]
    if session["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")
    
    results = session["results"]
    enhanced_summary = session.get("enhanced_summary", {})
    
    return {
        "executive_header": {
            "title": "RTGS AI ANALYST - ENHANCED ANALYSIS COMPLETED",
            "system_name": "RTGS AI ANALYST",
            "version": "v2.0.0 - LLM Enhanced"
        },
        "dual_mode_capabilities": {
            "traditional_pipeline": "Data cleaning, transformation & statistical analysis",
            "llm_enhancement": "AI pattern recognition & policy recommendations", 
            "multi_agent_system": "LangGraph orchestration with 8+ specialized agents",
            "domain_adaptive": "Automatically adapts to any government data domain"
        },
        "analysis_summary": results.get("analysis_summary", {}),
        "key_insights": {
            "ai_generated_insights": enhanced_summary.get("key_findings", []),
            "priority_actions": enhanced_summary.get("priority_actions", [])
        },
        "pipeline_performance": session.get("pipeline_stats", {}),
        "enhanced_capabilities_demonstrated": session.get("capabilities_demonstrated", {}),
        "comprehensive_outputs": results.get("generated_outputs", {}),
        "next_steps": results.get("next_steps", []),
        "system_capabilities_summary": results.get("system_capabilities", {})
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            
            # Handle specific websocket commands if needed
            try:
                message = json.loads(data)
                if message.get("type") == "subscribe" and message.get("run_id"):
                    # Could implement run-specific subscriptions here
                    await websocket.send_text(json.dumps({
                        "type": "subscription_confirmed",
                        "run_id": message["run_id"]
                    }))
            except json.JSONDecodeError:
                # Ignore non-JSON messages
                pass
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "enhanced_features": True,
        "active_sessions": len(analysis_sessions),
        "websocket_connections": len(manager.active_connections),
        "timestamp": datetime.now().isoformat()
    }

# Static file serving for production
if Path("Frontend/dist").exists():
    app.mount("/static", StaticFiles(directory="Frontend/dist"), name="static")

    @app.get("/app")
    async def serve_frontend():
        """Serve the frontend application"""
        return FileResponse("Frontend/dist/index.html")
else:
    @app.get("/app")
    async def redirect_to_frontend():
        """Redirect to frontend development server"""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="http://localhost:5173")

if __name__ == "__main__":
    uvicorn.run(
        "enhanced_backend_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
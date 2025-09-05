"""
RTGS AI Analyst - Logging Configuration
Centralized logging setup for all agents and components
"""

import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler
import uuid


class RTGSFormatter(logging.Formatter):
    """Custom formatter for RTGS logs with structured output"""
    
    def __init__(self):
        super().__init__()
        
    def format(self, record):
        # Create structured log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'agent_name'):
            log_entry['agent'] = record.agent_name
        if hasattr(record, 'run_id'):
            log_entry['run_id'] = record.run_id
        if hasattr(record, 'execution_time'):
            log_entry['execution_time_ms'] = record.execution_time
        if hasattr(record, 'data_shape'):
            log_entry['data_shape'] = record.data_shape
            
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry, ensure_ascii=False)


class RTGSFileHandler(RotatingFileHandler):
    """Custom file handler that creates log directory if needed"""
    
    def __init__(self, filename, mode='a', maxBytes=10*1024*1024, backupCount=5):
        # Ensure log directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        super().__init__(filename, mode, maxBytes, backupCount)


class TransformLogger:
    """Specialized logger for transformation operations"""
    
    def __init__(self, log_file: str, run_id: str):
        self.log_file = Path(log_file)
        self.run_id = run_id
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
    def log_transform(self, 
                     agent: str,
                     action: str,
                     column: Optional[str] = None,
                     rows_affected: Optional[int] = None,
                     rule_id: Optional[str] = None,
                     rationale: Optional[str] = None,
                     parameters: Optional[Dict] = None,
                     confidence: str = "medium",
                     preview_before: Optional[str] = None,
                     preview_after: Optional[str] = None):
        """Log a transformation action in JSONL format"""
        
        entry = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "run_id": self.run_id,
            "agent": agent,
            "action": action,
            "column": column,
            "rows_affected": rows_affected,
            "rule_id": rule_id or f"{action}_{uuid.uuid4().hex[:8]}",
            "rationale": rationale,
            "parameters": parameters or {},
            "confidence": confidence,
            "preview_sample_before": preview_before,
            "preview_sample_after": preview_after
        }
        
        # Append to JSONL file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    def log_quality_check(self,
                         check_name: str,
                         passed: bool,
                         details: Dict[str, Any],
                         remediation: Optional[str] = None):
        """Log a data quality check result"""
        
        entry = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z", 
            "run_id": self.run_id,
            "agent": "validator",
            "action": "quality_check",
            "check_name": check_name,
            "passed": passed,
            "details": details,
            "remediation": remediation
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def setup_logging(log_level: str = "INFO", 
                 log_dir: str = "artifacts/logs",
                 enable_console: bool = True,
                 enable_file: bool = True) -> None:
    """Setup centralized logging configuration"""
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with simple format for CLI
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with structured JSON format
    if enable_file:
        file_handler = RTGSFileHandler(
            filename=str(log_path / "rtgs_pipeline.log"),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(RTGSFormatter())
        root_logger.addHandler(file_handler)
    
    # Separate error log
    if enable_file:
        error_handler = RTGSFileHandler(
            filename=str(log_path / "rtgs_errors.log"),
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(RTGSFormatter())
        root_logger.addHandler(error_handler)
    
    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with optional agent context"""
    return logging.getLogger(name)


def get_agent_logger(agent_name: str, run_id: Optional[str] = None) -> logging.Logger:
    """Get a logger for a specific agent with context"""
    logger = logging.getLogger(f"rtgs.agent.{agent_name}")
    
    # Add agent context to all log records
    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.agent_name = agent_name
        if run_id:
            record.run_id = run_id
        return record
    
    logging.setLogRecordFactory(record_factory)
    
    return logger


def log_execution_time(func):
    """Decorator to log function execution time"""
    def wrapper(*args, **kwargs):
        start_time = datetime.utcnow()
        logger = get_logger(f"{func.__module__}.{func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            logger.info(
                f"Function {func.__name__} completed",
                extra={"execution_time": execution_time}
            )
            
            return result
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error(
                f"Function {func.__name__} failed: {str(e)}",
                extra={"execution_time": execution_time},
                exc_info=True
            )
            raise
            
    return wrapper


def log_data_shape(data, logger: logging.Logger, operation: str):
    """Log the shape/size of data being processed"""
    try:
        if hasattr(data, 'shape'):
            # Pandas DataFrame or similar
            shape_info = {
                "rows": data.shape[0], 
                "columns": data.shape[1],
                "memory_mb": round(data.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            }
        elif hasattr(data, '__len__'):
            # List or similar
            shape_info = {"length": len(data)}
        else:
            shape_info = {"type": type(data).__name__}
        
        logger.info(
            f"Data shape for {operation}",
            extra={"data_shape": shape_info}
        )
        
    except Exception:
        # Don't fail if we can't determine shape
        pass


class PerformanceTracker:
    """Track performance metrics across pipeline execution"""
    
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.metrics = {}
        self.start_times = {}
        
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = datetime.utcnow()
        
    def end_timer(self, operation: str) -> float:
        """End timing and return duration in seconds"""
        if operation not in self.start_times:
            return 0.0
            
        duration = (datetime.utcnow() - self.start_times[operation]).total_seconds()
        self.metrics[operation] = duration
        
        return duration
    
    def log_metric(self, metric_name: str, value: Any):
        """Log a custom metric"""
        self.metrics[metric_name] = value
        
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        total_time = sum(v for k, v in self.metrics.items() if k.endswith('_time'))
        
        return {
            "run_id": self.run_id,
            "total_execution_time": total_time,
            "detailed_metrics": self.metrics,
            "summary_generated_at": datetime.utcnow().isoformat()
        }
    
    def save_to_file(self, file_path: str):
        """Save performance metrics to file"""
        with open(file_path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)


# Global performance tracker instance
_performance_tracker = None

def get_performance_tracker(run_id: str) -> PerformanceTracker:
    """Get global performance tracker instance"""
    global _performance_tracker
    if _performance_tracker is None or _performance_tracker.run_id != run_id:
        _performance_tracker = PerformanceTracker(run_id)
    return _performance_tracker
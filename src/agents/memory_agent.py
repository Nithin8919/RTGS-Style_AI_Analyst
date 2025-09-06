"""
RTGS AI Analyst - Memory Agent
Handles long-term memory, schema reuse, and learning from past pipeline runs
"""

import pandas as pd
import numpy as np
import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pickle

from src.utils.logging import get_agent_logger


class MemoryAgent:
    """Agent responsible for long-term memory and learning from past runs"""
    
    def __init__(self, config_path: str = "config.yaml", memory_dir: str = "memory/store"):
        self.logger = get_agent_logger("memory")
        
        # Set up memory storage directory
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite database for metadata
        self.db_path = self.memory_dir / "memory.db"
        self._initialize_database()
        
        # Memory storage files
        self.schema_cache_path = self.memory_dir / "schema_cache.json"
        self.transform_rules_path = self.memory_dir / "transform_rules.json"
        self.run_history_path = self.memory_dir / "run_history.json"
        
        # Load existing memory
        self.schema_cache = self._load_json_file(self.schema_cache_path, {})
        self.transform_rules = self._load_json_file(self.transform_rules_path, {})
        self.run_history = self._load_json_file(self.run_history_path, [])
        
    def _initialize_database(self):
        """Initialize SQLite database for memory storage"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Runs metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    dataset_name TEXT,
                    domain TEXT,
                    schema_hash TEXT,
                    sample_hash TEXT,
                    timestamp TEXT,
                    confidence_score REAL,
                    quality_score REAL,
                    rows_processed INTEGER,
                    columns_processed INTEGER,
                    success INTEGER
                )
            """)
            
            # Schema mappings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_mappings (
                    schema_hash TEXT,
                    original_column TEXT,
                    canonical_column TEXT,
                    suggested_type TEXT,
                    confidence REAL,
                    usage_count INTEGER DEFAULT 1,
                    last_used TEXT,
                    PRIMARY KEY (schema_hash, original_column)
                )
            """)
            
            # Transform rules table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transform_rules (
                    rule_id TEXT PRIMARY KEY,
                    rule_type TEXT,
                    column_pattern TEXT,
                    action TEXT,
                    parameters TEXT,
                    success_rate REAL,
                    usage_count INTEGER DEFAULT 1,
                    created_at TEXT,
                    last_used TEXT
                )
            """)
            
            # Quality patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quality_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    domain TEXT,
                    issue_type TEXT,
                    pattern_description TEXT,
                    remediation TEXT,
                    effectiveness_score REAL,
                    usage_count INTEGER DEFAULT 1,
                    created_at TEXT
                )
            """)
            
            conn.commit()
    
    def _load_json_file(self, file_path: Path, default: Any) -> Any:
        """Load JSON file with fallback to default"""
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load {file_path}: {str(e)}")
        return default
    
    def _save_json_file(self, file_path: Path, data: Any):
        """Save data to JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save {file_path}: {str(e)}")
    
    async def process(self, state) -> Any:
        """Main memory processing - store current run and retrieve relevant history"""
        self.logger.info("Processing memory operations")
        
        try:
            # Extract run information
            run_manifest = state.run_manifest
            
            # Store current run
            await self._store_current_run(state)
            
            # Store schema mappings
            canonical_mapping = getattr(state, 'canonical_mapping', {})
            if canonical_mapping:
                await self._store_schema_mappings(canonical_mapping, state)
            
            # Store successful transform rules
            transformation_log = getattr(state, 'transformation_log', [])
            if transformation_log:
                await self._store_transform_rules(transformation_log, state)
            
            # Store quality patterns
            validation_report = getattr(state, 'validation_report', {})
            if validation_report:
                await self._store_quality_patterns(validation_report, state)
            
            # Generate memory insights
            memory_insights = await self._generate_memory_insights(state)
            
            # Update state with memory information
            state.memory_insights = memory_insights
            state.similar_runs = await self._find_similar_runs(state)
            
            self.logger.info("Memory processing completed")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Memory processing failed: {str(e)}")
            state.errors.append(f"Memory processing failed: {str(e)}")
            return state
    
    async def _store_current_run(self, state):
        """Store current run metadata"""
        
        run_manifest = state.run_manifest
        analysis_results = getattr(state, 'analysis_results', {})
        validation_report = getattr(state, 'validation_report', {})
        
        # Calculate hashes
        transformed_data = getattr(state, 'transformed_data', pd.DataFrame())
        schema_hash = self._calculate_schema_hash(transformed_data)
        sample_hash = self._calculate_sample_hash(transformed_data)
        
        # Determine success status
        success = 1 if len(state.errors) == 0 else 0
        
        run_record = {
            'run_id': run_manifest['run_id'],
            'dataset_name': run_manifest['dataset_info']['dataset_name'],
            'domain': run_manifest['dataset_info']['domain_hint'],
            'schema_hash': schema_hash,
            'sample_hash': sample_hash,
            'timestamp': datetime.utcnow().isoformat(),
            'confidence_score': getattr(state, 'quality_score', 0),
            'quality_score': validation_report.get('quality_metrics', {}).get('overall_quality_score', 0),
            'rows_processed': len(transformed_data),
            'columns_processed': len(transformed_data.columns),
            'success': success
        }
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO runs 
                (run_id, dataset_name, domain, schema_hash, sample_hash, timestamp, 
                 confidence_score, quality_score, rows_processed, columns_processed, success)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_record['run_id'], run_record['dataset_name'], run_record['domain'],
                run_record['schema_hash'], run_record['sample_hash'], run_record['timestamp'],
                run_record['confidence_score'], run_record['quality_score'],
                run_record['rows_processed'], run_record['columns_processed'], run_record['success']
            ))
            conn.commit()
        
        # Add to run history
        self.run_history.append(run_record)
        
        # Keep only last 100 runs in memory
        if len(self.run_history) > 100:
            self.run_history = self.run_history[-100:]
        
        # Save to file
        self._save_json_file(self.run_history_path, self.run_history)
    
    async def _store_schema_mappings(self, canonical_mapping: Dict, state):
        """Store schema mappings for future reuse"""
        
        transformed_data = getattr(state, 'transformed_data', pd.DataFrame())
        schema_hash = self._calculate_schema_hash(transformed_data)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for original_col, mapping_info in canonical_mapping.items():
                canonical_col = mapping_info['canonical_name']
                suggested_type = mapping_info['suggested_type']
                confidence = mapping_info['confidence']
                
                # Check if mapping already exists
                cursor.execute("""
                    SELECT usage_count FROM schema_mappings 
                    WHERE schema_hash = ? AND original_column = ?
                """, (schema_hash, original_col))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Update usage count
                    cursor.execute("""
                        UPDATE schema_mappings 
                        SET usage_count = usage_count + 1, last_used = ?
                        WHERE schema_hash = ? AND original_column = ?
                    """, (datetime.utcnow().isoformat(), schema_hash, original_col))
                else:
                    # Insert new mapping
                    cursor.execute("""
                        INSERT INTO schema_mappings 
                        (schema_hash, original_column, canonical_column, suggested_type, confidence, last_used)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (schema_hash, original_col, canonical_col, suggested_type, confidence, 
                          datetime.utcnow().isoformat()))
            
            conn.commit()
        
        # Update schema cache
        if schema_hash not in self.schema_cache:
            self.schema_cache[schema_hash] = {}
        
        self.schema_cache[schema_hash].update(canonical_mapping)
        self._save_json_file(self.schema_cache_path, self.schema_cache)
    
    async def _store_transform_rules(self, transformation_log: List, state):
        """Store successful transformation rules"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for transform in transformation_log:
                rule_id = f"{transform['transformation_type']}_{hashlib.md5(str(transform).encode()).hexdigest()[:8]}"
                
                rule_record = {
                    'rule_id': rule_id,
                    'rule_type': transform['transformation_type'],
                    'column_pattern': transform.get('base_column', ''),
                    'action': transform.get('description', ''),
                    'parameters': json.dumps(transform),
                    'success_rate': 1.0,  # Assume successful since it's in the log
                    'created_at': datetime.utcnow().isoformat(),
                    'last_used': datetime.utcnow().isoformat()
                }
                
                # Check if rule exists
                cursor.execute("SELECT usage_count FROM transform_rules WHERE rule_id = ?", (rule_id,))
                existing = cursor.fetchone()
                
                if existing:
                    cursor.execute("""
                        UPDATE transform_rules 
                        SET usage_count = usage_count + 1, last_used = ?
                        WHERE rule_id = ?
                    """, (datetime.utcnow().isoformat(), rule_id))
                else:
                    cursor.execute("""
                        INSERT INTO transform_rules 
                        (rule_id, rule_type, column_pattern, action, parameters, success_rate, created_at, last_used)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (rule_record['rule_id'], rule_record['rule_type'], rule_record['column_pattern'],
                          rule_record['action'], rule_record['parameters'], rule_record['success_rate'],
                          rule_record['created_at'], rule_record['last_used']))
                
            conn.commit()
    
    async def _store_quality_patterns(self, validation_report: Dict, state):
        """Store quality issue patterns and remediation strategies"""
        
        run_manifest = state.run_manifest
        domain = run_manifest['dataset_info']['domain_hint']
        
        recommendations = validation_report.get('recommendations', [])
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for rec in recommendations:
                pattern_id = f"{domain}_{rec['type']}_{hashlib.md5(rec['issue'].encode()).hexdigest()[:8]}"
                
                pattern_record = {
                    'pattern_id': pattern_id,
                    'domain': domain,
                    'issue_type': rec['type'],
                    'pattern_description': rec['issue'],
                    'remediation': rec['recommendation'],
                    'effectiveness_score': 0.8,  # Default effectiveness
                    'created_at': datetime.utcnow().isoformat()
                }
                
                cursor.execute("SELECT usage_count FROM quality_patterns WHERE pattern_id = ?", (pattern_id,))
                existing = cursor.fetchone()
                
                if existing:
                    cursor.execute("""
                        UPDATE quality_patterns 
                        SET usage_count = usage_count + 1
                        WHERE pattern_id = ?
                    """, (pattern_id,))
                else:
                    cursor.execute("""
                        INSERT INTO quality_patterns 
                        (pattern_id, domain, issue_type, pattern_description, remediation, effectiveness_score, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (pattern_record['pattern_id'], pattern_record['domain'], pattern_record['issue_type'],
                          pattern_record['pattern_description'], pattern_record['remediation'],
                          pattern_record['effectiveness_score'], pattern_record['created_at']))
                
            conn.commit()
    
    async def _generate_memory_insights(self, state) -> Dict[str, Any]:
        """Generate insights from memory about similar runs and patterns"""
        
        run_manifest = state.run_manifest
        domain = run_manifest['dataset_info']['domain_hint']
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Domain statistics
            cursor.execute("""
                SELECT COUNT(*), AVG(quality_score), AVG(confidence_score)
                FROM runs WHERE domain = ?
            """, (domain,))
            domain_stats = cursor.fetchone()
            
            # Recent successful runs
            cursor.execute("""
                SELECT COUNT(*) FROM runs 
                WHERE domain = ? AND success = 1 AND timestamp > ?
            """, (domain, (datetime.utcnow() - timedelta(days=30)).isoformat()))
            recent_successes = cursor.fetchone()[0]
            
            # Common quality issues in domain
            cursor.execute("""
                SELECT issue_type, COUNT(*) as frequency
                FROM quality_patterns WHERE domain = ?
                GROUP BY issue_type ORDER BY frequency DESC LIMIT 5
            """, (domain,))
            common_issues = cursor.fetchall()
            
            # Most effective remediation strategies
            cursor.execute("""
                SELECT remediation, AVG(effectiveness_score) as avg_effectiveness
                FROM quality_patterns WHERE domain = ?
                GROUP BY remediation ORDER BY avg_effectiveness DESC LIMIT 3
            """, (domain,))
            effective_remediations = cursor.fetchall()
        
        insights = {
            'domain_experience': {
                'total_runs': domain_stats[0] if domain_stats[0] else 0,
                'average_quality_score': domain_stats[1] if domain_stats[1] else 0,
                'average_confidence_score': domain_stats[2] if domain_stats[2] else 0,
                'recent_success_rate': recent_successes
            },
            'common_quality_issues': [
                {'issue_type': issue[0], 'frequency': issue[1]} 
                for issue in common_issues
            ],
            'effective_remediations': [
                {'strategy': remedy[0], 'effectiveness': remedy[1]}
                for remedy in effective_remediations
            ],
            'memory_recommendations': self._generate_memory_recommendations(domain_stats, common_issues)
        }
        
        return insights
    
    def _generate_memory_recommendations(self, domain_stats, common_issues) -> List[str]:
        """Generate recommendations based on memory patterns"""
        
        recommendations = []
        
        if domain_stats and domain_stats[1]:  # Has quality score data
            avg_quality = domain_stats[1]
            if avg_quality < 70:
                recommendations.append(f"This domain typically has lower quality scores (avg {avg_quality:.1f}). Consider additional data validation steps.")
        
        if common_issues:
            top_issue = common_issues[0][0]
            recommendations.append(f"Watch for '{top_issue}' issues - this is the most common problem in this domain based on past runs.")
        
        if domain_stats and domain_stats[0] > 5:
            recommendations.append(f"We have good experience with this domain ({domain_stats[0]} previous runs). Pipeline should perform reliably.")
        
        return recommendations
    
    async def _find_similar_runs(self, state) -> List[Dict[str, Any]]:
        """Find similar runs based on schema and domain"""
        
        transformed_data = getattr(state, 'transformed_data', pd.DataFrame())
        current_schema_hash = self._calculate_schema_hash(transformed_data)
        domain = state.run_manifest['dataset_info']['domain_hint']
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Find runs with same schema hash
            cursor.execute("""
                SELECT run_id, dataset_name, quality_score, confidence_score, timestamp
                FROM runs 
                WHERE schema_hash = ? AND run_id != ?
                ORDER BY timestamp DESC LIMIT 5
            """, (current_schema_hash, state.run_manifest['run_id']))
            
            exact_matches = cursor.fetchall()
            
            # Find runs in same domain with similar characteristics
            cursor.execute("""
                SELECT run_id, dataset_name, quality_score, confidence_score, timestamp,
                       ABS(columns_processed - ?) as column_diff
                FROM runs 
                WHERE domain = ? AND run_id != ?
                ORDER BY column_diff, timestamp DESC LIMIT 5
            """, (len(transformed_data.columns), domain, state.run_manifest['run_id']))
            
            similar_runs = cursor.fetchall()
        
        # Format results
        results = []
        
        for run in exact_matches:
            results.append({
                'run_id': run[0],
                'dataset_name': run[1],
                'quality_score': run[2],
                'confidence_score': run[3],
                'timestamp': run[4],
                'similarity_type': 'exact_schema_match'
            })
        
        for run in similar_runs:
            if run[0] not in [r['run_id'] for r in results]:  # Avoid duplicates
                results.append({
                    'run_id': run[0],
                    'dataset_name': run[1],
                    'quality_score': run[2],
                    'confidence_score': run[3],
                    'timestamp': run[4],
                    'similarity_type': 'domain_similarity'
                })
        
        return results[:5]  # Return top 5 similar runs
    
    def _calculate_schema_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of schema (column names and types)"""
        if df is None or df.empty:
            return "empty_schema"
        
        schema_info = {
            'columns': sorted(df.columns.tolist()),
            'dtypes': {col: str(dtype) for col, dtype in sorted(df.dtypes.items())}
        }
        
        schema_str = json.dumps(schema_info, sort_keys=True)
        return hashlib.md5(schema_str.encode()).hexdigest()
    
    def _calculate_sample_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of sample data content"""
        if df is None or df.empty:
            return "empty_data"
        
        # Use first 100 rows for hashing
        sample = df.head(100).fillna("NULL")
        sample_str = sample.to_csv(index=False)
        
        return hashlib.md5(sample_str.encode()).hexdigest()
    
    async def suggest_schema_mappings(self, df: pd.DataFrame, domain: str) -> Dict[str, Dict[str, Any]]:
        """Suggest schema mappings based on memory"""
        
        current_schema_hash = self._calculate_schema_hash(df)
        suggestions = {}
        
        # Check if we have exact schema match
        if current_schema_hash in self.schema_cache:
            self.logger.info(f"Found exact schema match for hash {current_schema_hash}")
            return self.schema_cache[current_schema_hash]
        
        # Search for similar column patterns in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for col in df.columns:
                # Look for exact column name matches
                cursor.execute("""
                    SELECT canonical_column, suggested_type, AVG(confidence) as avg_confidence, COUNT(*) as usage
                    FROM schema_mappings 
                    WHERE original_column = ?
                    GROUP BY canonical_column, suggested_type
                    ORDER BY usage DESC, avg_confidence DESC
                    LIMIT 1
                """, (col,))
                
                exact_match = cursor.fetchone()
                
                if exact_match:
                    suggestions[col] = {
                        'canonical_name': exact_match[0],
                        'suggested_type': exact_match[1],
                        'confidence': exact_match[2],
                        'source': 'memory_exact_match',
                        'usage_count': exact_match[3]
                    }
                else:
                    # Look for similar column names (fuzzy matching)
                    similar_suggestions = self._find_similar_column_names(col, cursor)
                    if similar_suggestions:
                        suggestions[col] = similar_suggestions
        
        return suggestions
    
    def _find_similar_column_names(self, column_name: str, cursor) -> Optional[Dict[str, Any]]:
        """Find similar column names using simple pattern matching"""
        
        # Simple similarity: check if column name contains common patterns
        col_lower = column_name.lower()
        
        # Common patterns to check
        patterns = [
            ('date', 'date'),
            ('time', 'datetime'),
            ('id', 'identifier'),
            ('name', 'text'),
            ('amount', 'numeric'),
            ('count', 'numeric'),
            ('total', 'numeric'),
            ('district', 'geographic'),
            ('mandal', 'geographic'),
            ('village', 'geographic')
        ]
        
        for pattern, category in patterns:
            if pattern in col_lower:
                # Look for mappings of similar columns
                cursor.execute("""
                    SELECT canonical_column, suggested_type, AVG(confidence) as avg_confidence
                    FROM schema_mappings 
                    WHERE LOWER(original_column) LIKE ?
                    GROUP BY canonical_column, suggested_type
                    ORDER BY avg_confidence DESC
                    LIMIT 1
                """, (f"%{pattern}%",))
                
                result = cursor.fetchone()
                if result:
                    return {
                        'canonical_name': result[0],
                        'suggested_type': result[1],
                        'confidence': result[2] * 0.8,  # Reduce confidence for fuzzy match
                        'source': 'memory_pattern_match',
                        'pattern_matched': pattern
                    }
        
        return None
    
    async def get_domain_best_practices(self, domain: str) -> Dict[str, Any]:
        """Get domain-specific best practices from memory"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get most successful transform rules for domain
            cursor.execute("""
                SELECT r.rule_type, r.action, AVG(r.success_rate) as avg_success, COUNT(*) as usage
                FROM transform_rules r
                JOIN runs ru ON r.last_used >= date(ru.timestamp, '-30 days')
                WHERE ru.domain = ?
                GROUP BY r.rule_type, r.action
                ORDER BY avg_success DESC, usage DESC
                LIMIT 10
            """, (domain,))
            
            best_transforms = cursor.fetchall()
            
            # Get quality patterns that work well
            cursor.execute("""
                SELECT remediation, AVG(effectiveness_score) as effectiveness, COUNT(*) as usage
                FROM quality_patterns 
                WHERE domain = ?
                GROUP BY remediation
                ORDER BY effectiveness DESC, usage DESC
                LIMIT 5
            """, (domain,))
            
            best_remediations = cursor.fetchall()
        
        return {
            'recommended_transforms': [
                {
                    'type': transform[0],
                    'action': transform[1],
                    'success_rate': transform[2],
                    'usage_count': transform[3]
                }
                for transform in best_transforms
            ],
            'effective_remediations': [
                {
                    'strategy': remedy[0],
                    'effectiveness': remedy[1],
                    'usage_count': remedy[2]
                }
                for remedy in best_remediations
            ]
        }
    
    async def cleanup_old_memory(self, days_to_keep: int = 90):
        """Clean up old memory entries to prevent storage bloat"""
        
        cutoff_date = (datetime.utcnow() - timedelta(days=days_to_keep)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Clean old runs
            cursor.execute("DELETE FROM runs WHERE timestamp < ?", (cutoff_date,))
            
            # Clean unused schema mappings
            cursor.execute("DELETE FROM schema_mappings WHERE last_used < ?", (cutoff_date,))
            
            # Clean old transform rules
            cursor.execute("DELETE FROM transform_rules WHERE last_used < ?", (cutoff_date,))
            
            conn.commit()
        
        # Clean in-memory caches
        self.run_history = [
            run for run in self.run_history 
            if run['timestamp'] > cutoff_date
        ]
        
        # Save cleaned data
        self._save_json_file(self.run_history_path, self.run_history)
        
        self.logger.info(f"Cleaned memory data older than {days_to_keep} days")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage and statistics"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count records in each table
            cursor.execute("SELECT COUNT(*) FROM runs")
            total_runs = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM schema_mappings")
            total_mappings = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM transform_rules")
            total_rules = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM quality_patterns")
            total_patterns = cursor.fetchone()[0]
            
            # Get domain distribution
            cursor.execute("""
                SELECT domain, COUNT(*) as count
                FROM runs
                GROUP BY domain
                ORDER BY count DESC
            """)
            domain_distribution = cursor.fetchall()
        
        return {
            'total_runs': total_runs,
            'total_schema_mappings': total_mappings,
            'total_transform_rules': total_rules,
            'total_quality_patterns': total_patterns,
            'domain_distribution': {domain: count for domain, count in domain_distribution},
            'database_size_mb': round(self.db_path.stat().st_size / 1024 / 1024, 2) if self.db_path.exists() else 0
        }
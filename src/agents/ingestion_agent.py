# Ingestion Agent (3.1)
"""
RTGS AI Analyst - Ingestion Agent
Handles data loading, encoding detection, and initial dataset profiling
"""

import pandas as pd
import numpy as np
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import chardet
import zipfile
import gzip

from src.utils.logging import get_agent_logger, log_data_shape, TransformLogger
from src.utils.data_helpers import detect_separator, detect_encoding, estimate_row_count


class IngestionAgent:
    """Agent responsible for data ingestion and initial profiling"""
    
    def __init__(self):
        self.logger = get_agent_logger("ingestion")
        
    async def process(self, state) -> Any:
        """Main ingestion processing pipeline"""
        self.logger.info("Starting data ingestion process")
        
        try:
            # Initialize transform logger
            transform_logger = TransformLogger(
                log_file=Path(state.run_manifest['artifacts_paths']['logs_dir']) / "transform_log.jsonl",
                run_id=state.run_manifest['run_id']
            )
            
            # Load dataset
            raw_data, file_info = await self._load_dataset(state.dataset_path)
            state.raw_data = raw_data
            
            # Generate dataset profile
            dataset_profile = await self._generate_dataset_profile(raw_data, file_info, state)
            
            # Save dataset profile
            profile_path = Path(state.run_manifest['artifacts_paths']['docs_dir']) / "dataset_profile.json"
            with open(profile_path, 'w') as f:
                json.dump(dataset_profile, f, indent=2)
            
            # Log ingestion action
            transform_logger.log_transform(
                agent="ingestion",
                action="load_dataset",
                rows_affected=len(raw_data) if raw_data is not None else 0,
                rule_id="ingestion_v1",
                rationale=f"Loaded {file_info['format']} file with {file_info['encoding']} encoding",
                parameters=file_info,
                confidence="high"
            )
            
            # Create sample for schema inference
            sample_data = await self._create_sample(raw_data, state.run_manifest['run_config']['sample_rows'])
            
            # Update state
            state.dataset_profile = dataset_profile
            state.sample_data = sample_data
            state.file_info = file_info
            
            self.logger.info(f"Ingestion completed: {len(raw_data)} rows, {len(raw_data.columns)} columns")
            log_data_shape(raw_data, self.logger, "ingestion_complete")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Ingestion failed: {str(e)}")
            state.errors.append(f"Ingestion failed: {str(e)}")
            return state

    async def _load_dataset(self, dataset_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load dataset from various formats with robust error handling"""
        path = Path(dataset_path)
        self.logger.info(f"Loading dataset from: {path}")
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        
        # Detect file info
        file_info = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
            "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        }
        
        # Handle compressed files
        if path.suffix.lower() in ['.gz', '.zip']:
            path, file_info = await self._handle_compressed_file(path, file_info)
        
        # Detect file format and encoding
        file_format = path.suffix.lower().lstrip('.')
        file_info['format'] = file_format
        
        if file_format == 'csv':
            return await self._load_csv(path, file_info)
        elif file_format in ['xlsx', 'xls']:
            return await self._load_excel(path, file_info)
        elif file_format == 'json':
            return await self._load_json(path, file_info)
        elif file_format == 'parquet':
            return await self._load_parquet(path, file_info)
        else:
            # Try to load as CSV by default
            self.logger.warning(f"Unknown format {file_format}, attempting CSV load")
            return await self._load_csv(path, file_info)

    async def _handle_compressed_file(self, path: Path, file_info: Dict) -> Tuple[Path, Dict]:
        """Handle compressed files (zip, gzip)"""
        self.logger.info(f"Handling compressed file: {path}")
        
        if path.suffix.lower() == '.gz':
            # For .gz files, assume single CSV inside
            with gzip.open(path, 'rb') as f:
                content = f.read()
            
            # Create temporary file
            temp_path = path.parent / f"temp_{path.stem}"
            with open(temp_path, 'wb') as f:
                f.write(content)
            
            file_info['compressed'] = True
            file_info['compression_format'] = 'gzip'
            return temp_path, file_info
            
        elif path.suffix.lower() == '.zip':
            with zipfile.ZipFile(path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                # Find the largest CSV/Excel file
                csv_files = [f for f in file_list if f.lower().endswith(('.csv', '.xlsx', '.xls'))]
                
                if not csv_files:
                    raise ValueError("No CSV or Excel files found in ZIP archive")
                
                # Extract the first CSV file
                target_file = csv_files[0]
                extract_path = path.parent / target_file
                zip_ref.extract(target_file, path.parent)
                
                file_info['compressed'] = True
                file_info['compression_format'] = 'zip'
                file_info['extracted_file'] = target_file
                
                return extract_path, file_info
        
        return path, file_info

    async def _load_csv(self, path: Path, file_info: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Load CSV file with robust encoding and separator detection"""
        self.logger.info(f"Loading CSV file: {path}")
        
        # Detect encoding
        encoding = detect_encoding(str(path))
        file_info['encoding'] = encoding
        
        # Detect separator
        separator = detect_separator(str(path), encoding)
        file_info['separator'] = separator
        
        # Estimate row count for large files
        estimated_rows = estimate_row_count(str(path))
        file_info['estimated_rows'] = estimated_rows
        
        try:
            # For large files, read in chunks
            if file_info['size_mb'] > 100:  # 100MB threshold
                self.logger.info("Large file detected, using chunked reading")
                chunks = []
                chunk_size = 10000
                
                for chunk in pd.read_csv(
                    path, 
                    encoding=encoding, 
                    sep=separator,
                    chunksize=chunk_size,
                    low_memory=False
                ):
                    chunks.append(chunk)
                    if len(chunks) * chunk_size > 100000:  # Limit to 100k rows for memory
                        break
                
                df = pd.concat(chunks, ignore_index=True)
                file_info['loading_method'] = 'chunked'
                
            else:
                # Load entire file
                df = pd.read_csv(
                    path,
                    encoding=encoding,
                    sep=separator,
                    low_memory=False
                )
                file_info['loading_method'] = 'full'
            
            file_info['actual_rows'] = len(df)
            file_info['actual_columns'] = len(df.columns)
            
            self.logger.info(f"CSV loaded successfully: {len(df)} rows, {len(df.columns)} columns")
            return df, file_info
            
        except Exception as e:
            self.logger.error(f"Failed to load CSV: {str(e)}")
            # Fallback: try with different parameters
            try:
                df = pd.read_csv(path, encoding='utf-8', sep=',', error_bad_lines=False)
                file_info['loading_method'] = 'fallback'
                file_info['encoding'] = 'utf-8'
                file_info['separator'] = ','
                return df, file_info
            except:
                raise ValueError(f"Could not load CSV file: {str(e)}")

    async def _load_excel(self, path: Path, file_info: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Load Excel file with sheet detection"""
        self.logger.info(f"Loading Excel file: {path}")
        
        try:
            # Read Excel file info
            excel_file = pd.ExcelFile(path)
            sheet_names = excel_file.sheet_names
            
            file_info['format'] = 'excel'
            file_info['sheet_names'] = sheet_names
            file_info['sheets_count'] = len(sheet_names)
            
            # Select sheet (first non-empty sheet by default)
            target_sheet = None
            for sheet in sheet_names:
                df_test = pd.read_excel(excel_file, sheet_name=sheet, nrows=5)
                if not df_test.empty:
                    target_sheet = sheet
                    break
            
            if target_sheet is None:
                target_sheet = sheet_names[0]
            
            file_info['selected_sheet'] = target_sheet
            
            # Load the selected sheet
            df = pd.read_excel(path, sheet_name=target_sheet)
            
            file_info['actual_rows'] = len(df)
            file_info['actual_columns'] = len(df.columns)
            
            self.logger.info(f"Excel loaded successfully: {len(df)} rows, {len(df.columns)} columns from sheet '{target_sheet}'")
            return df, file_info
            
        except Exception as e:
            raise ValueError(f"Could not load Excel file: {str(e)}")

    async def _load_json(self, path: Path, file_info: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Load JSON file and normalize to DataFrame"""
        self.logger.info(f"Loading JSON file: {path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Normalize JSON to DataFrame
            if isinstance(data, list):
                df = pd.json_normalize(data)
            elif isinstance(data, dict):
                if len(data) == 1 and isinstance(list(data.values())[0], list):
                    # Single key with list of records
                    df = pd.json_normalize(list(data.values())[0])
                else:
                    # Single record
                    df = pd.json_normalize([data])
            else:
                raise ValueError("JSON structure not supported")
            
            file_info['format'] = 'json'
            file_info['actual_rows'] = len(df)
            file_info['actual_columns'] = len(df.columns)
            
            self.logger.info(f"JSON loaded successfully: {len(df)} rows, {len(df.columns)} columns")
            return df, file_info
            
        except Exception as e:
            raise ValueError(f"Could not load JSON file: {str(e)}")

    async def _load_parquet(self, path: Path, file_info: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Load Parquet file"""
        self.logger.info(f"Loading Parquet file: {path}")
        
        try:
            df = pd.read_parquet(path)
            
            file_info['format'] = 'parquet'
            file_info['actual_rows'] = len(df)
            file_info['actual_columns'] = len(df.columns)
            
            self.logger.info(f"Parquet loaded successfully: {len(df)} rows, {len(df.columns)} columns")
            return df, file_info
            
        except Exception as e:
            raise ValueError(f"Could not load Parquet file: {str(e)}")

    async def _generate_dataset_profile(self, df: pd.DataFrame, file_info: Dict, state) -> Dict[str, Any]:
        """Generate comprehensive dataset profile"""
        self.logger.info("Generating dataset profile")
        
        # Basic statistics
        profile = {
            "dataset_name": state.run_manifest['dataset_info']['dataset_name'],
            "source_path": file_info['path'],
            "domain": state.run_manifest['dataset_info']['domain_hint'],
            "scope": state.run_manifest['dataset_info']['scope'],
            "description": state.run_manifest['dataset_info']['description'],
            "generated_at": datetime.utcnow().isoformat(),
            
            "file_info": file_info,
            
            "shape": {
                "rows_total": len(df),
                "columns_total": len(df.columns),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            },
            
            "columns": {
                "names": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
        }
        
        # Data fingerprints for memory/caching
        profile["fingerprints"] = {
            "schema_hash": self._calculate_schema_hash(df),
            "sample_hash": self._calculate_sample_hash(df),
            "file_hash": self._calculate_file_hash(file_info['path'])
        }
        
        # Quick quality assessment
        profile["quality_preview"] = {
            "missing_values_total": int(df.isnull().sum().sum()),
            "missing_columns": [col for col in df.columns if df[col].isnull().sum() > 0],
            "duplicate_rows": int(df.duplicated().sum()),
            "empty_columns": [col for col in df.columns if df[col].isnull().all()]
        }
        
        # Sample preview (first 5 rows, safe for display)
        profile["sample_preview"] = df.head().fillna("NULL").to_dict('records')
        
        return profile

    async def _create_sample(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Create representative sample for schema inference"""
        if len(df) <= sample_size:
            return df.copy()
        
        # Smart sampling: ensure we get diverse data
        # Take sample from beginning, middle, and end
        third = len(df) // 3
        sample_indices = []
        
        # Beginning third
        sample_indices.extend(range(0, min(third, sample_size // 3)))
        
        # Middle third  
        start_mid = third
        end_mid = min(2 * third, start_mid + sample_size // 3)
        sample_indices.extend(range(start_mid, end_mid))
        
        # End third
        remaining = sample_size - len(sample_indices)
        start_end = max(2 * third, len(df) - remaining)
        sample_indices.extend(range(start_end, len(df)))
        
        # Remove duplicates and limit to sample_size
        sample_indices = list(set(sample_indices))[:sample_size]
        
        return df.iloc[sample_indices].copy()

    def _calculate_schema_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of column names and types for schema fingerprinting"""
        schema_str = json.dumps({
            "columns": sorted(df.columns.tolist()),
            "dtypes": {col: str(dtype) for col, dtype in sorted(df.dtypes.items())}
        }, sort_keys=True)
        
        return hashlib.md5(schema_str.encode()).hexdigest()

    def _calculate_sample_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of sample data for content fingerprinting"""
        # Use first 100 rows for hashing
        sample = df.head(100).fillna("NULL")
        sample_str = sample.to_csv(index=False)
        
        return hashlib.md5(sample_str.encode()).hexdigest()

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate hash of original file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
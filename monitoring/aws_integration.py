"""
AWS Integration Module for PTZ Camera Calibration Monitoring

This module provides functionality for:
1. Athena table schema management with Iceberg format
2. S3 file upload/download utilities
3. Athena query and write operations

Table Schema:
- deployment_name: str
- device_id: str
- timestamp: timestamp
- pitch_offset: double
- yaw_offset: double
- roll_offset: double
- mode: str (passive/active)
- capture_positions: array<struct<pan:double,tilt:double,zoom:double>>
- files_location: str (S3 path)
- success: boolean
- failure_log: str
"""

import boto3
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import time

logger = logging.getLogger(__name__)


# S3 Configuration
S3_BUCKET_NAME = "camera-calibration-monitoring"
ATHENA_DATABASE = "camera_calibration_monitoring"
ATHENA_TABLE = "calibration_results"
ATHENA_OUTPUT_LOCATION = f"s3://{S3_BUCKET_NAME}/athena-results/"


class AWSIntegration:
    """
    AWS Integration class for managing S3 uploads and Athena queries
    for PTZ camera calibration monitoring.
    """
    
    def __init__(
        self,
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None
    ):
        """
        Initialize AWS clients.
        
        Args:
            region_name: AWS region name
            aws_access_key_id: Optional AWS access key (uses env/credentials otherwise)
            aws_secret_access_key: Optional AWS secret key (uses env/credentials otherwise)
        """
        session_kwargs = {"region_name": region_name}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
        
        self.session = boto3.Session(**session_kwargs)
        self.s3_client = self.session.client('s3')
        self.athena_client = self.session.client('athena')
        self.glue_client = self.session.client('glue')
        
    # ==================== Athena Table Schema ====================
    
    @staticmethod
    def get_table_schema() -> str:
        """
        Get the Athena Iceberg table schema DDL.
        
        Returns:
            CREATE TABLE statement for the calibration results table
        """
        return f"""
        CREATE TABLE IF NOT EXISTS {ATHENA_DATABASE}.{ATHENA_TABLE} (
            deployment_name STRING,
            device_id STRING,
            timestamp TIMESTAMP,
            pitch_offset DOUBLE,
            yaw_offset DOUBLE,
            roll_offset DOUBLE,
            mode STRING,
            capture_positions ARRAY<STRUCT<pan: DOUBLE, tilt: DOUBLE, zoom: DOUBLE>>,
            files_location STRING,
            success BOOLEAN,
            failure_log STRING
        )
        LOCATION 's3://{S3_BUCKET_NAME}/iceberg-data/'
        TBLPROPERTIES (
            'table_type' = 'ICEBERG'
        )
        """
    
    def create_table(self) -> bool:
        """
        Create the Athena table if it doesn't exist.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # First create the database if it doesn't exist
            create_db_query = f"CREATE DATABASE IF NOT EXISTS {ATHENA_DATABASE}"
            self._execute_query(create_db_query, wait=True)
            
            # Then create the table
            create_table_query = self.get_table_schema()
            query_execution_id = self._execute_query(create_table_query, wait=True)
            
            logger.info(f"Table {ATHENA_DATABASE}.{ATHENA_TABLE} created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            return False
    
    # ==================== S3 Utilities ====================
    
    def get_s3_path(
        self,
        deployment_name: str,
        camera_name: str,
        scan_type: str,
        data_type: str,
        timestamp: Optional[str] = None
    ) -> str:
        """
        Generate S3 path according to the standard structure.
        
        Args:
            deployment_name: Name of the deployment
            camera_name: Name of the camera/device
            scan_type: 'reference_scan' or 'query_scan'
            data_type: 'images' or 'features'
            timestamp: Optional timestamp for query scans
            
        Returns:
            S3 path string
            
        Example paths:
            - s3://camera-calibration-monitoring/deployment1/camera1/reference_scan/images/
            - s3://camera-calibration-monitoring/deployment1/camera1/query_scan/2024-11-15T10:30:00/images/
        """
        if scan_type == "reference_scan":
            path = f"s3://{S3_BUCKET_NAME}/{deployment_name}/{camera_name}/reference_scan/{data_type}/"
        elif scan_type == "query_scan":
            if not timestamp:
                raise ValueError("timestamp is required for query_scan")
            path = f"s3://{S3_BUCKET_NAME}/{deployment_name}/{camera_name}/query_scan/{timestamp}/{data_type}/"
        else:
            raise ValueError(f"Invalid scan_type: {scan_type}. Must be 'reference_scan' or 'query_scan'")
        
        return path
    
    def upload_file(
        self,
        local_file_path: str,
        deployment_name: str,
        camera_name: str,
        scan_type: str,
        data_type: str,
        timestamp: Optional[str] = None,
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Upload a file to S3 with the standard path structure.
        
        Args:
            local_file_path: Local path to the file to upload
            deployment_name: Name of the deployment
            camera_name: Name of the camera/device
            scan_type: 'reference_scan' or 'query_scan'
            data_type: 'images' or 'features'
            timestamp: Optional timestamp for query scans
            filename: Optional custom filename (uses original if not provided)
            
        Returns:
            S3 URI of uploaded file, or None if failed
        """
        try:
            local_path = Path(local_file_path)
            if not local_path.exists():
                logger.error(f"Local file does not exist: {local_file_path}")
                return None
            
            # Get base path
            s3_base_path = self.get_s3_path(
                deployment_name, camera_name, scan_type, data_type, timestamp
            )
            
            # Determine filename
            if filename is None:
                filename = local_path.name
            
            # Construct full S3 key (remove s3://bucket/ prefix)
            s3_key = s3_base_path.replace(f"s3://{S3_BUCKET_NAME}/", "") + filename
            
            # Upload file
            logger.info(f"Uploading {local_file_path} to s3://{S3_BUCKET_NAME}/{s3_key}")
            self.s3_client.upload_file(str(local_path), S3_BUCKET_NAME, s3_key)
            
            s3_uri = f"s3://{S3_BUCKET_NAME}/{s3_key}"
            logger.info(f"Successfully uploaded to {s3_uri}")
            return s3_uri
            
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            return None
    
    def upload_directory(
        self,
        local_dir_path: str,
        deployment_name: str,
        camera_name: str,
        scan_type: str,
        data_type: str,
        timestamp: Optional[str] = None,
        file_pattern: str = "*"
    ) -> List[str]:
        """
        Upload all files in a directory to S3.
        
        Args:
            local_dir_path: Local directory path
            deployment_name: Name of the deployment
            camera_name: Name of the camera/device
            scan_type: 'reference_scan' or 'query_scan'
            data_type: 'images' or 'features'
            timestamp: Optional timestamp for query scans
            file_pattern: Glob pattern for files to upload (default: all files)
            
        Returns:
            List of S3 URIs of uploaded files
        """
        local_dir = Path(local_dir_path)
        if not local_dir.exists() or not local_dir.is_dir():
            logger.error(f"Local directory does not exist: {local_dir_path}")
            return []
        
        uploaded_files = []
        for file_path in local_dir.glob(file_pattern):
            if file_path.is_file():
                s3_uri = self.upload_file(
                    str(file_path),
                    deployment_name,
                    camera_name,
                    scan_type,
                    data_type,
                    timestamp
                )
                if s3_uri:
                    uploaded_files.append(s3_uri)
        
        logger.info(f"Uploaded {len(uploaded_files)} files from {local_dir_path}")
        return uploaded_files
    
    def download_file(
        self,
        s3_uri: str,
        local_file_path: str
    ) -> bool:
        """
        Download a file from S3.
        
        Args:
            s3_uri: Full S3 URI (e.g., s3://bucket/key)
            local_file_path: Local path where file should be saved
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Parse S3 URI
            if not s3_uri.startswith("s3://"):
                raise ValueError(f"Invalid S3 URI: {s3_uri}")
            
            parts = s3_uri.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            
            # Create local directory if it doesn't exist
            local_path = Path(local_file_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            logger.info(f"Downloading {s3_uri} to {local_file_path}")
            self.s3_client.download_file(bucket, key, str(local_path))
            
            logger.info(f"Successfully downloaded to {local_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            return False
    
    def list_files(
        self,
        deployment_name: str,
        camera_name: str,
        scan_type: str,
        data_type: str,
        timestamp: Optional[str] = None
    ) -> List[str]:
        """
        List all files in an S3 path.
        
        Args:
            deployment_name: Name of the deployment
            camera_name: Name of the camera/device
            scan_type: 'reference_scan' or 'query_scan'
            data_type: 'images' or 'features'
            timestamp: Optional timestamp for query scans
            
        Returns:
            List of S3 URIs
        """
        try:
            s3_path = self.get_s3_path(
                deployment_name, camera_name, scan_type, data_type, timestamp
            )
            prefix = s3_path.replace(f"s3://{S3_BUCKET_NAME}/", "")
            
            response = self.s3_client.list_objects_v2(
                Bucket=S3_BUCKET_NAME,
                Prefix=prefix
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    files.append(f"s3://{S3_BUCKET_NAME}/{obj['Key']}")
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    # ==================== Athena Operations ====================
    
    def _execute_query(
        self,
        query: str,
        database: Optional[str] = None,
        wait: bool = False,
        max_wait_seconds: int = 300
    ) -> str:
        """
        Execute an Athena query.
        
        Args:
            query: SQL query string
            database: Database name (uses default if not provided)
            wait: Whether to wait for query completion
            max_wait_seconds: Maximum time to wait for query completion
            
        Returns:
            Query execution ID
        """
        query_context = {}
        if database:
            query_context['Database'] = database
        else:
            query_context['Database'] = ATHENA_DATABASE
        
        response = self.athena_client.start_query_execution(
            QueryString=query,
            QueryExecutionContext=query_context,
            ResultConfiguration={
                'OutputLocation': ATHENA_OUTPUT_LOCATION,
            }
        )
        
        query_execution_id = response['QueryExecutionId']
        logger.info(f"Started query execution: {query_execution_id}")
        
        if wait:
            self._wait_for_query(query_execution_id, max_wait_seconds)
        
        return query_execution_id
    
    def _wait_for_query(
        self,
        query_execution_id: str,
        max_wait_seconds: int = 300
    ) -> str:
        """
        Wait for a query to complete.
        
        Args:
            query_execution_id: Query execution ID
            max_wait_seconds: Maximum time to wait
            
        Returns:
            Final query state
        """
        start_time = time.time()
        
        while True:
            if time.time() - start_time > max_wait_seconds:
                raise TimeoutError(f"Query {query_execution_id} did not complete within {max_wait_seconds} seconds")
            
            response = self.athena_client.get_query_execution(
                QueryExecutionId=query_execution_id
            )
            
            state = response['QueryExecution']['Status']['State']
            
            if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                if state != 'SUCCEEDED':
                    reason = response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown')
                    raise RuntimeError(f"Query {query_execution_id} {state}: {reason}")
                return state
            
            time.sleep(2)
    
    def write_calibration_result(
        self,
        deployment_name: str,
        device_id: str,
        timestamp: datetime,
        pitch_offset: float,
        yaw_offset: float,
        roll_offset: float,
        mode: str,
        capture_positions: List[Dict[str, float]],
        files_location: str,
        success: bool,
        failure_log: str = ""
    ) -> Optional[str]:
        """
        Write a calibration result to Athena.
        
        Args:
            deployment_name: Name of the deployment
            device_id: ID of the device/camera
            timestamp: Timestamp of the calibration
            pitch_offset: Pitch offset in degrees
            yaw_offset: Yaw offset in degrees
            roll_offset: Roll offset in degrees
            mode: 'passive' or 'active'
            capture_positions: List of dicts with 'pan', 'tilt', 'zoom' keys
            files_location: S3 path to the files
            success: Whether the calibration was successful
            failure_log: Error message if failed
            
        Returns:
            Query execution ID, or None if failed
        """
        try:
            # Format timestamp
            ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
            # Format capture positions as array of structs
            positions_str = "ARRAY[" + ", ".join([
                f"CAST(ROW({pos['pan']}, {pos['tilt']}, {pos['zoom']}) AS ROW(pan DOUBLE, tilt DOUBLE, zoom DOUBLE))"
                for pos in capture_positions
            ]) + "]"
            
            # Format failure log (escape single quotes)
            failure_log_escaped = failure_log.replace("'", "''")
            
            # Extract partition values
            year = timestamp.year
            month = timestamp.month
            day = timestamp.day
            
            # Construct INSERT query
            query = f"""
            INSERT INTO {ATHENA_DATABASE}.{ATHENA_TABLE}
            VALUES (
                '{deployment_name}',
                '{device_id}',
                TIMESTAMP '{ts_str}',
                {pitch_offset},
                {yaw_offset},
                {roll_offset},
                '{mode}',
                {positions_str},
                '{files_location}',
                {str(success).lower()},
                '{failure_log_escaped}'
            )
            """
            
            query_execution_id = self._execute_query(query, wait=True)
            logger.info(f"Successfully wrote calibration result: {query_execution_id}")
            return query_execution_id
            
        except Exception as e:
            logger.error(f"Failed to write calibration result: {e}")
            return None
    
    def query_calibration_results(
        self,
        deployment_name: Optional[str] = None,
        device_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        mode: Optional[str] = None,
        success_only: bool = False,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query calibration results from Athena.
        
        Args:
            deployment_name: Filter by deployment name
            device_id: Filter by device ID
            start_date: Filter results after this date
            end_date: Filter results before this date
            mode: Filter by mode ('passive' or 'active')
            success_only: Only return successful calibrations
            limit: Maximum number of results to return
            
        Returns:
            List of calibration result dictionaries
        """
        try:
            # Build WHERE clause
            where_clauses = []
            if deployment_name:
                where_clauses.append(f"deployment_name = '{deployment_name}'")
            if device_id:
                where_clauses.append(f"device_id = '{device_id}'")
            if start_date:
                where_clauses.append(f"timestamp >= TIMESTAMP '{start_date.strftime('%Y-%m-%d %H:%M:%S')}'")
            if end_date:
                where_clauses.append(f"timestamp <= TIMESTAMP '{end_date.strftime('%Y-%m-%d %H:%M:%S')}'")
            if mode:
                where_clauses.append(f"mode = '{mode}'")
            if success_only:
                where_clauses.append("success = true")
            
            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            # Construct query
            query = f"""
            SELECT *
            FROM {ATHENA_DATABASE}.{ATHENA_TABLE}
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT {limit}
            """
            
            query_execution_id = self._execute_query(query, wait=True)
            
            # Get results
            results = self._get_query_results(query_execution_id)
            return results
            
        except Exception as e:
            logger.error(f"Failed to query calibration results: {e}")
            return []
    
    def _get_query_results(self, query_execution_id: str) -> List[Dict[str, Any]]:
        """
        Get results from a completed Athena query.
        
        Args:
            query_execution_id: Query execution ID
            
        Returns:
            List of result rows as dictionaries
        """
        try:
            response = self.athena_client.get_query_results(
                QueryExecutionId=query_execution_id
            )
            
            # Extract column names from first row
            columns = [col['Label'] for col in response['ResultSet']['ResultSetMetadata']['ColumnInfo']]
            
            # Extract data rows (skip header row)
            rows = response['ResultSet']['Rows'][1:]
            
            results = []
            for row in rows:
                values = [field.get('VarCharValue', None) for field in row['Data']]
                result_dict = dict(zip(columns, values))
                results.append(result_dict)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get query results: {e}")
            return []
    
    def get_latest_calibration(
        self,
        deployment_name: str,
        device_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the most recent calibration result for a device.
        
        Args:
            deployment_name: Name of the deployment
            device_id: ID of the device/camera
            
        Returns:
            Calibration result dictionary, or None if not found
        """
        results = self.query_calibration_results(
            deployment_name=deployment_name,
            device_id=device_id,
            limit=1
        )
        
        return results[0] if results else None


# Convenience functions for common operations

def upload_reference_scan(
    aws_integration: AWSIntegration,
    deployment_name: str,
    camera_name: str,
    images_dir: str,
    features_dir: Optional[str] = None
) -> Tuple[List[str], List[str]]:
    """
    Upload a complete reference scan (images and optionally features).
    
    Args:
        aws_integration: AWSIntegration instance
        deployment_name: Name of the deployment
        camera_name: Name of the camera
        images_dir: Local directory containing images
        features_dir: Optional local directory containing features
        
    Returns:
        Tuple of (image_s3_uris, feature_s3_uris)
    """
    image_uris = aws_integration.upload_directory(
        images_dir,
        deployment_name,
        camera_name,
        "reference_scan",
        "images"
    )
    
    feature_uris = []
    if features_dir:
        feature_uris = aws_integration.upload_directory(
            features_dir,
            deployment_name,
            camera_name,
            "reference_scan",
            "features"
        )
    
    return image_uris, feature_uris


def upload_query_scan(
    aws_integration: AWSIntegration,
    deployment_name: str,
    camera_name: str,
    timestamp: str,
    images_dir: str,
    features_dir: Optional[str] = None
) -> Tuple[List[str], List[str]]:
    """
    Upload a complete query scan (images and optionally features).
    
    Args:
        aws_integration: AWSIntegration instance
        deployment_name: Name of the deployment
        camera_name: Name of the camera
        timestamp: Timestamp string for the query scan
        images_dir: Local directory containing images
        features_dir: Optional local directory containing features
        
    Returns:
        Tuple of (image_s3_uris, feature_s3_uris)
    """
    image_uris = aws_integration.upload_directory(
        images_dir,
        deployment_name,
        camera_name,
        "query_scan",
        "images",
        timestamp=timestamp
    )
    
    feature_uris = []
    if features_dir:
        feature_uris = aws_integration.upload_directory(
            features_dir,
            deployment_name,
            camera_name,
            "query_scan",
            "features",
            timestamp=timestamp
        )
    
    return image_uris, feature_uris


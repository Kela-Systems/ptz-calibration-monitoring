#!/usr/bin/env python3
"""
Main Orchestration Script for PTZ Camera Calibration Monitoring

This script:
1. Loads devices from devices.yaml
2. For each device:
   - Switches kubectl context to device's cluster
   - Extracts query frames using query_extractor
   - Runs calibration algorithm via calibration_runner
   - Writes results to Athena table
   - Uploads query files to S3 query_scan/{timestamp}/
   - Sends Slack notification
3. Handles failures gracefully with logging and alerts
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.query_extractor import extract_query
from monitoring.calibration_runner import (
    CalibrationRunner,
    CalibrationConfig,
    load_camera_calibration,
    load_r_align
)
from monitoring.aws_integration import AWSIntegration
from monitoring.slack_notifier import SlackNotifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("calibration-orchestration")


class CalibrationMonitoringOrchestrator:
    """
    Main orchestrator for PTZ camera calibration monitoring.
    
    Coordinates all monitoring components:
    - Query extraction from cameras
    - Calibration algorithm execution
    - Results storage in Athena
    - File uploads to S3
    - Slack notifications
    """
    
    def __init__(
        self,
        devices_yaml_path: str = "devices.yaml",
        camera_intrinsics_path: str = "camera_intrinsics/calibration.npz",
        r_align_path: str = "r_align/10NOV2025_REF1_HOMOGRAPHY.npy",
        s3_bucket: str = "camera-calibration-monitoring",
        aws_region: str = "us-east-1",
        calibration_config: Optional[CalibrationConfig] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            devices_yaml_path: Path to devices.yaml configuration
            camera_intrinsics_path: Path to camera calibration .npz file
            r_align_path: Path to R_align .npy file
            s3_bucket: S3 bucket name for data storage
            aws_region: AWS region
            calibration_config: Optional calibration configuration
        """
        self.devices_yaml_path = devices_yaml_path
        self.camera_intrinsics_path = camera_intrinsics_path
        self.r_align_path = r_align_path
        self.s3_bucket = s3_bucket
        self.aws_region = aws_region
        
        # Load devices configuration
        self.devices = self._load_devices()
        
        # Initialize components
        self.calibration_runner = CalibrationRunner(
            s3_bucket=s3_bucket,
            aws_region=aws_region,
            config=calibration_config or CalibrationConfig()
        )
        self.aws_integration = AWSIntegration(region_name=aws_region)
        self.slack_notifier = SlackNotifier()
        
        # Load camera calibration data (shared across all devices)
        logger.info(f"Loading camera calibration from {camera_intrinsics_path}")
        self.camera_matrix, self.dist_coeff = load_camera_calibration(camera_intrinsics_path)
        
        logger.info(f"Loading R_align from {r_align_path}")
        self.r_align = load_r_align(r_align_path)
        
        logger.info("Orchestrator initialized successfully")
    
    def _load_devices(self) -> List[Dict]:
        """
        Load devices configuration from devices.yaml.
        
        Returns:
            List of device configurations
        """
        try:
            with open(self.devices_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            devices = []
            for deployment in data:
                deployment_name = deployment['name']
                for camera in deployment.get('cameras', []):
                    camera_name = camera['name']
                    devices.append({
                        'deployment_name': deployment_name,
                        'device_id': camera_name
                    })
            
            logger.info(f"Loaded {len(devices)} device(s) from {self.devices_yaml_path}")
            return devices
            
        except Exception as e:
            logger.error(f"Failed to load devices configuration: {e}")
            return []
    
    def _switch_kubectl_context(self, deployment_name: str) -> bool:
        """
        Switch kubectl context to the deployment's cluster.
        
        Args:
            deployment_name: Name of the deployment
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Attempt to switch context - assuming context name matches deployment name
            logger.info(f"Switching kubectl context to '{deployment_name}'")
            result = subprocess.run(
                ['kubectl', 'config', 'use-context', deployment_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully switched to context '{deployment_name}'")
                return True
            else:
                logger.error(f"Failed to switch context: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout while switching kubectl context")
            return False
        except Exception as e:
            logger.error(f"Error switching kubectl context: {e}")
            return False
    
    def _extract_query_frames(
        self,
        device_id: str,
        stabilize_time: int = 30,
        timeout: Optional[int] = None,
        active_ptz_stops: Optional[List[Tuple[float, float, float]]] = None
    ) -> Optional[Dict]:
        """
        Extract query frames from camera.
        
        Args:
            device_id: Camera device identifier
            stabilize_time: Seconds of stability required
            timeout: Optional timeout in seconds
            active_ptz_stops: Optional list of PTZ positions for active mode
            
        Returns:
            Query extraction result dict or None if failed
        """
        try:
            logger.info(f"Extracting query frames from {device_id}")
            result = extract_query(
                device_id=device_id,
                stabilize_time=stabilize_time,
                timeout=timeout,
                active_ptz_stops=active_ptz_stops
            )
            logger.info(f"Successfully extracted {len(result['camera_frames'])} frame(s)")
            return result
            
        except TimeoutError as e:
            logger.error(f"Query extraction timeout: {e}")
            return None
        except Exception as e:
            logger.error(f"Query extraction failed: {e}", exc_info=True)
            return None
    
    def _run_calibration(
        self,
        deployment_name: str,
        device_id: str,
        query_folder: str,
        query_manifest_path: str
    ) -> Optional[Dict]:
        """
        Run calibration algorithm.
        
        Args:
            deployment_name: Deployment name
            device_id: Device ID
            query_folder: Path to query frames directory
            query_manifest_path: Path to query manifest JSON
            
        Returns:
            Calibration result dict or None if failed
        """
        try:
            logger.info(f"Running calibration for {deployment_name}/{device_id}")
            result = self.calibration_runner.run_full_calibration_pipeline(
                deployment_name=deployment_name,
                device_id=device_id,
                query_folder=query_folder,
                query_manifest_path=query_manifest_path,
                camera_matrix=self.camera_matrix,
                dist_coeff=self.dist_coeff,
                r_align=self.r_align
            )
            
            if result:
                logger.info(
                    f"Calibration successful - "
                    f"Yaw: {result.median_yaw:.3f}°, "
                    f"Pitch: {result.median_pitch:.3f}°, "
                    f"Roll: {result.median_roll:.3f}°"
                )
            else:
                logger.warning("Calibration returned no result")
            
            return result
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}", exc_info=True)
            return None
    
    def _upload_query_files_to_s3(
        self,
        temp_dir: str,
        deployment_name: str,
        device_id: str,
        timestamp: str
    ) -> Optional[str]:
        """
        Upload query files to S3.
        
        Args:
            temp_dir: Temporary directory with query files
            deployment_name: Deployment name
            device_id: Device ID
            timestamp: ISO timestamp for the query scan
            
        Returns:
            S3 path (URI) to uploaded files or None if failed
        """
        try:
            logger.info(f"Uploading query files to S3 for {deployment_name}/{device_id}")
            
            # Upload all files from the temporary directory
            uploaded_files = self.aws_integration.upload_directory(
                local_dir_path=temp_dir,
                deployment_name=deployment_name,
                camera_name=device_id,
                scan_type="query_scan",
                data_type="images",
                timestamp=timestamp,
                file_pattern="*"
            )
            
            if uploaded_files:
                # Return the base S3 path (directory)
                s3_base_path = self.aws_integration.get_s3_path(
                    deployment_name=deployment_name,
                    camera_name=device_id,
                    scan_type="query_scan",
                    data_type="images",
                    timestamp=timestamp
                )
                logger.info(f"Successfully uploaded {len(uploaded_files)} file(s) to {s3_base_path}")
                return s3_base_path
            else:
                logger.error("No files were uploaded to S3")
                return None
                
        except Exception as e:
            logger.error(f"Failed to upload query files to S3: {e}", exc_info=True)
            return None
    
    def _write_result_to_athena(
        self,
        deployment_name: str,
        device_id: str,
        timestamp: datetime,
        calibration_result: Optional[Dict],
        capture_positions: List[Dict],
        files_location: str,
        success: bool,
        failure_log: str = ""
    ) -> bool:
        """
        Write calibration result to Athena.
        
        Args:
            deployment_name: Deployment name
            device_id: Device ID
            timestamp: Timestamp of calibration
            calibration_result: Calibration result (if successful)
            capture_positions: List of capture positions
            files_location: S3 path to files
            success: Whether calibration succeeded
            failure_log: Error message if failed
            
        Returns:
            True if write succeeded, False otherwise
        """
        try:
            # Extract offsets from result or use zeros for failures
            if calibration_result and success:
                pitch = calibration_result.median_pitch
                yaw = calibration_result.median_yaw
                roll = calibration_result.median_roll
            else:
                pitch = 0.0
                yaw = 0.0
                roll = 0.0
            
            # Determine mode from capture_positions
            mode = "passive" if len(capture_positions) == 1 else "active"
            
            logger.info(f"Writing result to Athena for {deployment_name}/{device_id}")
            query_id = self.aws_integration.write_calibration_result(
                deployment_name=deployment_name,
                device_id=device_id,
                timestamp=timestamp,
                pitch_offset=pitch,
                yaw_offset=yaw,
                roll_offset=roll,
                mode=mode,
                capture_positions=capture_positions,
                files_location=files_location,
                success=success,
                failure_log=failure_log
            )
            
            if query_id:
                logger.info(f"Successfully wrote result to Athena (query_id: {query_id})")
                return True
            else:
                logger.error("Failed to write result to Athena")
                return False
                
        except Exception as e:
            logger.error(f"Error writing to Athena: {e}", exc_info=True)
            return False
    
    def _send_slack_notification(
        self,
        deployment_name: str,
        device_id: str,
        calibration_result: Optional[Dict],
        success: bool,
        timestamp: str,
        failure_logs: Optional[List[str]] = None
    ) -> bool:
        """
        Send Slack notification.
        
        Args:
            deployment_name: Deployment name
            device_id: Device ID
            calibration_result: Calibration result (if successful)
            success: Whether calibration succeeded
            timestamp: ISO timestamp
            failure_logs: List of error messages if failed
            
        Returns:
            True if notification sent, False otherwise
        """
        try:
            # Extract offsets
            if calibration_result and success:
                pitch = calibration_result.median_pitch
                yaw = calibration_result.median_yaw
                roll = calibration_result.median_roll
            else:
                pitch = 0.0
                yaw = 0.0
                roll = 0.0
            
            logger.info(f"Sending Slack notification for {deployment_name}/{device_id}")
            result = self.slack_notifier.send_calibration_report(
                deployment=deployment_name,
                device_id=device_id,
                pitch=pitch,
                yaw=yaw,
                roll=roll,
                mode="passive",  # We can enhance this later
                success=success,
                timestamp=timestamp,
                failure_logs=failure_logs
            )
            
            if result:
                logger.info("Successfully sent Slack notification")
            else:
                logger.warning("Failed to send Slack notification")
            
            return result
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}", exc_info=True)
            return False
    
    def process_device(
        self,
        deployment_name: str,
        device_id: str,
        stabilize_time: int = 30,
        timeout: Optional[int] = None,
        active_ptz_stops: Optional[List[Tuple[float, float, float]]] = None
    ) -> bool:
        """
        Process a single device through the full calibration pipeline.
        
        Args:
            deployment_name: Deployment/cluster name
            device_id: Camera device identifier
            stabilize_time: Seconds of stability required for passive mode
            timeout: Optional timeout for query extraction
            active_ptz_stops: Optional PTZ positions for active mode
            
        Returns:
            True if successful, False otherwise
        """
        timestamp = datetime.utcnow()
        timestamp_str = timestamp.isoformat()
        
        logger.info("=" * 80)
        logger.info(f"Processing device: {deployment_name}/{device_id}")
        logger.info("=" * 80)
        
        failure_logs = []
        temp_dir = None
        
        try:
            # Step 1: Switch kubectl context
            if not self._switch_kubectl_context(deployment_name):
                failure_logs.append(f"Failed to switch kubectl context to '{deployment_name}'")
                raise RuntimeError("Kubectl context switch failed")
            
            # Step 2: Extract query frames
            query_result = self._extract_query_frames(
                device_id=device_id,
                stabilize_time=stabilize_time,
                timeout=timeout,
                active_ptz_stops=active_ptz_stops
            )
            
            if not query_result:
                failure_logs.append("Query frame extraction failed or timed out")
                raise RuntimeError("Query extraction failed")
            
            temp_dir = query_result['temp_dir']
            manifest_path = query_result['manifest_path']
            capture_positions = [
                {"pan": pos[0], "tilt": pos[1], "zoom": pos[2]}
                for pos in query_result['capture_positions']
            ]
            
            # Step 3: Run calibration algorithm
            calibration_result = self._run_calibration(
                deployment_name=deployment_name,
                device_id=device_id,
                query_folder=temp_dir,
                query_manifest_path=manifest_path
            )
            
            if not calibration_result:
                failure_logs.append("Calibration algorithm failed to produce results")
                raise RuntimeError("Calibration failed")
            
            # Step 4: Upload query files to S3
            s3_location = self._upload_query_files_to_s3(
                temp_dir=temp_dir,
                deployment_name=deployment_name,
                device_id=device_id,
                timestamp=timestamp_str
            )
            
            if not s3_location:
                failure_logs.append("Failed to upload query files to S3")
                # Don't fail the entire process if S3 upload fails
                s3_location = f"s3://{self.s3_bucket}/{deployment_name}/{device_id}/query_scan/{timestamp_str}/"
            
            # Step 5: Write results to Athena
            athena_success = self._write_result_to_athena(
                deployment_name=deployment_name,
                device_id=device_id,
                timestamp=timestamp,
                calibration_result=calibration_result,
                capture_positions=capture_positions,
                files_location=s3_location,
                success=True,
                failure_log=""
            )
            
            if not athena_success:
                logger.warning("Failed to write results to Athena, but continuing")
            
            # Step 6: Send Slack notification
            self._send_slack_notification(
                deployment_name=deployment_name,
                device_id=device_id,
                calibration_result=calibration_result,
                success=True,
                timestamp=timestamp_str,
                failure_logs=None
            )
            
            logger.info(f"Successfully processed {deployment_name}/{device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {deployment_name}/{device_id}: {e}", exc_info=True)
            
            # Log failure to Athena
            failure_log_str = " | ".join(failure_logs) if failure_logs else str(e)
            
            try:
                # Try to determine S3 location even on failure
                s3_location = f"s3://{self.s3_bucket}/{deployment_name}/{device_id}/query_scan/{timestamp_str}/"
                
                # If we have temp_dir, try to upload partial data
                if temp_dir and os.path.exists(temp_dir):
                    uploaded_location = self._upload_query_files_to_s3(
                        temp_dir=temp_dir,
                        deployment_name=deployment_name,
                        device_id=device_id,
                        timestamp=timestamp_str
                    )
                    if uploaded_location:
                        s3_location = uploaded_location
                
                # Write failure to Athena
                self._write_result_to_athena(
                    deployment_name=deployment_name,
                    device_id=device_id,
                    timestamp=timestamp,
                    calibration_result=None,
                    capture_positions=[],
                    files_location=s3_location,
                    success=False,
                    failure_log=failure_log_str
                )
                
                # Send failure notification to Slack
                self._send_slack_notification(
                    deployment_name=deployment_name,
                    device_id=device_id,
                    calibration_result=None,
                    success=False,
                    timestamp=timestamp_str,
                    failure_logs=failure_logs if failure_logs else [str(e)]
                )
                
            except Exception as notification_error:
                logger.error(f"Failed to log/notify failure: {notification_error}")
            
            return False
            
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")
    
    def run_all_devices(
        self,
        stabilize_time: int = 30,
        timeout: Optional[int] = None,
        active_ptz_stops: Optional[List[Tuple[float, float, float]]] = None
    ) -> Dict[str, bool]:
        """
        Run calibration monitoring for all devices.
        
        Args:
            stabilize_time: Seconds of stability required for passive mode
            timeout: Optional timeout for query extraction
            active_ptz_stops: Optional PTZ positions for active mode
            
        Returns:
            Dictionary mapping device IDs to success status
        """
        logger.info("Starting calibration monitoring for all devices")
        logger.info(f"Total devices to process: {len(self.devices)}")
        
        results = {}
        
        for device in self.devices:
            deployment_name = device['deployment_name']
            device_id = device['device_id']
            
            success = self.process_device(
                deployment_name=deployment_name,
                device_id=device_id,
                stabilize_time=stabilize_time,
                timeout=timeout,
                active_ptz_stops=active_ptz_stops
            )
            
            results[f"{deployment_name}/{device_id}"] = success
        
        # Summary
        logger.info("=" * 80)
        logger.info("Calibration Monitoring Complete - Summary")
        logger.info("=" * 80)
        
        successful = sum(1 for v in results.values() if v)
        failed = len(results) - successful
        
        logger.info(f"Total devices: {len(results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        
        for device_key, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            logger.info(f"  {device_key}: {status}")
        
        return results


def main():
    """Main entry point for the orchestration script."""
    parser = argparse.ArgumentParser(
        description="PTZ Camera Calibration Monitoring Orchestration Script"
    )
    
    parser.add_argument(
        "--devices-yaml",
        type=str,
        default="devices.yaml",
        help="Path to devices.yaml configuration file"
    )
    
    parser.add_argument(
        "--camera-intrinsics",
        type=str,
        default="camera_intrinsics/calibration.npz",
        help="Path to camera calibration .npz file"
    )
    
    parser.add_argument(
        "--r-align",
        type=str,
        default="r_align/10NOV2025_REF1_HOMOGRAPHY.npy",
        help="Path to R_align .npy file"
    )
    
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default="camera-calibration-monitoring",
        help="S3 bucket name for data storage"
    )
    
    parser.add_argument(
        "--aws-region",
        type=str,
        default="us-east-1",
        help="AWS region"
    )
    
    parser.add_argument(
        "--stabilize-time",
        type=int,
        default=30,
        help="Seconds of stability required for passive mode (default: 30)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional timeout in seconds for query extraction (default: None)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Process only a specific device (format: deployment_name/device_id)"
    )
    
    parser.add_argument(
        "--active-mode",
        action="store_true",
        help="Use active mode with predefined PTZ stops (not implemented yet)"
    )
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    try:
        orchestrator = CalibrationMonitoringOrchestrator(
            devices_yaml_path=args.devices_yaml,
            camera_intrinsics_path=args.camera_intrinsics,
            r_align_path=args.r_align,
            s3_bucket=args.s3_bucket,
            aws_region=args.aws_region
        )
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}", exc_info=True)
        sys.exit(1)
    
    # Process devices
    if args.device:
        # Process single device
        parts = args.device.split('/')
        if len(parts) != 2:
            logger.error("Device must be in format 'deployment_name/device_id'")
            sys.exit(1)
        
        deployment_name, device_id = parts
        success = orchestrator.process_device(
            deployment_name=deployment_name,
            device_id=device_id,
            stabilize_time=args.stabilize_time,
            timeout=args.timeout
        )
        
        sys.exit(0 if success else 1)
    else:
        # Process all devices
        results = orchestrator.run_all_devices(
            stabilize_time=args.stabilize_time,
            timeout=args.timeout
        )
        
        # Exit with error code if any device failed
        any_failed = any(not success for success in results.values())
        sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()


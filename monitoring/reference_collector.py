#!/usr/bin/env python3
"""
Reference Collection Module for PTZ Camera Calibration Monitoring.

This module extracts camera scanning logic from scan.py and adapts it for
collecting reference scans from multiple cameras across different clusters.

The reference collector:
1. Iterates through devices defined in devices.yaml
2. Switches kubectl context to appropriate cluster
3. Runs PTZ grid camera scan
4. Extracts features from captured images
5. Uploads images and features to S3
"""

import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import torch
import yaml
from botocore.exceptions import ClientError

# Import scan.py helpers - these are used by the extracted capture logic
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from port_forward_utils import start_port_forwards, stop_port_forwards
except ImportError:
    # Fallback if port_forward_utils is not available as a package
    # This will be needed if the helpers aren't set up yet
    logging.warning("Could not import port_forward_utils, port forwarding functions may not be available")
    def start_port_forwards(*args, **kwargs):
        logging.warning("start_port_forwards not implemented")
    def stop_port_forwards(*args, **kwargs):
        logging.warning("stop_port_forwards not implemented")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("reference-collector")


class ReferenceCollector:
    """
    Manages collection of reference scans from PTZ cameras across multiple deployments.
    
    This class handles:
    - Loading device configurations
    - Switching kubernetes contexts
    - Port forwarding to camera services
    - Capturing reference image grids
    - Extracting features from images
    - Uploading to S3 storage
    """
    
    def __init__(
        self,
        devices_yaml_path: str,
        secrets_path: Optional[str] = None,
        s3_bucket: str = "camera-calibration-monitoring",
        aws_region: str = "us-east-1"
    ):
        """
        Initialize the ReferenceCollector.
        
        Args:
            devices_yaml_path: Path to devices.yaml configuration file
            secrets_path: Path to secrets.json file (default: looks in project root)
            s3_bucket: S3 bucket name for storing reference data
            aws_region: AWS region for S3
        """
        self.devices_yaml_path = Path(devices_yaml_path)
        self.s3_bucket = s3_bucket
        self.aws_region = aws_region
        
        # Load device configurations
        self.devices = self._load_devices()
        
        # Load secrets for camera connection
        self.secrets = self._load_secrets(secrets_path)
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3', region_name=aws_region)
        
        logger.info(f"Initialized ReferenceCollector with {len(self.devices)} device(s)")
    
    def _load_devices(self) -> List[Dict]:
        """Load device configurations from devices.yaml."""
        if not self.devices_yaml_path.exists():
            raise FileNotFoundError(f"Devices file not found: {self.devices_yaml_path}")
        
        with open(self.devices_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if not data:
            raise ValueError("devices.yaml is empty or invalid")
        
        logger.info(f"Loaded {len(data)} device(s) from {self.devices_yaml_path}")
        return data
    
    def _load_secrets(self, secrets_path: Optional[Path] = None) -> Dict:
        """Load secrets from secrets.json file."""
        if secrets_path is None:
            # Look for secrets.json in project root (parent of this module)
            secrets_path = Path(__file__).resolve().parent.parent / "secrets.json"
        else:
            secrets_path = Path(secrets_path)
        
        if not secrets_path.exists():
            logger.warning(f"No secrets file found at {secrets_path}, using defaults")
            return {}
        
        try:
            with open(secrets_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load secrets from {secrets_path}: {e}")
            return {}
    
    def _switch_kubectl_context(self, cluster_name: str) -> bool:
        """
        Switch kubectl context to the specified cluster.
        
        Args:
            cluster_name: Name of the kubernetes cluster/context
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Switching kubectl context to: {cluster_name}")
            result = subprocess.run(
                ["kubectl", "config", "use-context", cluster_name],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Successfully switched to context: {cluster_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to switch kubectl context: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("kubectl command not found. Please ensure kubectl is installed and in PATH")
            return False
    
    def _capture_reference_scan(
        self,
        camera_name: str,
        variant: str = "main",
        output_dir: Optional[str] = None,
        horizontal_stops: int = 12,
        vertical_stops: int = 5,
        pitch_min: float = -70.0,
        pitch_max: float = 10.0
    ) -> Optional[Dict]:
        """
        Capture a reference scan from the camera using PTZ grid pattern.
        
        This method imports and uses the capture_views_and_render function from scan.py.
        
        Args:
            camera_name: Name of the camera (e.g., "onvifcam-1")
            variant: Camera variant ("main" or "thermal")
            output_dir: Directory to save captured frames
            horizontal_stops: Number of horizontal (yaw) stops
            vertical_stops: Number of vertical (pitch) stops
            pitch_min: Minimum pitch angle for capture grid
            pitch_max: Maximum pitch angle for capture grid
            
        Returns:
            Dict with capture metadata (frames_dir, manifest_path) or None if failed
        """
        try:
            # Import the capture function from scan.py
            from scan import capture_views_and_render
            
            # Create temporary output directory if not specified
            if output_dir is None:
                output_dir = tempfile.mkdtemp(prefix=f"ref_scan_{camera_name}_")
            
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"Starting reference scan for camera: {camera_name} (variant: {variant})")
            logger.info(f"Grid: {horizontal_stops}x{vertical_stops}, pitch range: [{pitch_min}, {pitch_max}]")
            
            # Call the capture function from scan.py
            # Note: This uses the same logic as scan.py's capture_views_and_render
            capture_views_and_render(
                camera_name=camera_name,
                variant=variant,
                output_dir=output_dir,
                horizontal_stops=horizontal_stops,
                vertical_stops=vertical_stops,
                secrets=self.secrets,
                pitch_min=pitch_min,
                pitch_max=pitch_max,
                tolerance=0.15,
                zoom_tolerance=0.05,
                use_rtsp=True,
                no_http_fallback=False,
                telemetry_crop_pct=0.1125,  # Crop telemetry overlay
                validate_ocr=True  # Enable OCR validation
            )
            
            # Find the created session directory
            # capture_views_and_render creates a timestamped directory
            session_dirs = [d for d in Path(output_dir).iterdir() if d.is_dir()]
            if not session_dirs:
                logger.error("No session directory created by capture_views_and_render")
                return None
            
            # Get the most recent session directory
            session_dir = max(session_dirs, key=lambda d: d.stat().st_mtime)
            frames_dir = session_dir / "frames"
            manifest_path = frames_dir / "manifest.json"
            
            if not manifest_path.exists():
                logger.error(f"Manifest file not found: {manifest_path}")
                return None
            
            # Load manifest to get frame count
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            logger.info(f"Successfully captured {len(manifest)} frames to {frames_dir}")
            
            return {
                "frames_dir": str(frames_dir),
                "manifest_path": str(manifest_path),
                "frame_count": len(manifest),
                "session_dir": str(session_dir)
            }
            
        except Exception as e:
            logger.error(f"Failed to capture reference scan: {e}", exc_info=True)
            return None
    
    def _extract_features(
        self,
        frames_dir: str,
        manifest_path: str,
        output_features_dir: Optional[str] = None
    ) -> Optional[str]:
        """
        Extract DISK features from captured frames and save in COLMAP format.
        
        This uses logic adapted from ptz_georeg/save_features_to_colmap.py.
        
        Args:
            frames_dir: Directory containing captured frame images
            manifest_path: Path to manifest.json with frame metadata
            output_features_dir: Directory to save extracted features
            
        Returns:
            Path to features directory or None if failed
        """
        try:
            # Import feature extraction utilities
            from ptz_georeg.utils import save_reference_map_to_COLMAP_txts
            
            if output_features_dir is None:
                output_features_dir = str(Path(frames_dir).parent / "features")
            
            os.makedirs(output_features_dir, exist_ok=True)
            
            logger.info(f"Extracting features from frames in: {frames_dir}")
            logger.info(f"Saving features to: {output_features_dir}")
            
            # Load manifest to create reference panorama format
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Convert manifest to reference panorama format expected by save_reference_map_to_COLMAP_txts
            reference_data = {
                "frames": []
            }
            
            for frame_filename, frame_data in manifest.items():
                frame_path = Path(frames_dir) / frame_filename
                if not frame_path.exists():
                    logger.warning(f"Frame file not found: {frame_path}")
                    continue
                
                reference_data["frames"].append({
                    "filename": frame_filename,
                    "image_path": str(frame_path),
                    "attitude": frame_data["attitude"],
                    "field_of_view_h": frame_data["field_of_view_h"],
                    "field_of_view_v": frame_data["field_of_view_v"],
                    "zoom": frame_data.get("zoom", 0.0)
                })
            
            # Save reference panorama JSON for feature extraction
            reference_json_path = Path(output_features_dir) / "reference_panorama.json"
            with open(reference_json_path, 'w') as f:
                json.dump(reference_data, f, indent=2)
            
            # Extract features and save to COLMAP format
            save_reference_map_to_COLMAP_txts(str(reference_json_path), output_features_dir)
            
            # Count extracted feature files
            feature_files = list(Path(output_features_dir).glob("*.txt"))
            logger.info(f"Successfully extracted features for {len(feature_files)} frames")
            
            return output_features_dir
            
        except Exception as e:
            logger.error(f"Failed to extract features: {e}", exc_info=True)
            return None
    
    def _upload_to_s3(
        self,
        local_dir: str,
        s3_prefix: str,
        file_extensions: Optional[List[str]] = None
    ) -> bool:
        """
        Upload files from local directory to S3.
        
        Args:
            local_dir: Local directory containing files to upload
            s3_prefix: S3 key prefix (e.g., "deployment/camera/reference_scan/images")
            file_extensions: Optional list of file extensions to filter (e.g., [".png", ".jpg"])
            
        Returns:
            True if successful, False otherwise
        """
        try:
            local_path = Path(local_dir)
            if not local_path.exists():
                logger.error(f"Local directory does not exist: {local_dir}")
                return False
            
            # Get list of files to upload
            if file_extensions:
                files = []
                for ext in file_extensions:
                    files.extend(local_path.glob(f"**/*{ext}"))
            else:
                files = [f for f in local_path.rglob("*") if f.is_file()]
            
            if not files:
                logger.warning(f"No files found to upload in: {local_dir}")
                return True
            
            logger.info(f"Uploading {len(files)} file(s) to s3://{self.s3_bucket}/{s3_prefix}")
            
            upload_count = 0
            for file_path in files:
                # Calculate relative path for S3 key
                rel_path = file_path.relative_to(local_path)
                s3_key = f"{s3_prefix}/{rel_path}".replace("\\", "/")  # Ensure forward slashes
                
                try:
                    self.s3_client.upload_file(
                        str(file_path),
                        self.s3_bucket,
                        s3_key
                    )
                    upload_count += 1
                    logger.debug(f"Uploaded: {file_path.name} -> s3://{self.s3_bucket}/{s3_key}")
                except ClientError as e:
                    logger.error(f"Failed to upload {file_path.name}: {e}")
                    return False
            
            logger.info(f"Successfully uploaded {upload_count} file(s) to S3")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}", exc_info=True)
            return False
    
    def collect_reference_for_device(
        self,
        device_name: str,
        camera_name: str,
        horizontal_stops: int = 12,
        vertical_stops: int = 5
    ) -> bool:
        """
        Collect reference scan for a specific device and camera.
        
        Args:
            device_name: Name of the device/deployment (e.g., "gan-shomron-dell")
            camera_name: Name of the camera (e.g., "onvifcam-1")
            horizontal_stops: Number of horizontal grid stops
            vertical_stops: Number of vertical grid stops
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"=== Starting reference collection for {device_name}/{camera_name} ===")
        
        try:
            # Switch kubectl context
            if not self._switch_kubectl_context(device_name):
                logger.error(f"Failed to switch to context: {device_name}")
                return False
            
            # Start port forwards
            logger.info("Starting port forwards...")
            required_ports = [1883, 8800, 8554, 8000]  # MQTT, HTTP, RTSP, ONVIF
            start_port_forwards(required_ports, onvif_device_name=camera_name)
            
            try:
                # Create temporary working directory
                with tempfile.TemporaryDirectory(prefix=f"ref_collect_{device_name}_{camera_name}_") as temp_dir:
                    logger.info(f"Working directory: {temp_dir}")
                    
                    # Capture reference scan
                    capture_result = self._capture_reference_scan(
                        camera_name=camera_name,
                        variant="main",
                        output_dir=temp_dir,
                        horizontal_stops=horizontal_stops,
                        vertical_stops=vertical_stops
                    )
                    
                    if not capture_result:
                        logger.error("Reference scan capture failed")
                        return False
                    
                    # Extract features
                    features_dir = self._extract_features(
                        frames_dir=capture_result["frames_dir"],
                        manifest_path=capture_result["manifest_path"]
                    )
                    
                    if not features_dir:
                        logger.error("Feature extraction failed")
                        return False
                    
                    # Upload images to S3
                    images_s3_prefix = f"{device_name}/{camera_name}/reference_scan/images"
                    if not self._upload_to_s3(
                        local_dir=capture_result["frames_dir"],
                        s3_prefix=images_s3_prefix,
                        file_extensions=[".png", ".jpg", ".jpeg"]
                    ):
                        logger.error("Failed to upload images to S3")
                        return False
                    
                    # Upload manifest to S3
                    manifest_s3_prefix = f"{device_name}/{camera_name}/reference_scan"
                    manifest_dir = Path(capture_result["manifest_path"]).parent
                    if not self._upload_to_s3(
                        local_dir=str(manifest_dir),
                        s3_prefix=manifest_s3_prefix,
                        file_extensions=[".json"]
                    ):
                        logger.error("Failed to upload manifest to S3")
                        return False
                    
                    # Upload features to S3
                    features_s3_prefix = f"{device_name}/{camera_name}/reference_scan/features"
                    if not self._upload_to_s3(
                        local_dir=features_dir,
                        s3_prefix=features_s3_prefix,
                        file_extensions=[".txt", ".json"]
                    ):
                        logger.error("Failed to upload features to S3")
                        return False
                    
                    logger.info(f"✓ Successfully collected and uploaded reference for {device_name}/{camera_name}")
                    return True
                    
            finally:
                # Always stop port forwards
                logger.info("Stopping port forwards...")
                stop_port_forwards()
        
        except Exception as e:
            logger.error(f"Failed to collect reference for {device_name}/{camera_name}: {e}", exc_info=True)
            return False
    
    def collect_all_references(
        self,
        horizontal_stops: int = 12,
        vertical_stops: int = 5
    ) -> Dict[str, Dict[str, bool]]:
        """
        Collect reference scans for all devices and cameras defined in devices.yaml.
        
        Args:
            horizontal_stops: Number of horizontal grid stops
            vertical_stops: Number of vertical grid stops
            
        Returns:
            Dict mapping device names to camera results
            Example: {"gan-shomron-dell": {"onvifcam-1": True, "onvifcam-2": False}}
        """
        logger.info("=== Starting reference collection for all devices ===")
        
        results = {}
        
        for device in self.devices:
            device_name = device.get("name")
            if not device_name:
                logger.warning("Device missing 'name' field, skipping")
                continue
            
            cameras = device.get("cameras", [])
            if not cameras:
                logger.warning(f"Device {device_name} has no cameras defined, skipping")
                continue
            
            device_results = {}
            
            for camera in cameras:
                camera_name = camera.get("name")
                if not camera_name:
                    logger.warning(f"Camera in device {device_name} missing 'name' field, skipping")
                    continue
                
                # Collect reference for this camera
                success = self.collect_reference_for_device(
                    device_name=device_name,
                    camera_name=camera_name,
                    horizontal_stops=horizontal_stops,
                    vertical_stops=vertical_stops
                )
                
                device_results[camera_name] = success
            
            results[device_name] = device_results
        
        # Print summary
        logger.info("\n=== Reference Collection Summary ===")
        total_cameras = 0
        successful_cameras = 0
        
        for device_name, camera_results in results.items():
            logger.info(f"\n{device_name}:")
            for camera_name, success in camera_results.items():
                total_cameras += 1
                status = "✓ SUCCESS" if success else "✗ FAILED"
                if success:
                    successful_cameras += 1
                logger.info(f"  {camera_name}: {status}")
        
        logger.info(f"\nOverall: {successful_cameras}/{total_cameras} cameras successful")
        
        return results


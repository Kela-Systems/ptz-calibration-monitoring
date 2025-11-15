#!/usr/bin/env python3
"""
Calibration Runner Module for PTZ Camera Calibration Monitoring.

This module wraps the existing calibration algorithm from ptz_georeg/utils.py
to enable offset calculation against reference features stored in S3.

Key Functions:
- Load reference features from S3
- Run calibration algorithm on query frames
- Extract pitch/yaw/roll offsets
- Return structured results with confidence metrics
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import numpy as np
import torch
from botocore.exceptions import ClientError

# Import from ptz_georeg
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ptz_georeg.utils import (
    calculate_offsets_from_visual_matches,
    calculate_final_offset,
    MatchingMethod,
    GeometryModel
)
from ptz_georeg.save_features_to_colmap import parse_colmap_txt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("calibration-runner")


@dataclass
class CalibrationConfig:
    """Configuration for calibration algorithm parameters."""
    
    # Matching parameters
    min_match_count: int = 15
    ransac_threshold: float = 5.0
    matching_method: MatchingMethod = MatchingMethod.SUPERGLUE
    geometry_model: GeometryModel = GeometryModel.HOMOGRAPHY
    
    # Feature extraction parameters
    num_of_deeplearning_features: int = 2048
    resize_scale_deep_learning: float = 1.0
    max_xy_norm: float = 0.1
    min_overlapping_ratio: float = 0.3
    
    # ROI (region of interest) as [x_min, y_min, x_max, y_max] normalized to [0, 1]
    roi: List[float] = None
    
    def __post_init__(self):
        if self.roi is None:
            # Default ROI excludes 10% border on all sides
            self.roi = [0.1, 0.1, 0.9, 0.9]


@dataclass
class CalibrationResult:
    """Structured result from calibration algorithm."""
    
    # Median offsets
    median_yaw: float
    median_pitch: float
    median_roll: float
    
    # Weighted mean offsets
    weighted_mean_yaw: float
    weighted_mean_pitch: float
    weighted_mean_roll: float
    
    # Standard deviations
    std_yaw: float
    std_pitch: float
    std_roll: float
    
    # Confidence metrics
    mean_angular_distance: float
    num_high_confidence_matches: int
    
    # Raw offset data for detailed analysis
    all_offsets: List[Dict]
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary."""
        return {
            "median_offsets": {
                "yaw": float(self.median_yaw),
                "pitch": float(self.median_pitch),
                "roll": float(self.median_roll)
            },
            "weighted_mean_offsets": {
                "yaw": float(self.weighted_mean_yaw),
                "pitch": float(self.weighted_mean_pitch),
                "roll": float(self.weighted_mean_roll)
            },
            "standard_deviations": {
                "yaw": float(self.std_yaw),
                "pitch": float(self.std_pitch),
                "roll": float(self.std_roll)
            },
            "confidence_metrics": {
                "mean_angular_distance": float(self.mean_angular_distance),
                "num_high_confidence_matches": int(self.num_high_confidence_matches)
            },
            "all_offsets": self.all_offsets
        }


class CalibrationRunner:
    """
    Manages calibration offset calculation for PTZ cameras.
    
    This class:
    - Downloads reference features from S3
    - Loads reference map into memory
    - Runs calibration algorithm on query frames
    - Returns structured results with offset measurements
    """
    
    def __init__(
        self,
        s3_bucket: str = "camera-calibration-monitoring",
        aws_region: str = "us-east-1",
        config: Optional[CalibrationConfig] = None
    ):
        """
        Initialize CalibrationRunner.
        
        Args:
            s3_bucket: S3 bucket name for reference data
            aws_region: AWS region
            config: Calibration configuration (uses defaults if None)
        """
        self.s3_bucket = s3_bucket
        self.aws_region = aws_region
        self.s3_client = boto3.client('s3', region_name=aws_region)
        self.config = config or CalibrationConfig()
        
        logger.info(f"Initialized CalibrationRunner for bucket: {s3_bucket}")
    
    def _download_s3_directory(
        self,
        s3_prefix: str,
        local_dir: str
    ) -> bool:
        """
        Download all files from S3 prefix to local directory.
        
        Args:
            s3_prefix: S3 key prefix to download
            local_dir: Local directory to save files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(local_dir, exist_ok=True)
            
            # List all objects in the prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.s3_bucket, Prefix=s3_prefix)
            
            file_count = 0
            for page in pages:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    # Skip if it's a "directory" marker
                    if s3_key.endswith('/'):
                        continue
                    
                    # Calculate relative path and local file path
                    relative_path = s3_key[len(s3_prefix):].lstrip('/')
                    local_file_path = os.path.join(local_dir, relative_path)
                    
                    # Create parent directories if needed
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    
                    # Download file
                    logger.debug(f"Downloading {s3_key} to {local_file_path}")
                    self.s3_client.download_file(
                        self.s3_bucket,
                        s3_key,
                        local_file_path
                    )
                    file_count += 1
            
            logger.info(f"Downloaded {file_count} file(s) from s3://{self.s3_bucket}/{s3_prefix}")
            return file_count > 0
            
        except ClientError as e:
            logger.error(f"Failed to download from S3: {e}")
            return False
    
    def _load_reference_map_from_colmap(
        self,
        features_dir: str,
        reference_json_path: str
    ) -> Optional[List[Dict]]:
        """
        Load reference map from COLMAP-format feature files.
        
        Args:
            features_dir: Directory containing COLMAP .txt feature files
            reference_json_path: Path to reference_panorama.json with telemetry data
            
        Returns:
            Reference map in format expected by calculate_offsets_from_visual_matches
        """
        try:
            # Load reference panorama JSON to get telemetry data
            with open(reference_json_path, 'r') as f:
                reference_data = json.load(f)
            
            # Build reference map
            reference_map = []
            device = torch.device('cpu')  # Load on CPU for memory efficiency
            
            for frame in reference_data.get('frames', []):
                filename = frame['filename']
                # COLMAP feature file has same name but .txt extension
                feature_file = os.path.join(
                    features_dir,
                    filename.replace('.png', '.txt').replace('.jpg', '.txt')
                )
                
                if not os.path.exists(feature_file):
                    logger.warning(f"Feature file not found: {feature_file}")
                    continue
                
                # Parse COLMAP feature file
                keypoints, descriptors = parse_colmap_txt(feature_file)
                
                # Move to appropriate device
                keypoints = keypoints.to(device)
                descriptors = descriptors.to(device)
                
                # Extract telemetry
                attitude = frame['attitude']
                telemetry = {
                    'yaw': attitude['yaw'],
                    'pitch': attitude['pitch'],
                    'roll': attitude['roll']
                }
                
                # Get image dimensions (height, width)
                # We need this for SUPERGLUE matching
                # Default to common PTZ camera resolution if not available
                hw = (1080, 1920)  # Default HD resolution
                
                # Build reference frame entry
                frame_entry = {
                    'filename': filename,
                    'telemetry': telemetry,
                    'keypoints': keypoints,
                    'descriptors': descriptors,
                    'hw': hw
                }
                
                reference_map.append(frame_entry)
            
            logger.info(f"Loaded reference map with {len(reference_map)} frames")
            return reference_map
            
        except Exception as e:
            logger.error(f"Failed to load reference map: {e}", exc_info=True)
            return None
    
    def load_reference_features(
        self,
        deployment_name: str,
        device_id: str
    ) -> Optional[Tuple[List[Dict], str]]:
        """
        Load reference features from S3.
        
        Args:
            deployment_name: Deployment name (e.g., "production")
            device_id: Camera device ID
            
        Returns:
            Tuple of (reference_map, temp_dir) or None if failed
            The temp_dir should be cleaned up by the caller
        """
        try:
            # Create temporary directory for downloaded files
            temp_dir = tempfile.mkdtemp(prefix="calibration_ref_")
            
            # S3 paths
            features_prefix = f"{deployment_name}/{device_id}/reference_scan/features/"
            
            # Download features directory
            features_local_dir = os.path.join(temp_dir, "features")
            if not self._download_s3_directory(features_prefix, features_local_dir):
                logger.error("Failed to download reference features from S3")
                return None
            
            # Look for reference_panorama.json
            reference_json_path = os.path.join(features_local_dir, "reference_panorama.json")
            if not os.path.exists(reference_json_path):
                logger.error(f"reference_panorama.json not found in {features_local_dir}")
                return None
            
            # Load reference map
            reference_map = self._load_reference_map_from_colmap(
                features_local_dir,
                reference_json_path
            )
            
            if reference_map is None:
                return None
            
            return reference_map, temp_dir
            
        except Exception as e:
            logger.error(f"Failed to load reference features: {e}", exc_info=True)
            return None
    
    def run_calibration(
        self,
        query_folder: str,
        query_manifest_path: str,
        reference_map: List[Dict],
        camera_matrix: np.ndarray,
        dist_coeff: np.ndarray,
        r_align: np.ndarray
    ) -> Optional[CalibrationResult]:
        """
        Run calibration algorithm on query frames.
        
        Args:
            query_folder: Directory containing query frame images
            query_manifest_path: Path to manifest.json for query frames
            reference_map: Reference feature map loaded from S3
            camera_matrix: Camera intrinsic matrix (3x3)
            dist_coeff: Camera distortion coefficients
            r_align: Rotation alignment matrix (3x3)
            
        Returns:
            CalibrationResult with offset measurements
        """
        try:
            logger.info("Running calibration algorithm...")
            
            # Run the main calibration algorithm
            all_offsets = calculate_offsets_from_visual_matches(
                query_folder=query_folder,
                query_manifest_path=query_manifest_path,
                reference_map=reference_map,
                min_match_count=self.config.min_match_count,
                matching_method=self.config.matching_method,
                ransac_threshold=self.config.ransac_threshold,
                camera_matrix=camera_matrix,
                dist_coeff=dist_coeff,
                geometry_model=self.config.geometry_model,
                r_align=r_align,
                roi=self.config.roi,
                num_of_deeplearning_featuers=self.config.num_of_deeplearning_features,
                resize_scale_deep_learning=self.config.resize_scale_deep_learning,
                max_xy_norm=self.config.max_xy_norm,
                min_overlapping_ratio=self.config.min_overlapping_ratio
            )
            
            if not all_offsets:
                logger.warning("No offsets calculated - no matches found")
                return None
            
            logger.info(f"Calculated {len(all_offsets)} offset measurements")
            
            # Calculate final offset statistics
            final_offset = calculate_final_offset(all_offsets)
            
            if final_offset is None:
                logger.warning("Failed to calculate final offset")
                return None
            
            # Build structured result
            result = CalibrationResult(
                median_yaw=final_offset['median_yaw'],
                median_pitch=final_offset['median_pitch'],
                median_roll=final_offset['median_roll'],
                weighted_mean_yaw=final_offset['weighted_mean_yaw'],
                weighted_mean_pitch=final_offset['weighted_mean_pitch'],
                weighted_mean_roll=final_offset['weighted_mean_roll'],
                std_yaw=final_offset['std_yaw_offset'],
                std_pitch=final_offset['std_pitch_offset'],
                std_roll=final_offset['std_roll_offset'],
                mean_angular_distance=final_offset['mean_angular_distances'],
                num_high_confidence_matches=len(all_offsets),
                all_offsets=all_offsets
            )
            
            logger.info("Calibration completed successfully")
            logger.info(f"Median offsets - Yaw: {result.median_yaw:.3f}°, "
                       f"Pitch: {result.median_pitch:.3f}°, Roll: {result.median_roll:.3f}°")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to run calibration: {e}", exc_info=True)
            return None
    
    def run_full_calibration_pipeline(
        self,
        deployment_name: str,
        device_id: str,
        query_folder: str,
        query_manifest_path: str,
        camera_matrix: np.ndarray,
        dist_coeff: np.ndarray,
        r_align: np.ndarray
    ) -> Optional[CalibrationResult]:
        """
        Run the complete calibration pipeline:
        1. Load reference features from S3
        2. Run calibration algorithm
        3. Return results
        
        Args:
            deployment_name: Deployment name
            device_id: Camera device ID
            query_folder: Directory with query images
            query_manifest_path: Path to query manifest
            camera_matrix: Camera intrinsic matrix
            dist_coeff: Camera distortion coefficients
            r_align: Rotation alignment matrix
            
        Returns:
            CalibrationResult or None if failed
        """
        temp_dir = None
        try:
            # Load reference features
            logger.info(f"Loading reference features for {deployment_name}/{device_id}")
            result = self.load_reference_features(deployment_name, device_id)
            
            if result is None:
                logger.error("Failed to load reference features")
                return None
            
            reference_map, temp_dir = result
            
            # Run calibration
            calibration_result = self.run_calibration(
                query_folder=query_folder,
                query_manifest_path=query_manifest_path,
                reference_map=reference_map,
                camera_matrix=camera_matrix,
                dist_coeff=dist_coeff,
                r_align=r_align
            )
            
            return calibration_result
            
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")


def load_camera_calibration(camera_intrinsics_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load camera intrinsics from .npz file.
    
    Args:
        camera_intrinsics_path: Path to calibration.npz file
        
    Returns:
        Tuple of (camera_matrix, dist_coeff)
    """
    data = np.load(camera_intrinsics_path)
    camera_matrix = data['camera_matrix']
    dist_coeff = data['dist_coeff']
    return camera_matrix, dist_coeff


def load_r_align(r_align_path: str) -> np.ndarray:
    """
    Load R_align rotation matrix.
    
    Args:
        r_align_path: Path to R_align .npy file
        
    Returns:
        3x3 rotation matrix
    """
    return np.load(r_align_path)


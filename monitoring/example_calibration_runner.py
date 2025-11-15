#!/usr/bin/env python3
"""
Example usage of CalibrationRunner for PTZ camera calibration monitoring.

This script demonstrates how to:
1. Load camera calibration data
2. Initialize the CalibrationRunner
3. Run calibration against reference features stored in S3
4. Process the results
"""

import logging
from pathlib import Path

from calibration_runner import (
    CalibrationRunner,
    CalibrationConfig,
    load_camera_calibration,
    load_r_align
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Example calibration run."""
    
    # Configuration
    deployment_name = "production"
    device_id = "camera-01"
    
    # Paths to local query data
    query_folder = "/path/to/query/frames"
    query_manifest_path = "/path/to/query/manifest.json"
    
    # Paths to calibration files
    project_root = Path(__file__).parent.parent
    camera_intrinsics_path = project_root / "camera_intrinsics" / "calibration.npz"
    r_align_path = project_root / "r_align" / "10NOV2025_REF1_HOMOGRAPHY.npy"
    
    # Load calibration data
    logger.info("Loading camera calibration data...")
    camera_matrix, dist_coeff = load_camera_calibration(str(camera_intrinsics_path))
    r_align = load_r_align(str(r_align_path))
    
    logger.info(f"Camera matrix:\n{camera_matrix}")
    logger.info(f"R_align shape: {r_align.shape}")
    
    # Initialize calibration runner
    config = CalibrationConfig(
        min_match_count=15,
        ransac_threshold=5.0,
        num_of_deeplearning_features=2048
    )
    
    runner = CalibrationRunner(
        s3_bucket="camera-calibration-monitoring",
        aws_region="us-east-1",
        config=config
    )
    
    # Run calibration
    logger.info(f"Running calibration for {deployment_name}/{device_id}...")
    result = runner.run_full_calibration_pipeline(
        deployment_name=deployment_name,
        device_id=device_id,
        query_folder=query_folder,
        query_manifest_path=query_manifest_path,
        camera_matrix=camera_matrix,
        dist_coeff=dist_coeff,
        r_align=r_align
    )
    
    if result is None:
        logger.error("Calibration failed!")
        return
    
    # Process results
    logger.info("=" * 60)
    logger.info("CALIBRATION RESULTS")
    logger.info("=" * 60)
    
    logger.info("\nMedian Offsets:")
    logger.info(f"  Yaw:   {result.median_yaw:>8.3f}°")
    logger.info(f"  Pitch: {result.median_pitch:>8.3f}°")
    logger.info(f"  Roll:  {result.median_roll:>8.3f}°")
    
    logger.info("\nWeighted Mean Offsets:")
    logger.info(f"  Yaw:   {result.weighted_mean_yaw:>8.3f}°")
    logger.info(f"  Pitch: {result.weighted_mean_pitch:>8.3f}°")
    logger.info(f"  Roll:  {result.weighted_mean_roll:>8.3f}°")
    
    logger.info("\nStandard Deviations:")
    logger.info(f"  Yaw:   {result.std_yaw:>8.3f}°")
    logger.info(f"  Pitch: {result.std_pitch:>8.3f}°")
    logger.info(f"  Roll:  {result.std_roll:>8.3f}°")
    
    logger.info("\nConfidence Metrics:")
    logger.info(f"  Mean Angular Distance: {result.mean_angular_distance:.3f}°")
    logger.info(f"  High Confidence Matches: {result.num_high_confidence_matches}")
    
    # Check if offsets exceed threshold
    threshold = 0.5  # degrees
    max_offset = max(
        abs(result.median_yaw),
        abs(result.median_pitch),
        abs(result.median_roll)
    )
    
    if max_offset > threshold:
        logger.warning(f"⚠️  Offset exceeds threshold ({max_offset:.3f}° > {threshold}°)")
    else:
        logger.info(f"✓ Offset within threshold ({max_offset:.3f}° <= {threshold}°)")
    
    # Convert to dict for JSON serialization
    result_dict = result.to_dict()
    logger.info(f"\nResult dictionary keys: {list(result_dict.keys())}")


if __name__ == "__main__":
    main()


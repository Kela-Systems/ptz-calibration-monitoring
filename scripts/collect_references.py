#!/usr/bin/env python3
"""
Script for one-time collection of reference scans from all cameras.

This script uses the ReferenceCollector module to iterate through all devices
and cameras defined in devices.yaml, capturing reference scans and uploading
them to S3 for use in calibration monitoring.

Usage:
    python scripts/collect_references.py [--devices path/to/devices.yaml] [--config]

Examples:
    # Use default devices.yaml in project root
    python scripts/collect_references.py
    
    # Use custom devices.yaml
    python scripts/collect_references.py --devices /path/to/devices.yaml
    
    # Use custom grid size
    python scripts/collect_references.py --horizontal-stops 16 --vertical-stops 6
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path to import monitoring module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitoring import ReferenceCollector


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('reference_collection.log')
    ]
)
logger = logging.getLogger("collect-references")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect reference scans from all PTZ cameras for calibration monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Path to devices.yaml configuration file (default: ./devices.yaml)"
    )
    
    parser.add_argument(
        "--secrets",
        type=str,
        default=None,
        help="Path to secrets.json file (default: ./secrets.json)"
    )
    
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default="camera-calibration-monitoring",
        help="S3 bucket name for storing reference data (default: camera-calibration-monitoring)"
    )
    
    parser.add_argument(
        "--aws-region",
        type=str,
        default="us-east-1",
        help="AWS region for S3 (default: us-east-1)"
    )
    
    parser.add_argument(
        "--horizontal-stops",
        type=int,
        default=12,
        help="Number of horizontal (yaw) stops in capture grid (default: 12)"
    )
    
    parser.add_argument(
        "--vertical-stops",
        type=int,
        default=5,
        help="Number of vertical (pitch) stops in capture grid (default: 5)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Only collect references for specified device/deployment (default: all devices)"
    )
    
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Only collect reference for specified camera within --device (requires --device)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for reference collection script."""
    args = parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Determine devices.yaml path
    if args.devices:
        devices_path = Path(args.devices)
    else:
        # Look for devices.yaml in project root
        project_root = Path(__file__).resolve().parent.parent
        devices_path = project_root / "devices.yaml"
    
    if not devices_path.exists():
        logger.error(f"Devices configuration file not found: {devices_path}")
        logger.error("Please create devices.yaml or specify --devices path")
        sys.exit(1)
    
    logger.info(f"Using devices configuration: {devices_path}")
    
    try:
        # Initialize reference collector
        collector = ReferenceCollector(
            devices_yaml_path=str(devices_path),
            secrets_path=args.secrets,
            s3_bucket=args.s3_bucket,
            aws_region=args.aws_region
        )
        
        # Check if single device/camera mode
        if args.camera and not args.device:
            logger.error("--camera requires --device to be specified")
            sys.exit(1)
        
        if args.device and args.camera:
            # Single camera mode
            logger.info(f"Collecting reference for single camera: {args.device}/{args.camera}")
            success = collector.collect_reference_for_device(
                device_name=args.device,
                camera_name=args.camera,
                horizontal_stops=args.horizontal_stops,
                vertical_stops=args.vertical_stops
            )
            
            if success:
                logger.info("✓ Reference collection completed successfully")
                sys.exit(0)
            else:
                logger.error("✗ Reference collection failed")
                sys.exit(1)
        
        elif args.device:
            # Single device mode (all cameras in device)
            logger.info(f"Collecting references for device: {args.device}")
            
            # Filter to single device
            device_config = None
            for device in collector.devices:
                if device.get("name") == args.device:
                    device_config = device
                    break
            
            if not device_config:
                logger.error(f"Device not found in configuration: {args.device}")
                sys.exit(1)
            
            cameras = device_config.get("cameras", [])
            if not cameras:
                logger.error(f"No cameras found for device: {args.device}")
                sys.exit(1)
            
            results = {}
            for camera in cameras:
                camera_name = camera.get("name")
                if not camera_name:
                    continue
                
                success = collector.collect_reference_for_device(
                    device_name=args.device,
                    camera_name=camera_name,
                    horizontal_stops=args.horizontal_stops,
                    vertical_stops=args.vertical_stops
                )
                results[camera_name] = success
            
            # Check overall success
            all_success = all(results.values())
            successful_count = sum(1 for s in results.values() if s)
            total_count = len(results)
            
            logger.info(f"Completed: {successful_count}/{total_count} cameras successful")
            
            if all_success:
                sys.exit(0)
            else:
                sys.exit(1)
        
        else:
            # All devices mode
            logger.info("Collecting references for all devices and cameras")
            results = collector.collect_all_references(
                horizontal_stops=args.horizontal_stops,
                vertical_stops=args.vertical_stops
            )
            
            # Check overall success
            total_cameras = sum(len(camera_results) for camera_results in results.values())
            successful_cameras = sum(
                sum(1 for success in camera_results.values() if success)
                for camera_results in results.values()
            )
            
            if successful_cameras == total_cameras:
                logger.info("✓ All reference collections completed successfully")
                sys.exit(0)
            else:
                logger.warning(f"⚠ Some reference collections failed: {successful_cameras}/{total_cameras} successful")
                sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("\nReference collection interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Reference collection failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


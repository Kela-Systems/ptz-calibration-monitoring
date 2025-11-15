"""
Example usage of the AWS Integration module for PTZ camera calibration monitoring.

This script demonstrates how to:
1. Initialize the AWS integration
2. Upload reference and query scans
3. Write calibration results to Athena
4. Query calibration results
"""

from datetime import datetime
from aws_integration import (
    AWSIntegration,
    upload_reference_scan,
    upload_query_scan
)

def main():
    # Initialize AWS integration (uses default credentials from environment/config)
    aws = AWSIntegration(region_name="us-east-1")
    
    # Example 1: Create the Athena table (only needs to be done once)
    print("Creating Athena table...")
    aws.create_table()
    
    # Example 2: Upload a reference scan
    print("\nUploading reference scan...")
    image_uris, feature_uris = upload_reference_scan(
        aws,
        deployment_name="test-deployment",
        camera_name="camera-01",
        images_dir="/path/to/reference/images",
        features_dir="/path/to/reference/features"
    )
    print(f"Uploaded {len(image_uris)} images and {len(feature_uris)} features")
    
    # Example 3: Upload a query scan
    print("\nUploading query scan...")
    timestamp_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    query_image_uris, query_feature_uris = upload_query_scan(
        aws,
        deployment_name="test-deployment",
        camera_name="camera-01",
        timestamp=timestamp_str,
        images_dir="/path/to/query/images",
        features_dir="/path/to/query/features"
    )
    print(f"Uploaded {len(query_image_uris)} images and {len(query_feature_uris)} features")
    
    # Example 4: Write a calibration result to Athena
    print("\nWriting calibration result...")
    capture_positions = [
        {"pan": 0.0, "tilt": 0.0, "zoom": 1.0},
        {"pan": 90.0, "tilt": 0.0, "zoom": 1.0},
        {"pan": 180.0, "tilt": 0.0, "zoom": 1.0},
    ]
    
    files_location = aws.get_s3_path(
        "test-deployment",
        "camera-01",
        "query_scan",
        "images",
        timestamp=timestamp_str
    )
    
    query_id = aws.write_calibration_result(
        deployment_name="test-deployment",
        device_id="camera-01",
        timestamp=datetime.now(),
        pitch_offset=0.5,
        yaw_offset=-0.3,
        roll_offset=0.1,
        mode="passive",
        capture_positions=capture_positions,
        files_location=files_location,
        success=True,
        failure_log=""
    )
    print(f"Calibration result written with query ID: {query_id}")
    
    # Example 5: Query calibration results
    print("\nQuerying calibration results...")
    results = aws.query_calibration_results(
        deployment_name="test-deployment",
        device_id="camera-01",
        success_only=True,
        limit=10
    )
    print(f"Found {len(results)} calibration results")
    for result in results:
        print(f"  - {result.get('timestamp')}: "
              f"pitch={result.get('pitch_offset')}, "
              f"yaw={result.get('yaw_offset')}, "
              f"roll={result.get('roll_offset')}")
    
    # Example 6: Get latest calibration for a device
    print("\nGetting latest calibration...")
    latest = aws.get_latest_calibration(
        deployment_name="test-deployment",
        device_id="camera-01"
    )
    if latest:
        print(f"Latest calibration: {latest.get('timestamp')}")
        print(f"  Success: {latest.get('success')}")
        print(f"  Offsets: pitch={latest.get('pitch_offset')}, "
              f"yaw={latest.get('yaw_offset')}, roll={latest.get('roll_offset')}")
    else:
        print("No calibration results found")


if __name__ == "__main__":
    main()


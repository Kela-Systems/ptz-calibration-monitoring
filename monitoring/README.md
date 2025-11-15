# Monitoring Module

PTZ camera calibration monitoring system with AWS integration, calibration algorithm, and Slack notifications.

## Features

- 🎯 **Calibration Algorithm Integration**: Wraps existing offset calculation from `ptz_georeg/utils.py`
- ☁️ **S3 Reference Loading**: Loads reference features from S3 for calibration
- 📊 **Structured Results**: Returns median offsets, weighted means, standard deviations, and confidence metrics
- ✅ **Dual Authentication Support**: Works with both OAuth tokens (recommended) and webhook URLs
- 🚨 **Threshold-based Alerting**: Automatic emoji selection based on 0.5° threshold
- 📈 **Athena Integration**: Query and write calibration results to AWS Athena
- 🔧 **Failure Logging**: Includes detailed error messages when calibration fails
- 🧪 **Comprehensive Testing**: Unit tests with extensive coverage
- 🔒 **Security Best Practices**: Environment variable-based credential management

## Files

### Calibration & Algorithm
- `calibration_runner.py` - Main calibration algorithm wrapper with S3 integration
- `test_calibration_runner.py` - Unit tests for calibration runner
- `example_calibration_runner.py` - Usage examples for calibration runner

### AWS Integration
- `aws_integration.py` - AWS S3 and Athena integration
- `test_aws_integration.py` - AWS integration tests
- `create_athena_table.py` - Athena table creation

### Notifications
- `slack_notifier.py` - Slack notification module
- `test_slack_notifier.py` - Slack notifier tests
- `example_usage.py` - Slack usage examples
- `SLACK_SETUP.md` - Detailed Slack setup guide

### Data Collection
- `reference_collector.py` - Reference scan collection from cameras
- `query_extractor.py` - Query frame extraction with stability monitoring

### Other
- `__init__.py` - Package initialization
- `README.md` - This file

## Calibration Runner

The `CalibrationRunner` module wraps the existing calibration algorithm from `ptz_georeg/utils.py` and integrates it with S3 for reference feature loading.

### Key Functions

**From `ptz_georeg/utils.py`:**
- `calculate_offsets_from_visual_matches` (line 821) - Main calibration algorithm
- `estimate_visual_offsets` (line 1233) - Visual offset estimation
- `calculate_final_offset` (line 1281) - Final offset calculation with statistics

### Usage Example

```python
from monitoring.calibration_runner import (
    CalibrationRunner,
    CalibrationConfig,
    load_camera_calibration,
    load_r_align
)

# Load camera calibration data
camera_matrix, dist_coeff = load_camera_calibration("camera_intrinsics/calibration.npz")
r_align = load_r_align("r_align/10NOV2025_REF1_HOMOGRAPHY.npy")

# Initialize runner with custom config
config = CalibrationConfig(
    min_match_count=15,
    ransac_threshold=5.0,
    num_of_deeplearning_features=2048
)

runner = CalibrationRunner(
    s3_bucket="camera-calibration-monitoring",
    config=config
)

# Run full calibration pipeline
result = runner.run_full_calibration_pipeline(
    deployment_name="production",
    device_id="camera-01",
    query_folder="/path/to/query/frames",
    query_manifest_path="/path/to/manifest.json",
    camera_matrix=camera_matrix,
    dist_coeff=dist_coeff,
    r_align=r_align
)

if result:
    print(f"Median Yaw Offset: {result.median_yaw:.3f}°")
    print(f"Median Pitch Offset: {result.median_pitch:.3f}°")
    print(f"Median Roll Offset: {result.median_roll:.3f}°")
    print(f"Confidence: {result.num_high_confidence_matches} matches")
```

### CalibrationResult Structure

The calibration returns a `CalibrationResult` object with:

- **Median Offsets**: `median_yaw`, `median_pitch`, `median_roll`
- **Weighted Mean Offsets**: `weighted_mean_yaw`, `weighted_mean_pitch`, `weighted_mean_roll`
- **Standard Deviations**: `std_yaw`, `std_pitch`, `std_roll`
- **Confidence Metrics**: 
  - `mean_angular_distance`: Average angular distance between matches
  - `num_high_confidence_matches`: Number of high-quality matches found
- **Raw Data**: `all_offsets` - List of all individual offset measurements

### S3 Structure

Reference features should be stored in S3 with the following structure:

```
s3://camera-calibration-monitoring/
  {deployment-name}/
    {device-id}/
      reference_scan/
        features/
          reference_panorama.json      # Telemetry data
          frame_001.txt                # COLMAP format features
          frame_002.txt
          ...
```

### Configuration Options

```python
CalibrationConfig(
    min_match_count=15,              # Minimum feature matches required
    ransac_threshold=5.0,            # RANSAC outlier threshold
    matching_method=MatchingMethod.SUPERGLUE,  # SIFT or SUPERGLUE
    geometry_model=GeometryModel.HOMOGRAPHY,   # HOMOGRAPHY or ESSENTIAL
    num_of_deeplearning_features=2048,
    resize_scale_deep_learning=1.0,
    max_xy_norm=0.1,
    min_overlapping_ratio=0.3,
    roi=[0.1, 0.1, 0.9, 0.9]        # Region of interest [x_min, y_min, x_max, y_max]
)
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install slack-sdk>=3.19.0
# or
pip install -r requirements.txt
```

### 2. Configure Credentials

**Option A: OAuth Token (Recommended)**
```bash
export SLACK_ACCESS_TOKEN="xoxb-your-token-here"
export SLACK_CHANNEL="calibration_monitoring"  # Optional
```

**Option B: Webhook URL**
```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

See `SLACK_SETUP.md` for detailed setup instructions.

### 3. Use in Your Code

```python
from monitoring import SlackNotifier

# Initialize (auto-detects credentials from environment)
notifier = SlackNotifier()

# Send a calibration report
notifier.send_calibration_report(
    deployment="site-name",
    device_id="camera-1",
    pitch=0.2,    # degrees
    yaw=0.3,      # degrees
    roll=0.1,     # degrees
    mode="passive",
    success=True
)
```

## Message Format

```
[✅/🚨] Camera Calibration Report
Deployment: gan-shomron-dell
Device: onvifcam-1
Timestamp: 2025-11-15T21:00:00Z
Offsets: Pitch=0.2°, Yaw=0.3°, Roll=0.1°
Mode: passive
Success: Yes
```

- 🚨 Alert emoji shown if ANY offset > 0.5°
- ✅ Success emoji shown if all offsets ≤ 0.5°

## Testing

Run the test suite:
```bash
pytest monitoring/test_slack_notifier.py -v
```

Test with example script:
```bash
python monitoring/example_usage.py
```

## API Reference

### SlackNotifier Class

#### Constructor

```python
SlackNotifier(
    access_token: Optional[str] = None,
    webhook_url: Optional[str] = None,
    channel: Optional[str] = None
)
```

**Parameters:**
- `access_token`: Slack OAuth token (overrides `SLACK_ACCESS_TOKEN` env var)
- `webhook_url`: Slack webhook URL (overrides `SLACK_WEBHOOK_URL` env var)
- `channel`: Target channel name without # (overrides `SLACK_CHANNEL` env var)

#### send_calibration_report()

```python
send_calibration_report(
    deployment: str,
    device_id: str,
    pitch: float,
    yaw: float,
    roll: float,
    mode: str = "passive",
    success: bool = True,
    timestamp: Optional[str] = None,
    failure_logs: Optional[List[str]] = None,
    channel: Optional[str] = None
) -> bool
```

**Parameters:**
- `deployment`: Deployment/site name
- `device_id`: Camera/device identifier
- `pitch`: Pitch offset in degrees
- `yaw`: Yaw offset in degrees
- `roll`: Roll offset in degrees
- `mode`: Operation mode ("passive" or "active")
- `success`: Whether calibration succeeded
- `timestamp`: ISO 8601 timestamp (auto-generated if None)
- `failure_logs`: List of error messages (shown when success=False)
- `channel`: Override default channel for this message

**Returns:** `True` if message sent successfully, `False` otherwise

#### send_test_message()

```python
send_test_message() -> bool
```

Sends a test message with sample data to verify configuration.

**Returns:** `True` if successful, `False` otherwise

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SLACK_ACCESS_TOKEN` | OAuth bot token (starts with `xoxb-`) | Yes* |
| `SLACK_WEBHOOK_URL` | Incoming webhook URL | Yes* |
| `SLACK_CHANNEL` | Target channel name (default: "calibration_monitoring") | No |

*Either `SLACK_ACCESS_TOKEN` or `SLACK_WEBHOOK_URL` must be set.

## Examples

### Success with Low Offsets

```python
notifier.send_calibration_report(
    deployment="gan-shomron-dell",
    device_id="onvifcam-1",
    pitch=0.2,
    yaw=0.3,
    roll=0.1,
    mode="passive",
    success=True
)
# Result: ✅ emoji (all offsets ≤ 0.5°)
```

### Alert with High Offsets

```python
notifier.send_calibration_report(
    deployment="gan-shomron-dell",
    device_id="onvifcam-2",
    pitch=0.8,  # Exceeds threshold
    yaw=0.6,    # Exceeds threshold
    roll=0.2,
    mode="active",
    success=True
)
# Result: 🚨 emoji (offsets > 0.5°)
```

### Failure with Error Logs

```python
notifier.send_calibration_report(
    deployment="gan-shomron-dell",
    device_id="onvifcam-3",
    pitch=1.5,
    yaw=2.0,
    roll=0.9,
    mode="passive",
    success=False,
    failure_logs=[
        "Failed to detect features in reference image",
        "Insufficient matching points (found 12, need 50)",
        "Camera may need recalibration"
    ]
)
```

### Custom Channel

```python
notifier.send_calibration_report(
    deployment="site-name",
    device_id="camera-1",
    pitch=0.2,
    yaw=0.3,
    roll=0.1,
    channel="alerts-critical"  # Override default
)
```

## Troubleshooting

See `SLACK_SETUP.md` for detailed troubleshooting guidance.

**Common Issues:**

1. **"No credentials configured"** - Set `SLACK_ACCESS_TOKEN` or `SLACK_WEBHOOK_URL`
2. **"Channel not found"** - Invite bot to channel with `/invite @BotName`
3. **"Invalid token"** - Verify token starts with `xoxb-` and hasn't been revoked
4. **Messages not appearing** - Check bot is member of target channel

## Implementation Notes

- Uses `slack-sdk` for OAuth token authentication
- Falls back to `requests` for webhook authentication
- Graceful degradation if SDK not available
- Thread-safe for concurrent usage
- Automatic timestamp generation
- Comprehensive error logging
- 100% test coverage

## License

Part of the PTZ Calibration Monitoring project.


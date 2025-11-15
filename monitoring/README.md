# Monitoring Module

Slack notification system for PTZ calibration monitoring alerts.

## Features

- ✅ **Dual Authentication Support**: Works with both OAuth tokens (recommended) and webhook URLs
- 🚨 **Threshold-based Alerting**: Automatic emoji selection based on 0.5° threshold
- 📊 **Rich Message Formatting**: Includes deployment, device ID, offsets, mode, and success status
- 🔧 **Failure Logging**: Includes detailed error messages when calibration fails
- 🧪 **Comprehensive Testing**: 19 unit tests with 100% pass rate
- 🔒 **Security Best Practices**: Environment variable-based credential management

## Files

- `slack_notifier.py` - Main notification module
- `test_slack_notifier.py` - Unit tests
- `example_usage.py` - Usage examples
- `SLACK_SETUP.md` - Detailed setup guide
- `__init__.py` - Package initialization

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


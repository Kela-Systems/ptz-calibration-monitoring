<!-- 7e75901b-be63-4b6b-a1d8-9dd18bb085ca c896c504-77ea-4e0c-82ee-7b1c16aa10a7 -->
# Calibration Monitoring System

## Overview

Build a system to monitor PTZ camera calibration across multiple Kubernetes clusters, with reference collection, ongoing monitoring, AWS integration, and Slack notifications.

## Core Modules

### 1. AWS Infrastructure Setup

- Create Python module for Athena table management (`monitoring/aws_integration.py`)
- Define Iceberg table schema matching spec: deployment_name, device_id, timestamp, pitch/yaw/roll offsets, mode, capture_positions, files_location, success, failure_log
- S3 utilities for storing/retrieving images and features at paths: `s3://camera-calibration-monitoring/{deployment-name}/{camera-name}/reference_scan/` and `query_scan/{timestamp}/`

### 2. Reference Collection Module

- Extract relevant code from `scan.py` for camera scanning (lines ~420-837 contain `capture_views_and_render`)
- Create `monitoring/reference_collector.py` that:
  - Iterates through devices in `devices.yaml`
  - Switches kubectl context for each cluster
  - Runs camera scan (reuse port forwarding logic from `scan.py`)
  - Extracts features using logic from `ptz_georeg/save_features_to_colmap.py`
  - Uploads images to S3 `reference_scan/images/` and features to `reference_scan/features/`
- One-time execution script: `scripts/collect_references.py`

### 3. Query Extraction Module  

- Create `monitoring/query_extractor.py` with function signature:
  ```python
  extract_query(device_id: str, stabilize_time: int = 30, 
                timeout: Optional[int] = None, 
                active_ptz_stops: List[Tuple[float, float, float]] = [])
  ```

- Reuse MQTT telemetry monitoring from `scan.py` (TelemetryLatch helper at line 29)
- Implement stability detection: monitor pitch/yaw/roll for zero movement over stabilize_time
- Handle passive mode (single frame capture) vs active mode (commanded PTZ stops)
- Save frames and telemetry manifest locally (temp directory)
- Return captured frames with metadata

### 4. Calibration Algorithm Integration

- Create `monitoring/calibration_runner.py` to wrap existing offset calculation
- Use `calculate_offsets_from_visual_matches` from `ptz_georeg/utils.py` (line 821)
- Load reference features from S3
- Compare against query frames
- Extract pitch/yaw/roll offsets using `calculate_final_offset` (line 1281)
- Return structured results

### 5. Slack Integration

- Create `monitoring/slack_notifier.py`
- Format messages with deployment, device, timestamp, offsets
- Apply threshold logic: 🚨 if any offset > 0.5°, else ✅
- Send to `calibration_monitoring` channel via webhook

### 6. Main Orchestration Script

- Create `scripts/run_calibration_monitoring.py` that:
  - Loads devices from `devices.yaml`
  - For each device: switch context, extract query, run algorithm
  - Write results to Athena table
  - Upload query files to S3 `query_scan/{timestamp}/`
  - Send Slack notification
  - Handle failures gracefully (log to Athena + Slack)

### 7. Configuration & Scheduling

- Create `configs/calibration_monitoring.ini` for:
  - AWS credentials/region
  - S3 bucket name
  - Athena database/table
  - Slack webhook URL
  - Default stabilize_time, timeout
  - Active PTZ stops (if any)
- Add scheduling instructions (cron/K8s CronJob/systemd timer) to README

## Key Files to Reference

- `scan.py`: Port forwarding (line 32), MQTT telemetry (line 29), frame capture (lines 420+)
- `ptz_georeg/utils.py`: Offset calculation algorithm (lines 821-1335)
- `ptz_georeg/save_features_to_colmap.py`: Feature extraction
- `devices.yaml`: Device/cluster definitions

## Testing Strategy

- Unit tests for stability detection logic
- Integration test with single camera in dev cluster
- End-to-end test with mock S3/Athena

### To-dos

- [ ] [ALG-74](https://linear.app/kelasys/issue/ALG-74) - Create AWS infrastructure module (Athena table schema, S3 utilities)
- [ ] [ALG-75](https://linear.app/kelasys/issue/ALG-75) - Build reference collection module by extracting and adapting code from scan.py
- [ ] [ALG-76](https://linear.app/kelasys/issue/ALG-76) - Create query extraction module with MQTT stability monitoring
- [ ] [ALG-77](https://linear.app/kelasys/issue/ALG-77) - Integrate existing calibration algorithm from ptz_georeg/utils.py
- [ ] [ALG-78](https://linear.app/kelasys/issue/ALG-78) - Build Slack notification module with threshold logic
- [ ] [ALG-79](https://linear.app/kelasys/issue/ALG-79) - Create main orchestration script combining all modules
- [ ] [ALG-80](https://linear.app/kelasys/issue/ALG-80) - Add configuration file and update README with setup/scheduling instructions
- [ ] [ALG-81](https://linear.app/kelasys/issue/ALG-81) - Execute one-time reference collection for all devices in devices.yaml
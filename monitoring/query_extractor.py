"""
Query Extraction Module - MQTT Stability Monitoring

This module provides functionality to capture camera frames with telemetry
when the camera is stable (not moving) or at commanded PTZ positions.
"""

import json
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.mqtt_helper import TelemetryLatch
from helpers.camera_control import ONVIFCameraControl, build_command_endpoint, wait_for_services
from helpers.rtsp_helper import RTSPCapture
from port_forward_utils import start_port_forwards, stop_port_forwards

logger = logging.getLogger(__name__)


def load_secrets(secrets_path: Optional[Path] = None) -> Dict:
    """
    Load secrets from secrets.json file.
    
    Args:
        secrets_path: Optional path to secrets file
    
    Returns:
        Dictionary containing secrets configuration
    """
    if secrets_path is None:
        secrets_path = Path(__file__).resolve().parent.parent / "secrets.json"
    
    if not secrets_path.exists():
        logger.warning(f"Secrets file not found: {secrets_path}")
        return {}
    
    try:
        with open(secrets_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load secrets from {secrets_path}: {e}")
        return {}


def build_camera_telemetry_topic(camera_name: str, variant: str = "main") -> str:
    """
    Build MQTT telemetry topic for a camera.
    
    Args:
        camera_name: Name of the camera device
        variant: Camera variant (main or thermal)
    
    Returns:
        MQTT topic string
    """
    return f"/device/onvifcam/{camera_name}/payloads/cameras/{variant}/telemetry"


def build_rtsp_url(camera_name: str, variant: str = "main", base_url: str = "rtsp://localhost:8554") -> str:
    """
    Build RTSP stream URL for a camera.
    
    Args:
        camera_name: Name of the camera device
        variant: Camera variant (main or thermal)
        base_url: Base RTSP URL
    
    Returns:
        Full RTSP URL
    """
    stream_name = f"{camera_name}-{variant}"
    if not base_url.endswith("/"):
        base_url = base_url + "/"
    return f"{base_url}{stream_name}"


def extract_query(
    device_id: str,
    stabilize_time: int = 30,
    timeout: Optional[int] = None,
    active_ptz_stops: List[Tuple[float, float, float]] = None
) -> Dict:
    """
    Extract query frames and telemetry from a camera.
    
    This function monitors camera telemetry via MQTT and captures frames when
    the camera is stable (passive mode) or at commanded PTZ positions (active mode).
    
    Args:
        device_id: Camera device identifier (e.g., "onvifcam-dev-1")
        stabilize_time: Seconds of stability required before capture (default: 30)
        timeout: Optional timeout in seconds for waiting for stability (None = no timeout)
        active_ptz_stops: Optional list of (pan, tilt, zoom) tuples for active mode.
                         If provided, camera will be commanded to each position.
                         If empty or None, runs in passive mode (wait for stability).
    
    Returns:
        Dictionary containing:
            - 'camera_frames': List of captured PIL Images
            - 'telemetry': Dictionary mapping frame index to telemetry data
            - 'capture_positions': List of (pan, tilt, zoom) positions where frames were captured
            - 'temp_dir': Path to temporary directory containing saved frames
            - 'manifest_path': Path to telemetry manifest JSON file
    
    Raises:
        TimeoutError: If timeout is specified and no stable period is found
        RuntimeError: If required services are not available
    """
    if active_ptz_stops is None:
        active_ptz_stops = []
    
    # Determine mode
    is_passive_mode = len(active_ptz_stops) == 0
    mode = "passive" if is_passive_mode else "active"
    
    logger.info(f"Starting query extraction for device '{device_id}' in {mode} mode")
    logger.info(f"  Stabilize time: {stabilize_time}s")
    logger.info(f"  Timeout: {timeout}s" if timeout else "  Timeout: None (will wait indefinitely)")
    
    if not is_passive_mode:
        logger.info(f"  Active PTZ stops: {len(active_ptz_stops)} positions")
    
    # Load secrets
    secrets = load_secrets()
    if not secrets:
        logger.warning("No secrets.json found, using default values")
    
    # Start port forwards
    required_ports = [1883, 8800, 8554, 8000]  # MQTT, HTTP, RTSP, ONVIF
    start_port_forwards(required_ports, onvif_device_name=device_id)
    
    try:
        # Wait for services to be available
        logger.info("Waiting for services to become available...")
        if not wait_for_services(secrets, timeout=30.0):
            raise RuntimeError("Required services are not available")
        
        # Get configuration from secrets
        mqtt_config = secrets.get("mqtt", {})
        mqtt_host = mqtt_config.get("host", "localhost")
        mqtt_port = mqtt_config.get("port", 1883)
        mqtt_user = mqtt_config.get("username", "admin")
        mqtt_pass = mqtt_config.get("password", "Kelasys123!")
        
        rtsp_base = secrets.get("rtsp", {}).get("base_url", "rtsp://localhost:8554")
        
        # Setup MQTT telemetry latch
        variant = "main"  # Default to main camera variant
        topic = build_camera_telemetry_topic(device_id, variant)
        latch = TelemetryLatch(mqtt_host, mqtt_port, mqtt_user, mqtt_pass, topic)
        
        # Create temporary directory for storing frames
        temp_dir = tempfile.mkdtemp(prefix=f"query_extract_{device_id}_")
        logger.info(f"Created temporary directory: {temp_dir}")
        
        # Initialize results
        camera_frames: List[Image.Image] = []
        telemetry_dict: Dict[int, Dict] = {}
        capture_positions: List[Tuple[float, float, float]] = []
        
        try:
            if is_passive_mode:
                # PASSIVE MODE: Wait for camera to be stable, then capture single frame
                logger.info(f"Passive mode: Waiting for camera to be stable for {stabilize_time}s...")
                
                start_time = time.time()
                stable = False
                
                while True:
                    # Check timeout
                    if timeout is not None and (time.time() - start_time) > timeout:
                        raise TimeoutError(
                            f"Timeout ({timeout}s) reached - no stable period of {stabilize_time}s found"
                        )
                    
                    # Check if camera is stable
                    if latch.is_stable(duration=stabilize_time, tolerance=0.1):
                        logger.info(f"Camera stable for {stabilize_time}s - capturing frame")
                        stable = True
                        break
                    
                    # Log progress every 10 seconds
                    elapsed = time.time() - start_time
                    if int(elapsed) % 10 == 0:
                        logger.info(f"Still waiting for stability... (elapsed: {int(elapsed)}s)")
                    
                    time.sleep(1.0)
                
                if not stable:
                    raise RuntimeError("Failed to detect stable period")
                
                # Get current telemetry
                telem = latch.get_latest()
                if telem is None:
                    raise RuntimeError("No telemetry available after stability detected")
                
                # Capture frame via RTSP
                rtsp_url = build_rtsp_url(device_id, variant, rtsp_base)
                rtsp_capture = RTSPCapture()
                rtsp_capture.connect(rtsp_url, use_tcp=True, low_latency=True)
                
                try:
                    # Capture frame
                    frame = rtsp_capture.capture_frame(timeout=10.0, convert_to_pil=True)
                    
                    # Save frame
                    frame_filename = f"frame_0000_passive.png"
                    frame_path = os.path.join(temp_dir, frame_filename)
                    frame.save(frame_path)
                    logger.info(f"Saved frame: {frame_filename}")
                    
                    # Store results
                    camera_frames.append(frame)
                    capture_positions.append((telem.yaw, telem.pitch, telem.zoom_factor))
                    
                    telemetry_dict[0] = {
                        "attitude": {
                            "yaw": telem.yaw,
                            "pitch": telem.pitch,
                            "roll": telem.roll
                        },
                        "field_of_view_h": telem.fov_h,
                        "field_of_view_v": telem.fov_v,
                        "zoom_factor": telem.zoom_factor,
                        "timestamp": datetime.now().isoformat(),
                        "mode": "passive"
                    }
                    
                finally:
                    rtsp_capture.close()
            
            else:
                # ACTIVE MODE: Command camera to each PTZ stop and capture
                logger.info(f"Active mode: Commanding camera to {len(active_ptz_stops)} positions")
                
                # Initialize camera controller and RTSP
                cmd_url = build_command_endpoint(device_id)
                controller = ONVIFCameraControl(device_id, command_url=cmd_url)
                
                rtsp_url = build_rtsp_url(device_id, variant, rtsp_base)
                rtsp_capture = RTSPCapture()
                rtsp_capture.connect(rtsp_url, use_tcp=True, low_latency=True)
                
                try:
                    for idx, (pan, tilt, zoom) in enumerate(active_ptz_stops):
                        logger.info(f"Position {idx + 1}/{len(active_ptz_stops)}: pan={pan:.1f}, tilt={tilt:.1f}, zoom={zoom:.3f}")
                        
                        # Clear stale telemetry
                        latch.get_once(timeout=0.1)
                        
                        # Move camera to position
                        controller.absolute_move(pan, tilt, zoom=zoom, timeout=10.0, retries=3)
                        logger.info("Camera move command sent, waiting for telemetry convergence...")
                        
                        # Wait for telemetry to converge
                        start_wait = time.time()
                        telemetry_timeout = 15.0
                        telem = None
                        
                        while time.time() - start_wait < telemetry_timeout:
                            fresh = latch.get_once(timeout=1.0)
                            if fresh is None:
                                time.sleep(0.1)
                                continue
                            
                            # Check convergence (tolerance: 1.0 degree for pan/tilt, 0.05 for zoom)
                            pan_diff = abs(fresh.yaw - pan)
                            if pan_diff > 180:
                                pan_diff = 360 - pan_diff
                            
                            tilt_diff = abs(fresh.pitch - tilt)
                            
                            # Extract zoom from raw_data (0-1 range)
                            zoom_value = fresh.raw_data.get("zoom") or fresh.raw_data.get("data", {}).get("zoom", 0.0)
                            zoom_diff = abs(float(zoom_value) - zoom)
                            
                            if pan_diff <= 1.0 and tilt_diff <= 1.0 and zoom_diff <= 0.05:
                                telem = fresh
                                break
                        
                        if telem is None:
                            logger.warning(f"Telemetry did not converge for position {idx}, skipping")
                            continue
                        
                        logger.info("Telemetry converged, settling before capture...")
                        time.sleep(1.5)  # Allow camera to settle and focus
                        
                        # Capture frame
                        frame = rtsp_capture.capture_frame(timeout=10.0, convert_to_pil=True)
                        
                        # Save frame
                        frame_filename = f"frame_{idx:04d}_pan_{pan:.1f}_tilt_{tilt:.1f}_zoom_{zoom:.3f}.png"
                        frame_path = os.path.join(temp_dir, frame_filename)
                        frame.save(frame_path)
                        logger.info(f"Saved frame: {frame_filename}")
                        
                        # Store results
                        camera_frames.append(frame)
                        capture_positions.append((telem.yaw, telem.pitch, telem.zoom_factor))
                        
                        telemetry_dict[idx] = {
                            "attitude": {
                                "yaw": telem.yaw,
                                "pitch": telem.pitch,
                                "roll": telem.roll
                            },
                            "field_of_view_h": telem.fov_h,
                            "field_of_view_v": telem.fov_v,
                            "zoom_factor": telem.zoom_factor,
                            "commanded_pan": pan,
                            "commanded_tilt": tilt,
                            "commanded_zoom": zoom,
                            "timestamp": datetime.now().isoformat(),
                            "mode": "active"
                        }
                
                finally:
                    rtsp_capture.close()
        
        finally:
            # Clean up MQTT latch
            latch.stop()
        
        # Save telemetry manifest
        manifest_path = os.path.join(temp_dir, "telemetry_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump({
                "device_id": device_id,
                "mode": mode,
                "stabilize_time": stabilize_time,
                "capture_count": len(camera_frames),
                "telemetry": telemetry_dict,
                "capture_positions": [
                    {"pan": pos[0], "tilt": pos[1], "zoom": pos[2]}
                    for pos in capture_positions
                ],
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Saved telemetry manifest: {manifest_path}")
        logger.info(f"Query extraction complete - captured {len(camera_frames)} frame(s)")
        
        return {
            'camera_frames': camera_frames,
            'telemetry': telemetry_dict,
            'capture_positions': capture_positions,
            'temp_dir': temp_dir,
            'manifest_path': manifest_path
        }
    
    finally:
        # Clean up port forwards
        stop_port_forwards()


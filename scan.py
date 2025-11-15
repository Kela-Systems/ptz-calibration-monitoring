#!/usr/bin/env python3
import argparse
import atexit
import json
import logging
import os
import signal
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import io
import requests
from PIL import Image

# Add current directory to path to import helpers
sys.path.insert(0, str(Path(__file__).parent))
from helpers.camera_control import (
    build_command_endpoint,
    check_stream_available,
    wait_for_services,
    ONVIFCameraControl,
)
from helpers.rtsp_helper import RTSPCapture
from helpers.stream_discovery import discover_streams
from helpers.mqtt_helper import TelemetryLatch
from helpers.offset_simulation import CameraBaseRotation, clamp, wrap_angle_deg
from helpers.ptz_ocr import extract_ptz_with_vision_model
from port_forward_utils import start_port_forwards, stop_port_forwards


"""
Grid-based frame capture for ONVIF PTZ cameras using media-server RTSP feed
and MQTT telemetry for precise orientation (yaw/pitch) and FOV per capture.

This script captures individual frames from a PTZ camera at grid positions
and saves them with synchronized telemetry metadata. Frames can be used to
generate panoramas using a separate rendering tool if needed.
"""


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("onvif-equirect-capture")


def load_secrets(secrets_path: Optional[Path] = None) -> Dict:
    """
    Load secrets from secrets.json file.
    Looks for secrets.json in the project root directory (same directory as this script).
    Returns empty dict if file doesn't exist.
    """
    if secrets_path is None:
        secrets_path = Path(__file__).resolve().parent / "secrets.json"
    
    if not secrets_path.exists():
        return {}
    
    try:
        with open(secrets_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load secrets from {secrets_path}: {e}")
        return {}

# Camera elevation limits (degrees) - this is the limit of how much the camera can tilt up and down - though we may limit it further when capturing the views
PITCH_MIN_HARDWARE = -88.0
PITCH_MAX_HARDWARE = 38.0


def build_media_server_endpoints(base_http: str, camera_name: str, variant: str):
    stream_name = f"{camera_name}-{variant}"
    return {
        "streams": f"{base_http}/streams",
        "capture": f"{base_http}/streams/{stream_name}/capture",
        "stream_name": stream_name,
    }


def build_onvif_command_endpoint(camera_name: str) -> str:
    """Backward compatibility wrapper for build_command_endpoint."""
    return build_command_endpoint(camera_name)


def build_camera_telemetry_topic(camera_name: str, variant: str) -> str:
    return f"/device/onvifcam/{camera_name}/payloads/cameras/{variant}/telemetry"


def http_absolute_move(command_url: str, az: float, el: float, zoom: Optional[float] = None, timeout: float = 5.0, retries: int = 3) -> None:
    """
    Backward compatibility wrapper for ONVIFCameraControl.absolute_move.
    
    This function maintains the original API signature while using the new helper class.
    """
    # Extract camera name from command URL for ONVIFCameraControl initialization
    # Format: http://localhost:8000/device/onvifcam/{camera_name}/command
    camera_name = command_url.split("/device/onvifcam/")[1].split("/command")[0]
    controller = ONVIFCameraControl(camera_name, command_url=command_url)
    controller.absolute_move(az, el, zoom=zoom, timeout=timeout, retries=retries)

# RTSP functions moved to helpers.rtsp_helper

def capture_single_frame_http(capture_url: str, timeout: float = 8.0) -> Image.Image:
    r = requests.get(capture_url, timeout=timeout)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content))


# Image rotation and MQTT telemetry functions moved to helpers


# check_stream_available and wait_for_services are imported from helpers.camera_control
# wrap_angle_deg is imported from helpers.offset_simulation


def crop_telemetry_region(img: Image.Image, top_pct: float, left_pct: float) -> Image.Image:
    """
    Crop the telemetry region from the top-left corner of an image.
    
    Args:
        img: PIL Image to crop
        top_pct: Percentage of image height to include from top (e.g., 0.1125 for 11.25%)
        left_pct: Percentage of image width to include from left (e.g., 0.35 for 35%)
    
    Returns:
        Cropped PIL Image containing the telemetry region
    """
    width, height = img.size
    crop_top = int(height * top_pct)
    crop_left = int(width * left_pct)
    # Crop box: (left, upper, right, lower)
    return img.crop((0, 0, crop_left, crop_top))


def validate_ocr_telemetry(
    img: Image.Image,
    expected_pan: float,
    expected_tilt: float,
    telemetry_zoom_factor: float,
    ocr_crop_top_pct: float,
    ocr_crop_left_pct: float,
    ocr_tolerance: float,
    negate_tilt: bool,
    logger: logging.Logger
) -> bool:
    """
    Validate PTZ telemetry using vision model OCR on the captured frame.
    
    Args:
        img: PIL Image containing the frame with telemetry overlay
        expected_pan: Expected pan value (degrees)
        expected_tilt: Expected tilt value (degrees)
        telemetry_zoom_factor: Zoom factor from telemetry. Camera feed displays zoom factor, but we command in zoom 0-1, not zoom factor - but we can assume that since we are checking that the telemetry has converged, the zoom factor from the telemetry is correct.
        ocr_crop_top_pct: Percentage of image height to crop from top for OCR region
        ocr_crop_left_pct: Percentage of image width to crop from left for OCR region
        ocr_tolerance: Tolerance in degrees for pan/tilt comparison
        negate_tilt: If True, negate the OCR-extracted tilt value before comparison
        logger: Logger instance for output
    
    Returns:
        True if OCR values match expected values within tolerance, False otherwise
    """
    try:
        # Crop telemetry region
        telemetry_region = crop_telemetry_region(img, ocr_crop_top_pct, ocr_crop_left_pct)
        
        # Use zoom factor directly from telemetry
        expected_zoom_factor = telemetry_zoom_factor
        # Adjust expected tilt if negate_tilt is enabled
        expected_tilt_for_vision = -expected_tilt if negate_tilt else expected_tilt
        
        # Extract PTZ values using vision model
        vision_result = extract_ptz_with_vision_model(
            telemetry_region,
            expected_pan=expected_pan,
            expected_tilt=expected_tilt_for_vision,
            expected_zoom=expected_zoom_factor,
            verbose=False
        )
        
        # Check if model said match
        if vision_result.get("match", False):
            logger.info("OCR validation (vision model): Match confirmed - Pan=%.1f°, Tilt=%.1f°, Zoom=%.1fX",
                       vision_result.get("pan", expected_pan),
                       vision_result.get("tilt", expected_tilt_for_vision),
                       vision_result.get("zoom", expected_zoom_factor))
            return True
        
        # Fallback: manually verify if angles match with wrap-around handling
        # Sometimes the model might miss the wrap-around equivalence
        if "pan" in vision_result and "tilt" in vision_result and "zoom" in vision_result:
            extracted_pan = vision_result["pan"]
            extracted_tilt = vision_result["tilt"]
            extracted_zoom = vision_result["zoom"]
            
            # Calculate angular difference with wrap-around for pan
            pan_diff = abs(wrap_angle_deg(extracted_pan - expected_pan))
            tilt_diff = abs(extracted_tilt - expected_tilt_for_vision)
            zoom_diff = abs(extracted_zoom - expected_zoom_factor)
            
            # Check if values actually match within tolerance
            if pan_diff <= ocr_tolerance and tilt_diff <= ocr_tolerance and zoom_diff <= 0.5:
                logger.info("OCR validation (vision model - manual verification): Match confirmed after wrap-around check - Pan=%.1f° (diff=%.2f°), Tilt=%.1f° (diff=%.2f°), Zoom=%.1fX (diff=%.2f)",
                           extracted_pan, pan_diff, extracted_tilt, tilt_diff, extracted_zoom, zoom_diff)
                return True
        
        # No match found
        reason = vision_result.get("reason", "Unknown reason")
        logger.warning("OCR validation (vision model): Mismatch - %s", reason)
        if "pan" in vision_result:
            logger.warning("  Extracted: Pan=%.1f°, Tilt=%.1f°, Zoom=%.1fX",
                         vision_result.get("pan", 0),
                         vision_result.get("tilt", 0),
                         vision_result.get("zoom", 0))
        return False
            
    except Exception as e:
        logger.warning("OCR validation failed with error: %s", e)
        return False


def run_parallel_ocr_validation(
    img: Image.Image,
    frame_path: str,
    frame_filename: str,
    expected_pan: float,
    expected_tilt: float,
    telemetry_zoom_factor: float,
    ocr_crop_top_pct: float,
    ocr_crop_left_pct: float,
    ocr_tolerance: float,
    negate_tilt: bool,
    manifest: Dict[str, Dict],
    manifest_lock: threading.Lock,
    logger: logging.Logger
) -> None:
    """
    Run OCR validation in a background thread. If validation fails, delete the image
    and remove the entry from the manifest.
    
    Args:
        img: PIL Image containing the frame with telemetry overlay
        frame_path: Full path to the saved image file
        frame_filename: Filename of the frame (key in manifest)
        expected_pan: Expected pan value (degrees)
        expected_tilt: Expected tilt value (degrees)
        telemetry_zoom_factor: Zoom factor from telemetry
        ocr_crop_top_pct: Percentage of image height to crop from top for OCR region
        ocr_crop_left_pct: Percentage of image width to crop from left for OCR region
        ocr_tolerance: Tolerance in degrees for pan/tilt comparison
        negate_tilt: If True, negate the OCR-extracted tilt value before comparison
        manifest: Manifest dictionary (shared, thread-safe access required)
        manifest_lock: Lock for thread-safe manifest access
        logger: Logger instance for output
    """
    try:
        logger.info("Starting parallel OCR validation for %s", frame_filename)
        ocr_validated = validate_ocr_telemetry(
            img, expected_pan, expected_tilt, telemetry_zoom_factor,
            ocr_crop_top_pct, ocr_crop_left_pct, ocr_tolerance, negate_tilt, logger
        )
        
        if not ocr_validated:
            logger.warning("Parallel OCR validation failed for %s - deleting image and removing from manifest", frame_filename)
            # Delete the image file
            try:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
                    logger.info("Deleted image file: %s", frame_path)
            except Exception as e:
                logger.error("Failed to delete image file %s: %s", frame_path, e)
            
            # Remove from manifest (thread-safe)
            with manifest_lock:
                if frame_filename in manifest:
                    del manifest[frame_filename]
                    logger.info("Removed %s from manifest", frame_filename)
        else:
            logger.info("Parallel OCR validation passed for %s", frame_filename)
    except Exception as e:
        logger.error("Error in parallel OCR validation for %s: %s", frame_filename, e)
        # On error, delete the image and remove from manifest to be safe
        try:
            if os.path.exists(frame_path):
                os.remove(frame_path)
                logger.info("Deleted image file due to OCR validation error: %s", frame_path)
        except Exception:
            pass
        with manifest_lock:
            if frame_filename in manifest:
                del manifest[frame_filename]
                logger.info("Removed %s from manifest due to OCR validation error", frame_filename)
            

def extract_zoom_from_telemetry(telemetry_sample) -> Optional[float]:
    """
    Extract zoom value (0-1 range) from telemetry sample's raw_data.
    
    Handles both flat structure (raw MQTT message) and nested structure (recording format).
    
    Args:
        telemetry_sample: TelemetrySample object with raw_data attribute
    
    Returns:
        Zoom value (0-1 range) or None if not found
    """
    if not telemetry_sample or not telemetry_sample.raw_data:
        return None
    
    # Try direct access first (flat structure - raw MQTT message)
    zoom = telemetry_sample.raw_data.get("zoom")
    if zoom is not None:
        return float(zoom)
    
    # If not found, try nested data wrapper (recording format)
    if "data" in telemetry_sample.raw_data:
        zoom = telemetry_sample.raw_data["data"].get("zoom")
        if zoom is not None:
            return float(zoom)
    
    return None


def parse_custom_stops(custom_stops_arg: str) -> List[Tuple[float, float, float]]:
    """Parse custom stops from JSON file or command-line string.
    
    Args:
        custom_stops_arg: Either a path to a JSON file or a JSON string
    
    Returns:
        List of (pan, tilt, zoom) tuples in degrees/factor
    
    Raises:
        ValueError: If the format is invalid
    """
    # Try to parse as file path first
    if os.path.exists(custom_stops_arg):
        try:
            with open(custom_stops_arg, 'r') as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse custom stops file: {e}")
    else:
        # Try to parse as JSON string
        try:
            data = json.loads(custom_stops_arg)
        except Exception as e:
            raise ValueError(f"Failed to parse custom stops JSON string: {e}")
    
    # Parse the data - support both list of lists and list of dicts
    stops = []
    if not isinstance(data, list):
        raise ValueError("Custom stops must be a JSON array")
    
    for i, item in enumerate(data):
        if isinstance(item, list):
            # Format: [pan, tilt, zoom]
            if len(item) != 3:
                raise ValueError(f"Stop {i}: List format must have exactly 3 elements [pan, tilt, zoom]")
            try:
                pan, tilt, zoom = float(item[0]), float(item[1]), float(item[2])
            except (ValueError, TypeError) as e:
                raise ValueError(f"Stop {i}: Invalid numeric values: {e}")
        elif isinstance(item, dict):
            # Format: {"pan": 0, "tilt": 0, "zoom": 0.0}
            try:
                pan = float(item.get("pan", 0.0))
                tilt = float(item.get("tilt", 0.0))
                zoom = float(item.get("zoom", 0.0))
            except (ValueError, TypeError) as e:
                raise ValueError(f"Stop {i}: Invalid numeric values in dict: {e}")
        else:
            raise ValueError(f"Stop {i}: Must be a list [pan, tilt, zoom] or dict {{pan, tilt, zoom}}")
        
        stops.append((pan, tilt, zoom))
    
    return stops


def plan_grid_views(horizontal_stops: int = 6, vertical_stops: int = 3, pitch_min: float = -70.0, pitch_max: float = 10.0) -> List[Tuple[float, float]]:
    """Plan grid views using equally spaced stops.
    
    Args:
        horizontal_stops: Number of yaw positions (divided equally across 0-360°)
        vertical_stops: Number of pitch positions (divided equally across pitch_min to pitch_max)
        pitch_min: Minimum pitch angle in degrees (default: -70.0)
        pitch_max: Maximum pitch angle in degrees (default: 10.0)
    
    Returns:
        List of (yaw, pitch) tuples in degrees
    """
    views: List[Tuple[float, float]] = []
    
    # Calculate yaw positions: equally divide 0-360° range
    if horizontal_stops <= 1:
        yaw_positions = [0.0]
    else:
        yaw_positions = [i * 360.0 / horizontal_stops for i in range(horizontal_stops)]
    
    # Calculate pitch positions: equally divide pitch_min to pitch_max range
    if vertical_stops <= 1:
        pitch_positions = [(pitch_min + pitch_max) / 2.0]
    else:
        pitch_range = pitch_max - pitch_min
        pitch_positions = [pitch_min + i * pitch_range / (vertical_stops - 1) for i in range(vertical_stops)]
    
    # Generate all combinations
    for pitch in pitch_positions:
        for yaw in yaw_positions:
            views.append((float(yaw), float(pitch)))
    
    return views






def capture_views_and_render(
    camera_name: str,
    variant: str,
    output_dir: str,
    horizontal_stops: int,
    vertical_stops: int,
    secrets: Dict,
    tolerance: float = 1.0,
    zoom_tolerance: float = 0.05,
    pitch_offset: float = 0.0,
    yaw_offset: float = 0.0,
    use_rtsp: bool = True,
    no_http_fallback: bool = False,
    pitch_yaw_roll_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    pitch_min: float = -70.0,
    pitch_max: float = 10.0,
    custom_stops: Optional[List[Tuple[float, float, float]]] = None,
    zoom_levels: Optional[List[float]] = None,
    telemetry_crop_pct: float = 0.1125,
    validate_ocr: bool = True,
    ocr_crop_top_pct: float = 0.115,
    ocr_crop_left_pct: float = 0.35,
    ocr_tolerance: float = 1.0,
    negate_tilt: bool = True,
):
    logger.info("Checking service availability...")
    if not wait_for_services(secrets):
        raise RuntimeError("Required services are not available. Please ensure port forwards are running.")

    # Log offset configuration
    if pitch_offset != 0.0 or yaw_offset != 0.0:
        logger.info("Using telemetry offsets: pitch_offset=%.2f°, yaw_offset=%.2f° (convergence will check against commanded + offset)", pitch_offset, yaw_offset)
    
    # Log crop configuration
    if telemetry_crop_pct > 0.0:
        logger.info("Telemetry crop enabled: will crop %.2f%% from top of captured images", telemetry_crop_pct * 100)
    
    # Log OCR validation configuration
    if validate_ocr:
        logger.info("OCR validation enabled (PARALLEL MODE) using GPT-5-nano vision model: images saved immediately, OCR validation runs in background. Failed validations will delete images and remove from manifest (crop: top %.2f%%, left %.2f%%, tolerance: %.2f°%s)",
                   ocr_crop_top_pct * 100, ocr_crop_left_pct * 100, ocr_tolerance,
                   ", tilt will be negated" if negate_tilt else "")

    # Get config from secrets
    media_http = secrets.get("media_server", {}).get("base_url", "http://localhost:8800")
    rtsp_base = secrets.get("rtsp", {}).get("base_url", "rtsp://localhost:8554")
    mqtt_config = secrets.get("mqtt", {})
    mqtt_host = mqtt_config.get("host", "localhost")
    mqtt_port = mqtt_config.get("port", 1883)
    mqtt_user = mqtt_config.get("username", "admin")
    mqtt_pass = mqtt_config.get("password", "Kelasys123!")

    endpoints = build_media_server_endpoints(media_http, camera_name, variant)
    desired_stream = endpoints["stream_name"]
    if not check_stream_available(endpoints["streams"], desired_stream):
        raise RuntimeError(f"Stream not available on media server: {desired_stream}")

    topic = build_camera_telemetry_topic(camera_name, variant)
    latch = TelemetryLatch(mqtt_host, mqtt_port, mqtt_user, mqtt_pass, topic)
    cmd_url = build_onvif_command_endpoint(camera_name)

    # Create timestamped directory for this capture session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trans_pitch, trans_yaw, trans_roll = pitch_yaw_roll_translation
    if trans_pitch != 0.0 or trans_yaw != 0.0 or trans_roll != 0.0:
        # Include translation in directory name if provided
        session_dir = os.path.join(output_dir, f"{timestamp}_scan_equirect_{trans_pitch:.1f}_{trans_yaw:.1f}_{trans_roll:.1f}")
    else:
        session_dir = os.path.join(output_dir, f"{timestamp}_scan_equirect")
    frames_dir = os.path.join(session_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    logger.info("Created timestamped capture directory: %s", session_dir)
    
    manifest: Dict[str, Dict] = {}
    
    # Track pending OCR validations when using parallel mode
    manifest_lock = threading.Lock()
    pending_ocr_validations: List[threading.Thread] = []

    # Initialize offset simulation helper if needed
    trans_pitch, trans_yaw, trans_roll = pitch_yaw_roll_translation
    rotation_helper = None
    if trans_pitch != 0.0 or trans_yaw != 0.0 or trans_roll != 0.0:
        rotation_helper = CameraBaseRotation(
            pitch_deg=trans_pitch,
            yaw_deg=trans_yaw,
            roll_deg=trans_roll
        )
    
    try:
        rtsp_capture = None
        if use_rtsp:
            # Build RTSP URL
            if not rtsp_base.endswith("/"):
                rtsp_base = rtsp_base + "/"
            rtsp_url = f"{rtsp_base}{desired_stream}"
            
            # Initialize RTSP capture with helper
            try:
                rtsp_capture = RTSPCapture()
                rtsp_capture.connect(rtsp_url, use_tcp=True, low_latency=True)
                
                # Verify connectivity before starting drainer (to avoid threading conflicts)
                logger.info("Verifying RTSP stream connectivity...")
                if rtsp_capture.verify_connectivity(timeout=3.0):
                    logger.info("RTSP stream verified - successfully read test frame")
                    # Start background drainer after verification
                    rtsp_capture.start_drainer()
                else:
                    logger.warning("RTSP stream opened but cannot read frames - will use HTTP fallback for all captures")
                    rtsp_capture.close()
                    rtsp_capture = None
            except Exception as e:
                logger.warning("Failed to connect to RTSP stream (%s), will use HTTP snapshots instead", e)
                rtsp_capture = None
        else:
            logger.info("RTSP disabled (--no-rtsp flag) - will use HTTP snapshots only")
        
        try:
            # Plan views: use custom stops if provided, otherwise use grid
            if custom_stops is not None:
                # Custom stops format: (pan, tilt, zoom)
                # Convert to views with zoom for each stop
                logger.info("Using custom waypoints: %d stops", len(custom_stops))
                base_views = [(pan, tilt, zoom) for pan, tilt, zoom in custom_stops]
            else:
                # Use legacy fixed grid (returns yaw, pitch tuples without zoom)
                grid_positions = plan_grid_views(horizontal_stops=horizontal_stops, vertical_stops=vertical_stops, pitch_min=pitch_min, pitch_max=pitch_max)
                # Convert grid positions to (yaw, pitch, zoom) format with default zoom
                base_views = [(yaw, pitch, 0.0) for yaw, pitch in grid_positions]
                logger.info("Using grid pattern: %d horizontal × %d vertical = %d stops", 
                           horizontal_stops, vertical_stops, len(base_views))
            
            # Multiply views by zoom levels if provided
            if zoom_levels is not None and len(zoom_levels) > 0:
                logger.info("Multiplying each position by %d zoom levels: %s", len(zoom_levels), zoom_levels)
                views_with_zoom = []
                for yaw, pitch, base_zoom in base_views:
                    for zoom_level in zoom_levels:
                        # Use zoom_level from --zoom-levels, overriding base_zoom from custom stops
                        views_with_zoom.append((yaw, pitch, zoom_level))
                views = views_with_zoom
                logger.info("Total captures: %d positions × %d zoom levels = %d frames", 
                           len(base_views), len(zoom_levels), len(views))
            else:
                # No zoom multiplication - use base views as-is
                views = base_views
                logger.info("No zoom level multiplication - capturing %d frames", len(views))

            for idx, (yaw, pitch, zoom) in enumerate(views):
                logger.info("\n\n=== Capturing view %d/%d: yaw=%.1f, pitch=%.1f, zoom=%.2f ===", idx + 1, len(views), yaw, pitch, zoom)

                # Clear stale telemetry
                latch.get_once(timeout=0.1)

                # Move camera; use zoom from view. Clamp pitch to camera hardware limits.
                # Store the untranslated command (what we intend to point at)
                untranslated_yaw = wrap_angle_deg(yaw)
                untranslated_pitch = clamp(pitch, PITCH_MIN_HARDWARE, PITCH_MAX_HARDWARE)
                untranslated_zoom = zoom
                
                # Apply rotation translation to simulate physical camera base rotation
                if rotation_helper is not None:
                    # Calculate the actual command to send to the rotated camera
                    cmd_yaw, cmd_pitch = rotation_helper.transform_command(
                        untranslated_yaw, untranslated_pitch
                    )
                    cmd_yaw = wrap_angle_deg(cmd_yaw)
                    cmd_pitch = clamp(cmd_pitch, PITCH_MIN_HARDWARE, PITCH_MAX_HARDWARE)
                    logger.info("Applied base rotation translation: untranslated=(yaw=%.1f, pitch=%.1f) -> rotated command=(yaw=%.1f, pitch=%.1f)",
                               untranslated_yaw, untranslated_pitch, cmd_yaw, cmd_pitch)
                else:
                    cmd_yaw = untranslated_yaw
                    cmd_pitch = untranslated_pitch
                
                # Retry logic for telemetry convergence (max 2 retries = 3 total attempts)
                max_retries = 2
                telem = None
                for retry_attempt in range(max_retries + 1):
                    if retry_attempt > 0:
                        logger.warning("Retry attempt %d/%d for telemetry convergence", retry_attempt, max_retries)
                        # Clear stale telemetry before retry
                        latch.get_once(timeout=0.1)
                    
                    http_absolute_move(cmd_url, cmd_yaw, cmd_pitch, zoom=untranslated_zoom)
                    logger.info("Command sent; waiting for telemetry alignment...")

                    start_wait = time.time()
                    telemetry_timeout = 12.0
                    while time.time() - start_wait < telemetry_timeout:
                        fresh = latch.get_once(timeout=1.0)
                        if fresh is None:
                            time.sleep(0.1)
                            continue
                        # Apply offsets to expected telemetry values for convergence check
                        # We expect the camera telemetry to match the rotated command we sent
                        expected_yaw = cmd_yaw + yaw_offset
                        expected_pitch = cmd_pitch + pitch_offset
                        yaw_diff = abs(wrap_angle_deg(fresh.yaw - expected_yaw))
                        pitch_diff = abs(fresh.pitch - expected_pitch)
                        # Get zoom value (0-1 range) from raw_data
                        telemetry_zoom = extract_zoom_from_telemetry(fresh)
                        if telemetry_zoom is None:
                            # Fallback: try to derive from zoom_factor if zoom not available
                            # This shouldn't happen with current telemetry format, but handle gracefully
                            logger.warning("Telemetry missing 'zoom' field, using zoom_factor as fallback")
                            telemetry_zoom = (fresh.zoom_factor - 1.0) / 55.0  # Rough conversion, but shouldn't be used
                        zoom_diff = abs(telemetry_zoom - untranslated_zoom)
                        logger.debug(
                            "Telemetry sample: yaw=%.2f pitch=%.2f zoom=%.3f (zoom_factor=%.3f) FOVh=%.2f FOVv=%.2f | target yaw=%.2f pitch=%.2f zoom=%.3f | expected (with offset) yaw=%.2f pitch=%.2f | diffs yaw=%.2f pitch=%.2f zoom=%.3f",
                            fresh.yaw, fresh.pitch, telemetry_zoom, fresh.zoom_factor, fresh.fov_h, fresh.fov_v,
                            cmd_yaw, cmd_pitch, untranslated_zoom, expected_yaw, expected_pitch, yaw_diff, pitch_diff, zoom_diff,
                        )
                        if yaw_diff <= tolerance and pitch_diff <= tolerance and zoom_diff <= zoom_tolerance:
                            telem = fresh
                            break
                        # # Accept if yaw converged and pitch is stable (not still moving)
                        # # Use relaxed pitch tolerance since camera may have calibration offset
                        # if yaw_diff <= tolerance:
                        #     # Check if pitch has stabilized by comparing to previous sample
                        #     if telem is not None and abs(fresh.pitch - telem.pitch) < 0.5:
                        #         # Pitch has stabilized, accept this position
                        #         telem = fresh
                        #         break
                        #     telem = fresh  # Store for next iteration to check stability
                    
                    if telem is not None:
                        # Telemetry converged successfully
                        if retry_attempt > 0:
                            logger.info("Telemetry converged after %d retry attempt(s)", retry_attempt)
                        break
                    else:
                        logger.warning("Telemetry did not converge on attempt %d/%d", retry_attempt + 1, max_retries + 1)
                
                if telem is None:
                    logger.warning("Telemetry did not converge after %d attempts; skipping capture for this view", max_retries + 1)
                    continue

                # Settle delay after telemetry convergence
                # This ensures physical camera movement has fully stopped
                logger.info("Telemetry converged (yaw=%.1f, pitch=%.1f, zoom=%.2f). Settling before capture...",
                            telem.yaw, telem.pitch, getattr(telem, "zoom_factor", float("nan")))
                
                # Allow camera to focus after telemetry convergence
                time.sleep(1.0)
                
                # For HTTP: Wait for camera to physically settle after movement
                # if use_rtsp and cap is not None and cap.isOpened():
                    # frames_drained = drain_rtsp_buffer_continuously(cap, settle_time)
                    # logger.info(f"Camera settled for {settle_time}s while continuously draining {frames_drained} frames to stay at live edge")

                if not use_rtsp:
                    settle_time = 2.0  # seconds
                    time.sleep(settle_time)
                    logger.debug(f"Camera settled for {settle_time}s")

                # Capture frame (OCR validation will run in parallel if enabled)
                img = None
                capture_failed = False
                
                # Single capture attempt - OCR validation runs in parallel after saving
                logger.info("Starting capture for yaw=%.1f pitch=%.1f", telem.yaw, telem.pitch)
                try:
                    if use_rtsp and rtsp_capture is not None:
                        try:
                            img = rtsp_capture.capture_frame(timeout=6.0, convert_to_pil=True).convert("RGB")
                        except Exception as e:
                            if no_http_fallback:
                                logger.warning("RTSP capture failed (%s). HTTP fallback is disabled (--no-http-fallback). Skipping this view.", e)
                                capture_failed = True
                            else:
                                logger.warning("RTSP capture failed (%s). Falling back to HTTP snapshot...", e)
                                http_img = capture_single_frame_http(endpoints["capture"], timeout=6.0)
                                img = http_img.convert("RGB")
                    else:
                        logger.debug("Using HTTP snapshot (RTSP disabled)")
                        http_img = capture_single_frame_http(endpoints["capture"], timeout=6.0)
                        img = http_img.convert("RGB")
                except Exception as e:
                    logger.error("Frame capture failed: %s", e)
                    capture_failed = True
                
                if capture_failed:
                    continue
                
                # Override telemetry to simulate physical camera base rotation
                # When translation is applied, we override telemetry to show the untranslated (intended) direction
                # This makes the system think the camera is pointing where we intended, but physically it's rotated
                
                # Calculate the simulated roll based on the camera position and base rotation
                # This accounts for how the apparent roll changes as the camera pans around a pitched base
                if rotation_helper is not None:
                    simulated_roll = rotation_helper.calculate_image_roll(cmd_yaw, cmd_pitch)
                    logger.info("Calculated simulated roll: %.2f° (cmd_yaw=%.1f, cmd_pitch=%.1f, trans=[%.1f, %.1f, %.1f])",
                               simulated_roll, cmd_yaw, cmd_pitch, rotation_helper.pitch_deg, rotation_helper.yaw_deg, rotation_helper.roll_deg)
                else:
                    simulated_roll = 0.0
                
                # Apply roll rotation to simulate physical camera roll (before cropping for OCR)
                # Note: we negate the roll angle because image rotation is clockwise positive, but the roll angle is counterclockwise positive (ENU)
                # This makes the captured image appear as if the camera base was physically rolled
                if abs(simulated_roll) > 0.01:
                    logger.info("Applying simulated roll rotation of %.2f° to captured image", simulated_roll)
                    img = rotation_helper.rotate_image(img, -1*simulated_roll)
                
                # Save original image copy for OCR validation if enabled
                # OCR needs the uncropped image to see the telemetry overlay
                # Save after roll rotation so OCR sees the correct overlay position
                img_for_ocr = None
                if validate_ocr:
                    img_for_ocr = img.copy()
                
                # Apply telemetry crop (remove top percentage of image)
                if telemetry_crop_pct > 0.0:
                    width, height = img.size
                    crop_pixels = int(height * telemetry_crop_pct)
                    # Crop box: (left, upper, right, lower)
                    img = img.crop((0, crop_pixels, width, height))
                    logger.debug("Cropped %d pixels (%.2f%%) from top of image. New size: %dx%d", 
                                crop_pixels, telemetry_crop_pct * 100, img.size[0], img.size[1])
                
                # Save individual frame with untranslated (intended) yaw, pitch, and zoom in filename
                frame_filename = f"frame_{idx:04d}_yaw_{untranslated_yaw:.1f}_pitch_{untranslated_pitch:.1f}_zoom_{untranslated_zoom:.2f}.png"
                frame_path = os.path.join(frames_dir, frame_filename)
                img.save(frame_path)
                if rotation_helper is not None:
                    # Override: telemetry shows untranslated values (what we intended to point at)
                    saved_yaw = untranslated_yaw
                    saved_pitch = untranslated_pitch
                    saved_roll = 0.0  # Roll is always 0 in the untranslated frame
                    logger.info("Overriding telemetry: actual camera=(yaw=%.1f, pitch=%.1f) -> reported=(yaw=%.1f, pitch=%.1f, roll=%.1f)",
                               telem.yaw, telem.pitch, saved_yaw, saved_pitch, saved_roll)
                else:
                    # No translation: use actual telemetry
                    saved_yaw = telem.yaw
                    saved_pitch = telem.pitch
                    saved_roll = telem.roll
                
                # Extract zoom value (0-1 range) from raw_data
                telemetry_zoom = extract_zoom_from_telemetry(telem)
                
                frame_telemetry = {
                    "attitude": {
                        "yaw": saved_yaw,
                        "pitch": saved_pitch,
                        "roll": saved_roll
                    },
                    "field_of_view_h": telem.fov_h,
                    "field_of_view_v": telem.fov_v,
                    "zoom": telemetry_zoom,  # Save zoom (0-1 range) instead of zoom_factor
                    "zoom_factor": telem.zoom_factor,  # Keep zoom_factor for reference
                    "commanded_zoom": untranslated_zoom,
                    "capture_index": idx,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add to manifest (thread-safe if parallel OCR is enabled)
                with manifest_lock:
                    manifest[frame_filename] = frame_telemetry
                
                logger.info("Captured frame %d; actual camera yaw=%.1f pitch=%.1f; intended (saved) yaw=%.1f pitch=%.1f; FOVh=%.1f FOVv=%.1f; saved as %s", 
                           idx + 1, telem.yaw, telem.pitch, saved_yaw, saved_pitch, telem.fov_h, telem.fov_v, frame_filename)
                
                # Start parallel OCR validation if enabled
                if validate_ocr and img_for_ocr is not None:
                    # Use commanded values (with offsets) for comparison
                    expected_pan = cmd_yaw + yaw_offset
                    expected_tilt = cmd_pitch + pitch_offset
                    
                    # Create and start OCR validation thread
                    ocr_thread = threading.Thread(
                        target=run_parallel_ocr_validation,
                        args=(
                            img_for_ocr,  # Use original uncropped image for OCR
                            frame_path,
                            frame_filename,
                            expected_pan,
                            expected_tilt,
                            telem.zoom_factor,
                            ocr_crop_top_pct,
                            ocr_crop_left_pct,
                            ocr_tolerance,
                            negate_tilt,
                            manifest,
                            manifest_lock,
                            logger
                        ),
                        daemon=True  # Daemon thread so it doesn't prevent program exit
                    )
                    ocr_thread.start()
                    pending_ocr_validations.append(ocr_thread)
                    logger.info("Started parallel OCR validation thread for %s", frame_filename)

            # Wait for all pending OCR validations to complete before saving manifest
            if validate_ocr and pending_ocr_validations:
                logger.info("Waiting for %d pending OCR validation(s) to complete...", len(pending_ocr_validations))
                for thread in pending_ocr_validations:
                    thread.join(timeout=30.0)  # Wait up to 30 seconds per thread
                    if thread.is_alive():
                        logger.warning("OCR validation thread did not complete within timeout")
                logger.info("All OCR validations completed")
            
            # Save manifest (final version after OCR validations may have removed entries)
            manifest_path = os.path.join(frames_dir, "manifest.json")
            with manifest_lock:
                with open(manifest_path, 'w') as f:
                    json.dump(manifest, f, indent=2)
            logger.info("Saved frame manifest to %s with %d frames", manifest_path, len(manifest))
        finally:
            if rtsp_capture is not None:
                try:
                    rtsp_capture.close()
                except Exception:
                    pass
    finally:
        latch.stop()

def parse_args():
    p = argparse.ArgumentParser(description="Capture frames from ONVIF PTZ camera using grid scanning pattern with synchronized telemetry")
    p.add_argument("--camera-name", required=False, help="Camera name (e.g., onvifcam-dev-1). Not required when --discover is set")
    p.add_argument("--thermal", action="store_true", help="Use thermal variant (default: main)")
    p.add_argument("--discover", action="store_true", help="List available streams via media server and exit")
    p.add_argument("--output-dir", default=os.path.expanduser("~/Downloads"), help="Output directory for frames (default: ~/Downloads)")
    p.add_argument("--horizontal-stops", type=int, default=12, help="Number of horizontal (yaw) stops - equally divided across full 360° pan range")
    p.add_argument("--vertical-stops", type=int, default=5, help="Number of vertical (pitch) stops - equally divided between --pitch-min and --pitch-max")
    p.add_argument("--pitch-min", type=float, default=-70.0, help="Minimum pitch angle in degrees for capture grid (default: -70.0)")
    p.add_argument("--pitch-max", type=float, default=10.0, help="Maximum pitch angle in degrees for capture grid (default: 10.0)")
    p.add_argument("--custom-stops", type=str, help="Custom waypoints for scanning. Can be a JSON file path or a JSON string with format: [[pan, tilt, zoom], ...] or [{\"pan\": 0, \"tilt\": 0, \"zoom\": 0.0}, ...]. Overrides --horizontal-stops and --vertical-stops if provided.")
    p.add_argument("--zoom-levels", type=float, nargs='+', help="List of zoom values to multiply each grid position by. Each grid stop will be captured at each zoom level. Example: --zoom-levels 0.0 0.5 1.0 means each position captures 3 frames (one per zoom level). Total captures = grid_stops × zoom_levels")
    p.add_argument("--tolerance", type=float, default=0.15, help="Tolerance in degrees for yaw/pitch difference when waiting for telemetry convergence (default: 0.15)")
    p.add_argument("--zoom-tolerance", type=float, default=0.05, help="Tolerance for zoom factor difference when waiting for telemetry convergence (default: 0.05)")
    p.add_argument("--pitch-offset", type=float, default=0.0, help="Pitch offset in degrees to account for systematic difference between commanded and actual pitch (e.g., -11.0 if telemetry is consistently 11° less than commanded). Convergence checks against (commanded_pitch + offset)")
    p.add_argument("--yaw-offset", type=float, default=0.0, help="Yaw offset in degrees to account for systematic difference between commanded and actual yaw (e.g., 5.0 if telemetry is consistently 5° more than commanded). Convergence checks against (commanded_yaw + offset)")
    p.add_argument("--no-rtsp", action="store_true", help="Skip RTSP capture entirely and use only HTTP snapshots (faster if RTSP is not working)")
    p.add_argument("--no-http-fallback", action="store_true", help="Disable HTTP fallback when RTSP capture fails. If set, the script will skip the view instead of falling back to HTTP snapshots")
    p.add_argument(
        "--pitch-yaw-roll-translation",
        type=float,
        nargs=3,
        metavar=("PITCH", "YAW", "ROLL"),
        default=[0.0, 0.0, 0.0],
        help="Simulates physical camera base rotation (deg) as pitch yaw roll. Commands are transformed to account for the rotation, and telemetry is overridden to show untranslated values, simulating a miscalibrated camera mount."
    )
    p.add_argument("--telemetry-crop-pct", type=float, default=0, help="Percentage of image to crop from the top (default: 0, though the telemetry is at 0.1125 from the top)")
    p.add_argument("--no-validate-ocr", dest="validate_ocr", action="store_false", help="Disable OCR validation (default: enabled)")
    p.add_argument("--validate-ocr", dest="validate_ocr", action="store_true", help="Enable OCR validation of PTZ telemetry overlay after convergence. Validates that the telemetry text in the frame matches the requested values. (default: enabled)")
    p.add_argument("--ocr-crop-top-pct", type=float, default=0.115, help="Percentage of image height to crop from top for OCR telemetry region (default: 0.115 = 11.5%%)")
    p.add_argument("--ocr-crop-left-pct", type=float, default=0.35, help="Percentage of image width to crop from left for OCR telemetry region (default: 0.35 = 35%%)")
    p.add_argument("--ocr-tolerance", type=float, default=0.1, help="Tolerance in degrees for comparing OCR-extracted pan/tilt values with requested values (default: 0.1)")
    p.add_argument("--no-negate-tilt", dest="negate_tilt", action="store_false", help="Do not negate the OCR-extracted tilt value (default: negate enabled)")
    p.add_argument("--negate-tilt", dest="negate_tilt", action="store_true", help="Negate the OCR-extracted tilt value before comparison. Use when the telemetry overlay shows the negative inverse of MQTT telemetry. (default: enabled)")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(validate_ocr=True, negate_tilt=True)
    return p.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Register cleanup handler
    atexit.register(stop_port_forwards)
    signal.signal(signal.SIGINT, lambda sig, frame: (stop_port_forwards(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda sig, frame: (stop_port_forwards(), sys.exit(0)))

    # Load secrets
    secrets = load_secrets()
    if not secrets:
        logger.warning("No secrets.json found, using default values")

    # Handle discovery mode (doesn't need camera name)
    if args.discover:
        # Start port forwards for discovery (no ONVIF needed)
        required_ports = [1883, 8800, 8554]  # MQTT, HTTP, RTSP
        start_port_forwards(required_ports)
        media_http = secrets.get("media_server", {}).get("base_url", "http://localhost:8800")
        # Use a dummy camera name for building endpoints - discovery doesn't care about the actual name
        dummy_camera = args.camera_name or "dummy"
        endpoints = build_media_server_endpoints(media_http, dummy_camera, "thermal" if args.thermal else "main")
        discover_streams(endpoints["streams"])
        return

    # Validate camera args when not using discovery mode
    if not args.camera_name:
        raise SystemExit("--camera-name is required unless --discover is provided")

    # Start port forwards with ONVIF device
    required_ports = [1883, 8800, 8554, 8000]  # MQTT, HTTP, RTSP, ONVIF
    start_port_forwards(required_ports, onvif_device_name=args.camera_name)

    # Create base output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    variant = "thermal" if args.thermal else "main"
    
    # Parse custom stops if provided
    custom_stops = None
    if args.custom_stops:
        try:
            custom_stops = parse_custom_stops(args.custom_stops)
            logger.info("Parsed %d custom waypoints", len(custom_stops))
        except ValueError as e:
            raise SystemExit(f"Error parsing custom stops: {e}")
    
    # Parse zoom levels if provided
    zoom_levels = args.zoom_levels if args.zoom_levels else None
    
    # Note: capture_views_and_render will create its own timestamped directory
    capture_views_and_render(
        camera_name=args.camera_name,
        variant=variant,
        output_dir=output_dir,
        horizontal_stops=int(args.horizontal_stops),
        vertical_stops=int(args.vertical_stops),
        secrets=secrets,
        tolerance=float(args.tolerance),
        zoom_tolerance=float(args.zoom_tolerance),
        pitch_offset=float(args.pitch_offset),
        yaw_offset=float(args.yaw_offset),
        use_rtsp=not args.no_rtsp,
        no_http_fallback=args.no_http_fallback,
        pitch_yaw_roll_translation=(
            float(args.pitch_yaw_roll_translation[0]),
            float(args.pitch_yaw_roll_translation[1]),
            float(args.pitch_yaw_roll_translation[2]),
        ),
        pitch_min=float(args.pitch_min),
        pitch_max=float(args.pitch_max),
        custom_stops=custom_stops,
        zoom_levels=zoom_levels,
        telemetry_crop_pct=float(args.telemetry_crop_pct),
        validate_ocr=args.validate_ocr,
        ocr_crop_top_pct=float(args.ocr_crop_top_pct),
        ocr_crop_left_pct=float(args.ocr_crop_left_pct),
        ocr_tolerance=float(args.ocr_tolerance),
        negate_tilt=args.negate_tilt,
    )


if __name__ == "__main__":
    main()



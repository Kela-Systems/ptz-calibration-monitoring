"""Camera control utilities for ONVIF PTZ cameras."""

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def build_command_endpoint(camera_name: str, base_url: str = "http://localhost:8000") -> str:
    """
    Build the ONVIF command endpoint URL for a camera.
    
    Args:
        camera_name: Name of the camera device
        base_url: Base URL for the ONVIF service
    
    Returns:
        Full URL for camera commands
    """
    return f"{base_url}/device/onvifcam/{camera_name}/command"


def check_stream_available(streams_endpoint: str, stream_name: str, timeout: float = 5.0) -> bool:
    """
    Check if a stream is available on the media server.
    
    Args:
        streams_endpoint: URL of the streams list endpoint
        stream_name: Name of the stream to check
        timeout: Request timeout in seconds
    
    Returns:
        True if stream is available, False otherwise
    """
    try:
        response = requests.get(streams_endpoint, timeout=timeout)
        response.raise_for_status()
        streams = response.json()
        return stream_name in streams
    except Exception as e:
        logger.error(f"Failed to check stream availability: {e}")
        return False


def wait_for_services(secrets: dict, timeout: float = 30.0) -> bool:
    """
    Wait for required services to become available.
    
    Args:
        secrets: Dictionary containing service configuration
        timeout: Maximum time to wait for services
    
    Returns:
        True if all services are available, False otherwise
    """
    import socket
    
    # Check required ports
    required_ports = [1883, 8800, 8554]  # MQTT, HTTP, RTSP
    
    start_time = time.time()
    
    for port in required_ports:
        while time.time() - start_time < timeout:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1.0)
                    result = sock.connect_ex(('localhost', port))
                    if result == 0:
                        logger.info(f"Service on port {port} is available")
                        break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            logger.error(f"Timeout waiting for service on port {port}")
            return False
    
    return True


class ONVIFCameraControl:
    """Simple ONVIF camera control interface."""
    
    def __init__(self, camera_name: str, command_url: Optional[str] = None):
        """
        Initialize ONVIF camera controller.
        
        Args:
            camera_name: Name of the camera device
            command_url: Optional custom command URL
        """
        self.camera_name = camera_name
        self.command_url = command_url or build_command_endpoint(camera_name)
    
    def absolute_move(
        self,
        pan: float,
        tilt: float,
        zoom: Optional[float] = None,
        timeout: float = 5.0,
        retries: int = 3
    ) -> None:
        """
        Move camera to absolute pan/tilt/zoom position.
        
        Args:
            pan: Pan angle in degrees (-180 to 180)
            tilt: Tilt angle in degrees (-90 to 90)
            zoom: Zoom level (0.0 to 1.0), None to skip
            timeout: Request timeout in seconds
            retries: Number of retry attempts
        """
        payload = {
            "command": "absolute_move",
            "pan": pan,
            "tilt": tilt
        }
        
        if zoom is not None:
            payload["zoom"] = zoom
        
        for attempt in range(retries):
            try:
                response = requests.post(
                    self.command_url,
                    json=payload,
                    timeout=timeout
                )
                response.raise_for_status()
                logger.info(f"Camera moved to pan={pan:.1f}, tilt={tilt:.1f}, zoom={zoom}")
                return
            except Exception as e:
                if attempt < retries - 1:
                    logger.warning(f"Camera move failed (attempt {attempt + 1}/{retries}): {e}")
                    time.sleep(1.0)
                else:
                    logger.error(f"Camera move failed after {retries} attempts: {e}")
                    raise


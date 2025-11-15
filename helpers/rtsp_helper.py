"""RTSP stream capture utilities."""

import logging
import time
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class RTSPCapture:
    """RTSP stream capture helper with frame buffering."""
    
    def __init__(self):
        """Initialize RTSP capture."""
        self._cap: Optional[cv2.VideoCapture] = None
        self._url: Optional[str] = None
        self._is_connected = False
    
    def connect(self, url: str, use_tcp: bool = True, low_latency: bool = True) -> None:
        """
        Connect to RTSP stream.
        
        Args:
            url: RTSP stream URL
            use_tcp: Use TCP transport instead of UDP
            low_latency: Enable low-latency mode
        """
        self._url = url
        
        # OpenCV VideoCapture options
        self._cap = cv2.VideoCapture(url)
        
        if use_tcp:
            self._cap.set(cv2.CAP_PROP_RTSP_TRANSPORT, cv2.CAP_PROP_RTSP_TRANSPORT_TCP)
        
        if low_latency:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if self._cap.isOpened():
            self._is_connected = True
            logger.info(f"Connected to RTSP stream: {url}")
        else:
            raise RuntimeError(f"Failed to connect to RTSP stream: {url}")
    
    def verify_connectivity(self, timeout: float = 3.0) -> bool:
        """
        Verify that we can read frames from the stream.
        
        Args:
            timeout: Maximum time to wait for a frame
        
        Returns:
            True if frame can be read, False otherwise
        """
        if not self._is_connected or self._cap is None:
            return False
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                return True
            time.sleep(0.1)
        
        return False
    
    def start_drainer(self) -> None:
        """Start background frame drainer (placeholder for compatibility)."""
        # In a full implementation, this would start a background thread
        # to continuously drain the buffer. For now, we'll just log.
        logger.debug("Frame drainer started (placeholder)")
    
    def capture_frame(self, timeout: float = 5.0, convert_to_pil: bool = False):
        """
        Capture a single frame from the stream.
        
        Args:
            timeout: Maximum time to wait for a frame
            convert_to_pil: Convert to PIL Image instead of numpy array
        
        Returns:
            PIL Image if convert_to_pil=True, numpy array otherwise
        """
        if not self._is_connected or self._cap is None:
            raise RuntimeError("Not connected to RTSP stream")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            ret, frame = self._cap.read()
            
            if ret and frame is not None:
                if convert_to_pil:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(frame_rgb)
                else:
                    return frame
            
            time.sleep(0.01)
        
        raise TimeoutError(f"Failed to capture frame within {timeout} seconds")
    
    def close(self) -> None:
        """Close the RTSP connection."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._is_connected = False
            logger.info("RTSP connection closed")


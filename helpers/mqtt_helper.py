"""MQTT telemetry monitoring helper for PTZ cameras."""

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Any, Dict

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


@dataclass
class TelemetrySample:
    """Container for telemetry data from MQTT."""
    yaw: float
    pitch: float
    roll: float
    fov_h: float
    fov_v: float
    zoom_factor: float
    timestamp: float
    raw_data: Dict[str, Any]


class TelemetryLatch:
    """
    Latches onto MQTT telemetry topic and provides access to most recent sample.
    
    This class connects to an MQTT broker and subscribes to a telemetry topic,
    storing the most recent telemetry sample in a thread-safe manner.
    """
    
    def __init__(self, host: str, port: int, username: str, password: str, topic: str):
        """
        Initialize MQTT telemetry latch.
        
        Args:
            host: MQTT broker hostname
            port: MQTT broker port
            username: MQTT username
            password: MQTT password
            topic: MQTT topic to subscribe to
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.topic = topic
        
        self._latest_sample: Optional[TelemetrySample] = None
        self._lock = threading.Lock()
        self._connected = threading.Event()
        self._stop_flag = threading.Event()
        
        # Create MQTT client
        self._client = mqtt.Client(
            client_id=f"telemetry_latch_{time.time()}",
            protocol=mqtt.MQTTv311
        )
        self._client.username_pw_set(username, password)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect
        
        # Start connection
        logger.info(f"Connecting to MQTT broker at {host}:{port}")
        self._client.connect(host, port, keepalive=60)
        self._client.loop_start()
        
        # Wait for connection (with timeout)
        if not self._connected.wait(timeout=10.0):
            logger.warning("MQTT connection timeout - may not be connected")
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for MQTT connection."""
        if rc == 0:
            logger.info(f"Connected to MQTT broker, subscribing to {self.topic}")
            self._client.subscribe(self.topic)
            self._connected.set()
        else:
            logger.error(f"MQTT connection failed with code {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback for MQTT disconnection."""
        if rc != 0:
            logger.warning(f"Unexpected MQTT disconnection (code {rc})")
        self._connected.clear()
    
    def _on_message(self, client, userdata, msg):
        """Callback for MQTT message received."""
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            
            # Extract telemetry data
            # Handle both flat and nested structures
            data = payload.get("data", payload)
            
            # Parse telemetry fields
            attitude = data.get("attitude", {})
            sample = TelemetrySample(
                yaw=float(attitude.get("yaw", 0.0)),
                pitch=float(attitude.get("pitch", 0.0)),
                roll=float(attitude.get("roll", 0.0)),
                fov_h=float(data.get("field_of_view_h", 0.0)),
                fov_v=float(data.get("field_of_view_v", 0.0)),
                zoom_factor=float(data.get("zoom_factor", 1.0)),
                timestamp=time.time(),
                raw_data=payload
            )
            
            # Store latest sample (thread-safe)
            with self._lock:
                self._latest_sample = sample
                
        except Exception as e:
            logger.error(f"Error parsing MQTT telemetry: {e}")
    
    def get_once(self, timeout: float = 1.0) -> Optional[TelemetrySample]:
        """
        Get the most recent telemetry sample and clear it.
        
        This method returns the latest telemetry sample and clears the internal
        buffer. If called multiple times without new messages, it will return None.
        
        Args:
            timeout: Maximum time to wait for a sample (seconds)
        
        Returns:
            TelemetrySample if available, None otherwise
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._lock:
                if self._latest_sample is not None:
                    sample = self._latest_sample
                    self._latest_sample = None  # Clear after reading
                    return sample
            time.sleep(0.01)  # Small sleep to avoid busy waiting
        
        # Timeout - return None
        return None
    
    def get_latest(self) -> Optional[TelemetrySample]:
        """
        Get the most recent telemetry sample without clearing it.
        
        Returns:
            TelemetrySample if available, None otherwise
        """
        with self._lock:
            return self._latest_sample
    
    def is_stable(self, duration: float, tolerance: float = 0.1) -> bool:
        """
        Check if camera has been stable (not moving) for a given duration.
        
        Args:
            duration: Time period to check stability (seconds)
            tolerance: Maximum allowed change in yaw/pitch/roll (degrees)
        
        Returns:
            True if camera has been stable for the duration, False otherwise
        """
        # Get initial sample
        initial = self.get_latest()
        if initial is None:
            return False
        
        start_time = time.time()
        last_sample = initial
        
        while time.time() - start_time < duration:
            time.sleep(0.5)  # Check every 0.5 seconds
            
            current = self.get_latest()
            if current is None:
                return False
            
            # Check if movement detected
            yaw_diff = abs(current.yaw - last_sample.yaw)
            pitch_diff = abs(current.pitch - last_sample.pitch)
            roll_diff = abs(current.roll - last_sample.roll)
            
            # Handle yaw wrap-around (360° = 0°)
            if yaw_diff > 180:
                yaw_diff = 360 - yaw_diff
            
            if yaw_diff > tolerance or pitch_diff > tolerance or roll_diff > tolerance:
                # Movement detected, reset timer
                start_time = time.time()
                last_sample = current
        
        return True
    
    def stop(self):
        """Stop the MQTT client and disconnect."""
        logger.info("Stopping MQTT telemetry latch")
        self._stop_flag.set()
        self._client.loop_stop()
        self._client.disconnect()


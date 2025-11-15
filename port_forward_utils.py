"""Port forwarding utilities for connecting to remote Kubernetes services."""

import logging
import os
import subprocess
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

# Global list to track port forward processes
_port_forward_processes: List[subprocess.Popen] = []


def start_port_forwards(
    required_ports: List[int],
    onvif_device_name: Optional[str] = None,
    namespace: str = "default"
) -> None:
    """
    Start kubectl port forwards for required services.
    
    This function sets up port forwarding for the following services:
    - 1883: MQTT broker
    - 8800: Media server HTTP
    - 8554: RTSP streaming server
    - 8000: ONVIF camera control (if onvif_device_name is provided)
    
    Args:
        required_ports: List of port numbers to forward
        onvif_device_name: Optional camera device name for ONVIF port forwarding
        namespace: Kubernetes namespace (default: "default")
    """
    global _port_forward_processes
    
    logger.info(f"Starting port forwards for ports: {required_ports}")
    
    # Stop any existing port forwards first
    stop_port_forwards()
    
    # Port mapping: local_port -> (service, remote_port)
    port_mappings = {
        1883: ("mqtt-broker", 1883),
        8800: ("media-server", 8800),
        8554: ("rtsp-server", 8554),
    }
    
    # Add ONVIF port forwarding if device name is provided
    if onvif_device_name and 8000 in required_ports:
        port_mappings[8000] = (f"onvif-{onvif_device_name}", 8000)
    
    # Start port forwards
    for local_port in required_ports:
        if local_port not in port_mappings:
            logger.warning(f"Unknown port {local_port}, skipping")
            continue
        
        service_name, remote_port = port_mappings[local_port]
        
        # Build kubectl port-forward command
        cmd = [
            "kubectl",
            "port-forward",
            f"service/{service_name}",
            f"{local_port}:{remote_port}",
            "-n", namespace
        ]
        
        try:
            # Start port forward process in background
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL
            )
            
            _port_forward_processes.append(process)
            logger.info(f"Started port forward: localhost:{local_port} -> {service_name}:{remote_port}")
            
            # Small delay to let port forward establish
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Failed to start port forward for {service_name}: {e}")
    
    # Give all port forwards time to establish
    if _port_forward_processes:
        logger.info(f"Waiting for {len(_port_forward_processes)} port forward(s) to establish...")
        time.sleep(2.0)


def stop_port_forwards() -> None:
    """Stop all active port forward processes."""
    global _port_forward_processes
    
    if not _port_forward_processes:
        return
    
    logger.info(f"Stopping {len(_port_forward_processes)} port forward(s)")
    
    for process in _port_forward_processes:
        try:
            process.terminate()
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
        except Exception as e:
            logger.warning(f"Error stopping port forward: {e}")
    
    _port_forward_processes.clear()
    logger.info("All port forwards stopped")


def check_port_available(port: int) -> bool:
    """
    Check if a port is available for use.
    
    Args:
        port: Port number to check
    
    Returns:
        True if port is available, False otherwise
    """
    import socket
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(('localhost', port))
            return True
    except OSError:
        return False


def wait_for_port(port: int, timeout: float = 10.0) -> bool:
    """
    Wait for a port to become available (i.e., a service is listening on it).
    
    Args:
        port: Port number to wait for
        timeout: Maximum time to wait (seconds)
    
    Returns:
        True if port becomes available within timeout, False otherwise
    """
    import socket
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)
                result = sock.connect_ex(('localhost', port))
                if result == 0:
                    logger.info(f"Port {port} is now available")
                    return True
        except Exception:
            pass
        
        time.sleep(0.5)
    
    logger.warning(f"Timeout waiting for port {port}")
    return False


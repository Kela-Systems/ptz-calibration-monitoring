#!/usr/bin/env python3
"""
Port forwarding utilities for kubectl port-forward management.

This module provides functions to start and stop kubectl port-forward processes
for accessing services in a Kubernetes cluster.
"""

import logging
import subprocess
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

# Global list to track active port-forward processes
_active_port_forwards = []


def start_port_forwards(
    required_ports: List[int],
    onvif_device_name: Optional[str] = None
) -> bool:
    """
    Start kubectl port-forward for required services.
    
    Args:
        required_ports: List of ports to forward (1883=MQTT, 8800=HTTP, 8554=RTSP, 8000=ONVIF)
        onvif_device_name: Optional camera name for ONVIF device port forwarding
        
    Returns:
        True if successful, False otherwise
    """
    global _active_port_forwards
    
    logger.info(f"Starting port forwards for ports: {required_ports}")
    if onvif_device_name:
        logger.info(f"ONVIF device: {onvif_device_name}")
    
    # Port mapping for services
    port_service_map = {
        1883: "mosquitto",  # MQTT broker
        8800: "media-server",  # Media server HTTP
        8554: "media-server",  # Media server RTSP
        8000: f"onvif-{onvif_device_name}" if onvif_device_name else None  # ONVIF device
    }
    
    # Start port forwards for each required port
    for port in required_ports:
        service = port_service_map.get(port)
        if not service:
            logger.warning(f"Unknown service for port {port}, skipping")
            continue
        
        try:
            # Build kubectl port-forward command
            cmd = [
                "kubectl",
                "port-forward",
                f"svc/{service}",
                f"{port}:{port}",
                "--address=0.0.0.0"
            ]
            
            logger.info(f"Starting port-forward: {' '.join(cmd)}")
            
            # Start the process in the background
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            _active_port_forwards.append(process)
            logger.info(f"Started port-forward for {service} on port {port} (PID: {process.pid})")
            
        except Exception as e:
            logger.error(f"Failed to start port-forward for port {port}: {e}")
            return False
    
    # Wait a moment for port forwards to establish
    if _active_port_forwards:
        logger.info("Waiting for port forwards to establish...")
        time.sleep(3)
        
        # Check if any processes have already died
        for process in _active_port_forwards[:]:
            if process.poll() is not None:
                # Process has terminated
                _, stderr = process.communicate()
                logger.error(f"Port-forward process (PID {process.pid}) failed: {stderr}")
                _active_port_forwards.remove(process)
                return False
    
    logger.info(f"Successfully started {len(_active_port_forwards)} port-forward(s)")
    return True


def stop_port_forwards() -> None:
    """
    Stop all active port-forward processes.
    """
    global _active_port_forwards
    
    if not _active_port_forwards:
        logger.debug("No active port-forwards to stop")
        return
    
    logger.info(f"Stopping {len(_active_port_forwards)} port-forward(s)...")
    
    for process in _active_port_forwards:
        try:
            if process.poll() is None:
                # Process is still running
                logger.info(f"Terminating port-forward process (PID: {process.pid})")
                process.terminate()
                
                # Wait up to 5 seconds for graceful termination
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Process {process.pid} did not terminate gracefully, killing...")
                    process.kill()
                    process.wait()
                
                logger.info(f"Port-forward process {process.pid} stopped")
        except Exception as e:
            logger.error(f"Error stopping port-forward process: {e}")
    
    _active_port_forwards.clear()
    logger.info("All port-forwards stopped")


def get_active_port_forwards() -> int:
    """
    Get the number of active port-forward processes.
    
    Returns:
        Number of active port-forwards
    """
    global _active_port_forwards
    
    # Clean up any dead processes
    _active_port_forwards = [p for p in _active_port_forwards if p.poll() is None]
    
    return len(_active_port_forwards)


"""
Monitoring module for PTZ camera calibration monitoring.

This module provides functionality for collecting reference scans from PTZ cameras
and monitoring their calibration over time.
"""

from .reference_collector import ReferenceCollector

__all__ = ['ReferenceCollector']


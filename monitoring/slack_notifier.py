"""
Slack notification module for calibration monitoring alerts.

Sends formatted calibration reports to Slack with threshold-based alerting.
Supports both OAuth tokens and webhook URLs for maximum flexibility.
"""

import os
import requests
from datetime import datetime
from typing import Optional, List
import logging

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_SDK_AVAILABLE = True
except ImportError:
    SLACK_SDK_AVAILABLE = False
    WebClient = None
    SlackApiError = Exception

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SlackNotifier:
    """
    Handles Slack notifications for camera calibration monitoring.
    
    Supports two authentication methods:
    1. OAuth tokens (SLACK_ACCESS_TOKEN) - Recommended, more flexible
    2. Webhook URLs (SLACK_WEBHOOK_URL) - Simpler, limited functionality
    
    Sends formatted messages with:
    - Deployment name, Device ID, Timestamp
    - Pitch/Yaw/Roll offsets
    - Mode (passive/active)
    - Success status
    - Threshold-based alerting (🚨 if any offset > 0.5°, ✅ otherwise)
    """
    
    # Alert threshold in degrees
    ALERT_THRESHOLD = 0.5
    
    # Emojis
    SUCCESS_EMOJI = "✅"
    ALERT_EMOJI = "🚨"
    
    # Default channel
    DEFAULT_CHANNEL = "calibration-monitoring"
    
    def __init__(
        self,
        access_token: Optional[str] = None,
        webhook_url: Optional[str] = None,
        channel: Optional[str] = None
    ):
        """
        Initialize the Slack notifier.
        
        Priority order for authentication:
        1. access_token parameter
        2. SLACK_ACCESS_TOKEN environment variable
        3. webhook_url parameter
        4. SLACK_WEBHOOK_URL environment variable
        
        Args:
            access_token: Slack OAuth access token (preferred method)
            webhook_url: Slack webhook URL (fallback method)
            channel: Target Slack channel name (without #). Defaults to "calibration_monitoring"
        """
        self.channel = channel or os.getenv('SLACK_CHANNEL', self.DEFAULT_CHANNEL)
        # Ensure channel doesn't have # prefix
        if self.channel and self.channel.startswith('#'):
            self.channel = self.channel[1:]
        self.access_token = access_token or os.getenv('SLACK_ACCESS_TOKEN')
        self.webhook_url = webhook_url or os.getenv('SLACK_WEBHOOK_URL')
        self.client = None
        
        # Initialize Slack Web API client if token is available
        if self.access_token:
            if not SLACK_SDK_AVAILABLE:
                logger.error(
                    "slack-sdk is not installed. Install it with: pip install slack-sdk"
                )
            else:
                self.client = WebClient(token=self.access_token)
                logger.info("Initialized Slack notifier with OAuth token (Web API)")
        elif self.webhook_url:
            logger.info("Initialized Slack notifier with webhook URL")
        else:
            logger.warning(
                "No Slack credentials provided. Set SLACK_ACCESS_TOKEN or "
                "SLACK_WEBHOOK_URL environment variable, or pass credentials to constructor."
            )
    
    def _check_threshold(self, pitch: float, yaw: float, roll: float) -> bool:
        """
        Check if any offset exceeds the alert threshold.
        
        Args:
            pitch: Pitch offset in degrees
            yaw: Yaw offset in degrees
            roll: Roll offset in degrees
            
        Returns:
            True if any offset exceeds threshold, False otherwise
        """
        return (abs(pitch) > self.ALERT_THRESHOLD or 
                abs(yaw) > self.ALERT_THRESHOLD or 
                abs(roll) > self.ALERT_THRESHOLD)
    
    def _format_timestamp(self, timestamp: Optional[str] = None) -> str:
        """
        Format timestamp to ISO 8601 format.
        
        Args:
            timestamp: Timestamp string. If None, uses current time.
            
        Returns:
            ISO 8601 formatted timestamp string
        """
        if timestamp is None:
            from datetime import timezone
            return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # If timestamp is already a string in good format, return it
        if isinstance(timestamp, str):
            return timestamp
        
        # If it's a datetime object, format it
        if isinstance(timestamp, datetime):
            return timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        return str(timestamp)
    
    def _format_message(
        self,
        deployment: str,
        device_id: str,
        timestamp: str,
        pitch: float,
        yaw: float,
        roll: float,
        mode: str,
        success: bool,
        failure_logs: Optional[List[str]] = None
    ) -> str:
        """
        Format the calibration report message.
        
        Args:
            deployment: Deployment name
            device_id: Device identifier
            timestamp: ISO timestamp
            pitch: Pitch offset in degrees
            yaw: Yaw offset in degrees
            roll: Roll offset in degrees
            mode: Operation mode (passive/active)
            success: Whether calibration was successful
            failure_logs: Optional list of failure log messages
            
        Returns:
            Formatted message string
        """
        # Determine emoji based on threshold
        threshold_exceeded = self._check_threshold(pitch, yaw, roll)
        emoji = self.ALERT_EMOJI if threshold_exceeded else self.SUCCESS_EMOJI
        
        # Build base message
        message_lines = [
            f"{emoji} Camera Calibration Report",
            f"Deployment: {deployment}",
            f"Device: {device_id}",
            f"Timestamp: {timestamp}",
            f"Offsets: Pitch={pitch:.1f}°, Yaw={yaw:.1f}°, Roll={roll:.1f}°",
            f"Mode: {mode}",
            f"Success: {'Yes' if success else 'No'}"
        ]
        
        # Add failure logs if provided and success is False
        if not success and failure_logs:
            message_lines.append("\nFailure Logs:")
            for log in failure_logs:
                message_lines.append(f"  • {log}")
        
        return "\n".join(message_lines)
    
    def _send_via_webhook(self, message: str) -> bool:
        """
        Send message via webhook URL.
        
        Args:
            message: Formatted message text
            
        Returns:
            True if successful, False otherwise
        """
        if not self.webhook_url:
            return False
        
        # Webhook doesn't need channel specified if webhook is configured for specific channel
        payload = {
            "text": message
        }
        
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return True
            else:
                logger.error(
                    f"Webhook failed. Status: {response.status_code}, "
                    f"Response: {response.text}"
                )
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending via webhook: {e}")
            return False
    
    def _send_via_api(self, message: str) -> bool:
        """
        Send message via Slack Web API using OAuth token.
        
        Args:
            message: Formatted message text
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False
        
        try:
            response = self.client.chat_postMessage(
                channel=self.channel,
                text=message
            )
            
            if response["ok"]:
                return True
            else:
                logger.error(f"Slack API returned ok=False: {response}")
                return False
                
        except SlackApiError as e:
            logger.error(f"Slack API error: {e.response['error']}")
            return False
        except Exception as e:
            logger.error(f"Error sending via Slack API: {e}")
            return False
    
    def send_calibration_report(
        self,
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
    ) -> bool:
        """
        Send a calibration report to Slack.
        
        Args:
            deployment: Deployment name (e.g., "gan-shomron-dell")
            device_id: Device identifier (e.g., "onvifcam-1")
            pitch: Pitch offset in degrees
            yaw: Yaw offset in degrees
            roll: Roll offset in degrees
            mode: Operation mode ("passive" or "active"), defaults to "passive"
            success: Whether calibration was successful, defaults to True
            timestamp: ISO timestamp string. If None, uses current time.
            failure_logs: Optional list of failure messages to include when success=False
            channel: Optional channel override (without #)
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        if not self.client and not self.webhook_url:
            logger.error("Cannot send notification: No credentials configured")
            return False
        
        # Format timestamp
        formatted_timestamp = self._format_timestamp(timestamp)
        
        # Format message
        message = self._format_message(
            deployment=deployment,
            device_id=device_id,
            timestamp=formatted_timestamp,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            mode=mode,
            success=success,
            failure_logs=failure_logs
        )
        
        # Override channel if provided
        original_channel = None
        if channel:
            original_channel = self.channel
            self.channel = channel
        
        # Try Web API first (preferred), fall back to webhook
        result = False
        if self.client:
            result = self._send_via_api(message)
            if result:
                logger.info(
                    f"Successfully sent calibration report for {device_id} "
                    f"on {deployment} via Slack API"
                )
        
        if not result and self.webhook_url:
            result = self._send_via_webhook(message)
            if result:
                logger.info(
                    f"Successfully sent calibration report for {device_id} "
                    f"on {deployment} via webhook"
                )
        
        # Restore original channel if it was overridden
        if original_channel:
            self.channel = original_channel
        
        return result
    
    def send_test_message(self) -> bool:
        """
        Send a test calibration report with sample data.
        
        Returns:
            True if message was sent successfully, False otherwise
        """
        return self.send_calibration_report(
            deployment="test-deployment",
            device_id="test-camera-1",
            pitch=0.2,
            yaw=0.3,
            roll=0.1,
            mode="passive",
            success=True
        )


# Example usage
if __name__ == "__main__":
    # Initialize notifier (uses SLACK_ACCESS_TOKEN or SLACK_WEBHOOK_URL from environment)
    notifier = SlackNotifier()
    
    # Example 1: Success case with low offsets
    notifier.send_calibration_report(
        deployment="gan-shomron-dell",
        device_id="onvifcam-1",
        pitch=0.2,
        yaw=0.3,
        roll=0.1,
        mode="passive",
        success=True
    )
    
    # Example 2: Alert case with high offsets
    notifier.send_calibration_report(
        deployment="gan-shomron-dell",
        device_id="onvifcam-2",
        pitch=0.8,
        yaw=0.6,
        roll=0.2,
        mode="active",
        success=True
    )
    
    # Example 3: Failure case with logs
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
            "Insufficient matching points (found 12, need 50)"
        ]
    )

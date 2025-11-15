"""
Unit tests for the SlackNotifier module.

Run with: pytest monitoring/test_slack_notifier.py
"""

import pytest
from unittest.mock import patch, MagicMock
from monitoring.slack_notifier import SlackNotifier


class TestSlackNotifier:
    """Test suite for SlackNotifier class."""
    
    def test_init_with_access_token(self):
        """Test initialization with OAuth access token."""
        with patch('monitoring.slack_notifier.SLACK_SDK_AVAILABLE', True):
            with patch('monitoring.slack_notifier.WebClient') as mock_client:
                notifier = SlackNotifier(access_token="xoxb-test-token")
                assert notifier.access_token == "xoxb-test-token"
                assert notifier.client is not None
    
    def test_init_with_webhook_url(self):
        """Test initialization with explicit webhook URL."""
        notifier = SlackNotifier(webhook_url="https://test.webhook.url")
        assert notifier.webhook_url == "https://test.webhook.url"
    
    def test_init_from_environment_token(self):
        """Test initialization from SLACK_ACCESS_TOKEN environment variable."""
        with patch.dict('os.environ', {'SLACK_ACCESS_TOKEN': 'xoxb-env-token'}):
            with patch('monitoring.slack_notifier.SLACK_SDK_AVAILABLE', True):
                with patch('monitoring.slack_notifier.WebClient'):
                    notifier = SlackNotifier()
                    assert notifier.access_token == "xoxb-env-token"
    
    def test_init_from_environment_webhook(self):
        """Test initialization from SLACK_WEBHOOK_URL environment variable."""
        with patch.dict('os.environ', {'SLACK_WEBHOOK_URL': 'https://env.webhook.url'}, clear=True):
            notifier = SlackNotifier()
            assert notifier.webhook_url == "https://env.webhook.url"
    
    def test_check_threshold_below(self):
        """Test threshold check with offsets below threshold."""
        notifier = SlackNotifier()
        result = notifier._check_threshold(pitch=0.2, yaw=0.3, roll=0.1)
        assert result is False
    
    def test_check_threshold_at_boundary(self):
        """Test threshold check with offsets at boundary."""
        notifier = SlackNotifier()
        result = notifier._check_threshold(pitch=0.5, yaw=0.5, roll=0.5)
        assert result is False
    
    def test_check_threshold_above(self):
        """Test threshold check with offsets above threshold."""
        notifier = SlackNotifier()
        result = notifier._check_threshold(pitch=0.6, yaw=0.3, roll=0.1)
        assert result is True
    
    def test_check_threshold_negative_values(self):
        """Test threshold check with negative offset values."""
        notifier = SlackNotifier()
        result = notifier._check_threshold(pitch=-0.6, yaw=0.3, roll=0.1)
        assert result is True
    
    def test_format_timestamp_none(self):
        """Test timestamp formatting with None input."""
        notifier = SlackNotifier()
        timestamp = notifier._format_timestamp(None)
        # Should return current time in ISO format
        assert 'T' in timestamp
        assert 'Z' in timestamp
    
    def test_format_timestamp_string(self):
        """Test timestamp formatting with string input."""
        notifier = SlackNotifier()
        input_time = "2025-11-15T21:00:00Z"
        timestamp = notifier._format_timestamp(input_time)
        assert timestamp == input_time
    
    def test_format_message_success(self):
        """Test message formatting for success case."""
        notifier = SlackNotifier()
        message = notifier._format_message(
            deployment="test-deployment",
            device_id="test-device",
            timestamp="2025-11-15T21:00:00Z",
            pitch=0.2,
            yaw=0.3,
            roll=0.1,
            mode="passive",
            success=True
        )
        
        assert "✅ Camera Calibration Report" in message
        assert "Deployment: test-deployment" in message
        assert "Device: test-device" in message
        assert "Pitch=0.2°, Yaw=0.3°, Roll=0.1°" in message
        assert "Mode: passive" in message
        assert "Success: Yes" in message
    
    def test_format_message_alert(self):
        """Test message formatting for alert case."""
        notifier = SlackNotifier()
        message = notifier._format_message(
            deployment="test-deployment",
            device_id="test-device",
            timestamp="2025-11-15T21:00:00Z",
            pitch=0.8,  # Above threshold
            yaw=0.3,
            roll=0.1,
            mode="active",
            success=True
        )
        
        assert "🚨 Camera Calibration Report" in message
        assert "Mode: active" in message
    
    def test_format_message_with_failure_logs(self):
        """Test message formatting with failure logs."""
        notifier = SlackNotifier()
        failure_logs = ["Error 1", "Error 2"]
        message = notifier._format_message(
            deployment="test-deployment",
            device_id="test-device",
            timestamp="2025-11-15T21:00:00Z",
            pitch=1.5,
            yaw=2.0,
            roll=0.9,
            mode="passive",
            success=False,
            failure_logs=failure_logs
        )
        
        assert "Success: No" in message
        assert "Failure Logs:" in message
        assert "Error 1" in message
        assert "Error 2" in message
    
    @patch('monitoring.slack_notifier.requests.post')
    def test_send_calibration_report_via_webhook(self, mock_post):
        """Test successful sending of calibration report via webhook."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        notifier = SlackNotifier(webhook_url="https://test.webhook.url")
        result = notifier.send_calibration_report(
            deployment="test-deployment",
            device_id="test-device",
            pitch=0.2,
            yaw=0.3,
            roll=0.1
        )
        
        assert result is True
        assert mock_post.called
        
        # Verify the payload
        call_args = mock_post.call_args
        payload = call_args.kwargs['json']
        assert payload['channel'] == "#calibration_monitoring"
        assert "Camera Calibration Report" in payload['text']
    
    def test_send_calibration_report_via_api(self):
        """Test successful sending of calibration report via Slack API."""
        with patch('monitoring.slack_notifier.SLACK_SDK_AVAILABLE', True):
            with patch('monitoring.slack_notifier.WebClient') as mock_client_class:
                # Mock the client instance
                mock_client = MagicMock()
                mock_client.chat_postMessage.return_value = {"ok": True}
                mock_client_class.return_value = mock_client
                
                notifier = SlackNotifier(access_token="xoxb-test-token")
                result = notifier.send_calibration_report(
                    deployment="test-deployment",
                    device_id="test-device",
                    pitch=0.2,
                    yaw=0.3,
                    roll=0.1
                )
                
                assert result is True
                assert mock_client.chat_postMessage.called
                
                # Verify the call
                call_args = mock_client.chat_postMessage.call_args
                assert call_args.kwargs['channel'] == "calibration_monitoring"
                assert "Camera Calibration Report" in call_args.kwargs['text']
    
    @patch('monitoring.slack_notifier.requests.post')
    def test_send_calibration_report_failure(self, mock_post):
        """Test failed sending of calibration report."""
        # Mock failed response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response
        
        notifier = SlackNotifier(webhook_url="https://test.webhook.url")
        result = notifier.send_calibration_report(
            deployment="test-deployment",
            device_id="test-device",
            pitch=0.2,
            yaw=0.3,
            roll=0.1
        )
        
        assert result is False
    
    def test_send_calibration_report_no_webhook(self):
        """Test sending without webhook URL configured."""
        notifier = SlackNotifier()  # No webhook URL
        result = notifier.send_calibration_report(
            deployment="test-deployment",
            device_id="test-device",
            pitch=0.2,
            yaw=0.3,
            roll=0.1
        )
        
        assert result is False
    
    @patch('monitoring.slack_notifier.requests.post')
    def test_send_calibration_report_network_error(self, mock_post):
        """Test handling of network errors."""
        # Mock network exception using RequestException (which is properly handled)
        from requests.exceptions import RequestException
        mock_post.side_effect = RequestException("Network error")
        
        notifier = SlackNotifier(webhook_url="https://test.webhook.url")
        result = notifier.send_calibration_report(
            deployment="test-deployment",
            device_id="test-device",
            pitch=0.2,
            yaw=0.3,
            roll=0.1
        )
        
        assert result is False
    
    @patch('monitoring.slack_notifier.requests.post')
    def test_send_test_message(self, mock_post):
        """Test the send_test_message convenience method."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        notifier = SlackNotifier(webhook_url="https://test.webhook.url")
        result = notifier.send_test_message()
        
        assert result is True
        assert mock_post.called
        
        # Verify test data in payload
        call_args = mock_post.call_args
        payload = call_args.kwargs['json']
        assert "test-deployment" in payload['text']
        assert "test-camera-1" in payload['text']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


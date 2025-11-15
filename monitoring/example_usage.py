"""
Example usage of the SlackNotifier for calibration monitoring.

Run this script to test the Slack notification functionality.

Setup (OAuth Token - Recommended):
    export SLACK_ACCESS_TOKEN="xoxb-your-token-here"
    export SLACK_CHANNEL="calibration-monitoring"  # Optional
    python monitoring/example_usage.py

Alternative Setup (Webhook):
    export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    python monitoring/example_usage.py
"""

from monitoring.slack_notifier import SlackNotifier


def main():
    """Run example notifications."""
    
    # Initialize notifier (reads SLACK_ACCESS_TOKEN or SLACK_WEBHOOK_URL from environment)
    notifier = SlackNotifier()
    
    print("=" * 70)
    print("Slack Notifier - Example Usage")
    print("=" * 70)
    print()
    
    # Example 1: Success case with low offsets (✅ emoji)
    print("Example 1: Sending success report with low offsets...")
    success_1 = notifier.send_calibration_report(
        deployment="gan-shomron-dell",
        device_id="onvifcam-1",
        pitch=0.2,
        yaw=0.3,
        roll=0.1,
        mode="passive",
        success=True
    )
    print(f"  Result: {'✅ Sent successfully' if success_1 else '❌ Failed to send'}")
    print()
    
    # Example 2: Alert case with high offsets (🚨 emoji)
    print("Example 2: Sending alert report with high offsets...")
    success_2 = notifier.send_calibration_report(
        deployment="gan-shomron-dell",
        device_id="onvifcam-2",
        pitch=0.8,  # Exceeds 0.5° threshold
        yaw=0.6,    # Exceeds 0.5° threshold
        roll=0.2,
        mode="active",
        success=True
    )
    print(f"  Result: {'✅ Sent successfully' if success_2 else '❌ Failed to send'}")
    print()
    
    # Example 3: Failure case with logs
    print("Example 3: Sending failure report with error logs...")
    success_3 = notifier.send_calibration_report(
        deployment="gan-shomron-dell",
        device_id="onvifcam-3",
        pitch=1.5,
        yaw=2.0,
        roll=0.9,
        mode="passive",
        success=False,
        failure_logs=[
            "Failed to detect features in reference image",
            "Insufficient matching points (found 12, need 50)",
            "Camera may need recalibration"
        ]
    )
    print(f"  Result: {'✅ Sent successfully' if success_3 else '❌ Failed to send'}")
    print()
    
    # Example 4: Using custom timestamp
    print("Example 4: Sending report with custom timestamp...")
    success_4 = notifier.send_calibration_report(
        deployment="test-site-alpha",
        device_id="ptz-camera-04",
        pitch=0.15,
        yaw=0.25,
        roll=0.05,
        mode="active",
        success=True,
        timestamp="2025-11-15T14:30:00Z"
    )
    print(f"  Result: {'✅ Sent successfully' if success_4 else '❌ Failed to send'}")
    print()
    
    print("=" * 70)
    print(f"Summary: {sum([success_1, success_2, success_3, success_4])}/4 messages sent")
    print("=" * 70)
    
    # Test the convenience method
    print("\nTesting send_test_message() convenience method...")
    test_result = notifier.send_test_message()
    print(f"  Result: {'✅ Sent successfully' if test_result else '❌ Failed to send'}")

if __name__ == "__main__":
    main()


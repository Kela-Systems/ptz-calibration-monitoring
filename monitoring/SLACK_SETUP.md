# Slack Integration Setup Guide

This guide will help you set up Slack notifications for calibration monitoring using OAuth tokens.

## Setup with OAuth Access Token (Recommended)

OAuth tokens provide more flexibility and better control over your Slack integration.

### Step 1: Create a Slack App

1. Go to [Slack API Apps](https://api.slack.com/apps)
2. Click **"Create New App"**
3. Choose **"From scratch"**
4. Name your app (e.g., "PTZ Calibration Monitor")
5. Select your workspace
6. Click **"Create App"**

### Step 2: Configure OAuth Permissions

1. In your app settings, navigate to **"OAuth & Permissions"** (in the left sidebar)
2. Scroll down to **"Scopes"** section
3. Under **"Bot Token Scopes"**, click **"Add an OAuth Scope"**
4. Add the following scope:
   - `chat:write` - Post messages to channels

### Step 3: Install App to Workspace

1. Scroll back to the top of the **"OAuth & Permissions"** page
2. Click **"Install to Workspace"**
3. Review the permissions and click **"Allow"**
4. You'll see a **"Bot User OAuth Token"** that starts with `xoxb-`
5. Copy this token (you'll need it in the next step)

### Step 4: Invite Bot to Channel

1. In Slack, go to your `#calibration_monitoring` channel (create it if needed)
2. Type: `/invite @PTZ Calibration Monitor`
3. Or click the channel name → Integrations → Add apps → select your app

### Step 5: Configure Environment Variables

Set the following environment variables:

```bash
export SLACK_ACCESS_TOKEN="xoxb-your-actual-token-here"
export SLACK_CHANNEL="calibration_monitoring"  # Optional, defaults to "calibration_monitoring"
```

To make these permanent, add them to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
echo 'export SLACK_ACCESS_TOKEN="xoxb-your-actual-token-here"' >> ~/.bashrc
echo 'export SLACK_CHANNEL="calibration_monitoring"' >> ~/.bashrc
source ~/.bashrc
```

### Step 6: Install Python Dependencies

Make sure you have the Slack SDK installed:

```bash
pip install slack-sdk>=3.19.0
```

Or install all project dependencies:

```bash
pip install -r requirements.txt
```

### Step 7: Test the Integration

Run the example script to test:

```bash
python monitoring/example_usage.py
```

Or use the test method in your code:

```python
from monitoring import SlackNotifier

notifier = SlackNotifier()
notifier.send_test_message()
```

## Alternative: Webhook URL Setup

If you prefer using webhooks instead of OAuth tokens:

### Step 1: Create Incoming Webhook

1. Go to [Slack Incoming Webhooks](https://api.slack.com/messaging/webhooks)
2. Click **"Create your Slack app"** or select an existing app
3. Enable **"Incoming Webhooks"**
4. Click **"Add New Webhook to Workspace"**
5. Select `#calibration_monitoring` channel
6. Copy the webhook URL

### Step 2: Configure Environment Variable

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

## Using with Refresh Tokens

If you have `SLACK_REFRESH_TOKEN` for long-lived authentication, you'll need to implement token refresh logic. The current implementation supports:

1. **Direct Access Token** (`SLACK_ACCESS_TOKEN`) - Used directly for API calls
2. **Webhook URL** (`SLACK_WEBHOOK_URL`) - Fallback method

For production use with refresh tokens, consider implementing an automatic token refresh mechanism using the Slack OAuth API.

## Troubleshooting

### "No credentials configured" error

- Make sure `SLACK_ACCESS_TOKEN` or `SLACK_WEBHOOK_URL` is set
- Verify the environment variable is exported in the current shell session
- Check for typos in the variable name

### "Channel not found" error

- Ensure the bot has been invited to the target channel
- Use the channel name without the `#` prefix
- Channel must exist before sending messages

### "Not authed" or "Invalid token" error

- Verify your token starts with `xoxb-`
- Check that the token hasn't been revoked
- Ensure you copied the entire token (they're quite long)

### Messages not appearing

- Check the bot is a member of the channel
- Verify the channel name is correct
- Look at the Python logs for error messages

## Usage Examples

### Basic Usage

```python
from monitoring import SlackNotifier

notifier = SlackNotifier()

notifier.send_calibration_report(
    deployment="site-name",
    device_id="camera-1",
    pitch=0.2,
    yaw=0.3,
    roll=0.1,
    mode="passive",
    success=True
)
```

### With Failure Logs

```python
notifier.send_calibration_report(
    deployment="site-name",
    device_id="camera-1",
    pitch=1.5,
    yaw=2.0,
    roll=0.9,
    mode="active",
    success=False,
    failure_logs=[
        "Feature detection failed",
        "Insufficient matching points"
    ]
)
```

### Custom Channel

```python
notifier.send_calibration_report(
    deployment="site-name",
    device_id="camera-1",
    pitch=0.2,
    yaw=0.3,
    roll=0.1,
    channel="alerts-critical"  # Override default channel
)
```

## Security Best Practices

1. **Never commit tokens to version control**
   - Add `.env` to `.gitignore` if using env files
   - Use environment variables or secret management systems

2. **Rotate tokens periodically**
   - Regenerate tokens every 90 days
   - Use refresh tokens for long-running services

3. **Use minimal scopes**
   - Only request `chat:write` scope
   - Don't add unnecessary permissions

4. **Monitor token usage**
   - Check Slack app dashboard for unusual activity
   - Set up alerts for failed authentication attempts


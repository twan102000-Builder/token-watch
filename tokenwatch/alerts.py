import os
import json
import logging
import httpx
from typing import Optional, Dict, Any
from tokenwatch.models import AlertPayload

logger = logging.getLogger("tokenwatch.alerts")

class LocalAlertDispatcher:
    """
    Dispatches alerts privately on your local home network:
    - Home Assistant local webhook
    - Local logging
    """
    def __init__(self, ha_webhook_url: Optional[str] = None):
        self.ha_webhook_url = ha_webhook_url or os.environ.get("TOKENWATCH_HA_WEBHOOK_URL")

    async def send_alert(self, alert: AlertPayload):
        logger.warning(f"[ALERT][{alert.severity}] {alert.title}: {alert.message}")

        # Post to Home Assistant local webhook if configured
        if self.ha_webhook_url:
            try:
                async with httpx.AsyncClient(timeout=4.0) as client:
                    await client.post(
                        self.ha_webhook_url,
                        json={
                            "title": alert.title,
                            "message": alert.message,
                            "alert_type": alert.alert_type,
                            "severity": alert.severity,
                            "details": alert.details or {}
                        }
                    )
            except Exception as e:
                logger.error(f"Failed to post alert to Home Assistant: {e}")

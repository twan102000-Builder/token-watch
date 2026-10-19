import logging
import os

import httpx

from tokenwatch.models import AlertPayload
from tokenwatch.security import load_dotenv_if_present

logger = logging.getLogger("tokenwatch.alerts")


class LocalAlertDispatcher:
    """
    Dispatches alerts privately on your local home network:
    - Home Assistant local webhook
    - Local logging
    """

    def __init__(self, ha_webhook_url: str | None = None):
        load_dotenv_if_present()
        self.ha_webhook_url = ha_webhook_url or os.environ.get("TOKENWATCH_HA_WEBHOOK_URL")

    async def send_alert(self, alert: AlertPayload) -> None:
        logger.warning(f"[ALERT][{alert.severity}] {alert.title}: {alert.message}")

        # Post to Home Assistant local webhook if configured
        webhook_url = self.ha_webhook_url or os.environ.get("TOKENWATCH_HA_WEBHOOK_URL")
        if webhook_url:
            try:
                async with httpx.AsyncClient(timeout=4.0) as client:
                    await client.post(
                        webhook_url,
                        json={
                            "title": alert.title,
                            "message": alert.message,
                            "alert_type": alert.alert_type,
                            "severity": alert.severity,
                            "details": alert.details or {},
                        },
                    )
            except httpx.HTTPError as e:
                logger.error(f"Failed to post alert to Home Assistant: {e}")


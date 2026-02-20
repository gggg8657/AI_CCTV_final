"""
웹훅 알림 채널
"""

import json
import logging
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

from .base import NotificationChannel, NotificationPayload

logger = logging.getLogger(__name__)


class WebhookChannel(NotificationChannel):
    """HTTP POST 웹훅 알림 (Slack, Discord, Teams, custom 등)"""

    def __init__(self, url: str, timeout: float = 5.0, headers: Optional[dict] = None):
        self._url = url
        self._timeout = timeout
        self._headers = headers or {}

    @property
    def name(self) -> str:
        return "webhook"

    def send(self, payload: NotificationPayload) -> bool:
        body = json.dumps({
            "text": payload.title,
            "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": payload.body}}],
            **payload.to_dict(),
        }).encode("utf-8")

        req = Request(self._url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        for k, v in self._headers.items():
            req.add_header(k, v)

        try:
            with urlopen(req, timeout=self._timeout) as resp:
                ok = 200 <= resp.status < 300
                if ok:
                    logger.info("Webhook sent: %s", payload.title)
                else:
                    logger.warning("Webhook returned %d", resp.status)
                return ok
        except URLError as exc:
            logger.error("Webhook failed: %s", exc)
            return False

    @property
    def is_available(self) -> bool:
        return bool(self._url)

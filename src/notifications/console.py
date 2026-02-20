"""
콘솔 알림 채널 — 개발/디버깅용
"""

import logging
from .base import NotificationChannel, NotificationPayload

logger = logging.getLogger(__name__)


class ConsoleChannel(NotificationChannel):
    """로그로 알림 출력 — 항상 사용 가능"""

    @property
    def name(self) -> str:
        return "console"

    def send(self, payload: NotificationPayload) -> bool:
        logger.warning("ALERT %s\n%s", payload.title, payload.body)
        return True

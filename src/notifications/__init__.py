"""
알림 시스템
==========

이상 감지 시 웹훅/이메일/콘솔 채널로 알림 발송.
NotificationEngine이 규칙 기반으로 필터링 및 중복 방지 후 발송.
"""

from .base import NotificationChannel, NotificationPayload
from .webhook import WebhookChannel
from .email import EmailChannel
from .console import ConsoleChannel
from .engine import NotificationEngine

__all__ = [
    "NotificationChannel",
    "NotificationPayload",
    "WebhookChannel",
    "EmailChannel",
    "ConsoleChannel",
    "NotificationEngine",
]

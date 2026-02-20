"""
이메일 알림 채널 (SMTP)
"""

import logging
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional

from .base import NotificationChannel, NotificationPayload

logger = logging.getLogger(__name__)


class EmailChannel(NotificationChannel):
    """SMTP 이메일 알림 — Gmail/Naver/기타 SMTP 지원"""

    def __init__(
        self,
        smtp_host: str = "",
        smtp_port: int = 587,
        username: str = "",
        password: str = "",
        from_addr: str = "",
        to_addrs: Optional[List[str]] = None,
        use_tls: bool = True,
    ):
        self._host = smtp_host or os.getenv("SMTP_HOST", "")
        self._port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self._username = username or os.getenv("SMTP_USERNAME", "")
        self._password = password or os.getenv("SMTP_PASSWORD", "")
        self._from = from_addr or os.getenv("SMTP_FROM", self._username)
        self._to = to_addrs or os.getenv("SMTP_TO", "").split(",")
        self._use_tls = use_tls

    @property
    def name(self) -> str:
        return "email"

    @property
    def is_available(self) -> bool:
        return bool(self._host and self._username and self._to)

    def send(self, payload: NotificationPayload) -> bool:
        if not self.is_available:
            logger.warning("Email channel not configured — skipping")
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = payload.title
        msg["From"] = self._from
        msg["To"] = ", ".join(self._to)

        html = f"""<html><body>
<h2>{payload.title}</h2>
<pre>{payload.body}</pre>
<hr>
<small>AI CCTV Alert System</small>
</body></html>"""

        msg.attach(MIMEText(payload.body, "plain"))
        msg.attach(MIMEText(html, "html"))

        try:
            with smtplib.SMTP(self._host, self._port, timeout=10) as server:
                if self._use_tls:
                    server.starttls()
                server.login(self._username, self._password)
                server.sendmail(self._from, self._to, msg.as_string())
            logger.info("Email sent to %s: %s", self._to, payload.title)
            return True
        except Exception as exc:
            logger.error("Email send failed: %s", exc)
            return False

"""
tools/email_sender.py
─────────────────────
Send emails via SMTP with optional attachments, CC/BCC, and thread support.
Includes CSV logging for sent emails.
"""

import csv
import os
import smtplib
import time
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

LOG_FILE = "sent_emails_log.csv"
LOG_HEADERS = ["timestamp", "company", "recipient_email", "subject", "status", "error"]


# ── Helper ─────────────────────────────────────────────────

def _wrap_html(body: str) -> str:
    """Wrap email body in a styled HTML div."""
    html = body.replace("\n", "<br>")
    return (
        "<div style='font-family: Arial, sans-serif; "
        f"font-size: 14px; line-height: 1.6;'>{html}</div>"
    )


def _attach_files(msg: MIMEMultipart, paths: list[str] | None) -> None:
    """Attach files to a MIME message."""
    if not paths:
        return
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={os.path.basename(path)}",
        )
        msg.attach(part)


def _collect_recipients(to: str, cc: str = "", bcc: str = "") -> list[str]:
    """Build a flat list of all recipient addresses."""
    recipients = [to]
    if cc:
        recipients += [a.strip() for a in cc.split(",") if a.strip()]
    if bcc:
        recipients += [a.strip() for a in bcc.split(",") if a.strip()]
    return recipients


def _smtp_send(msg: MIMEMultipart, recipients: list[str],
               smtp_server: str, smtp_port: int,
               sender_email: str, sender_password: str) -> dict:
    """Connect to SMTP, authenticate, and send."""
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipients, msg.as_string())
        return {"success": True, "error": None}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Public API ─────────────────────────────────────────────

def send_email(
    to_email: str,
    subject: str,
    body: str,
    attachment_path: str | None,
    smtp_server: str,
    smtp_port: int,
    sender_email: str,
    sender_password: str,
) -> dict:
    """
    Send an outreach email with an optional PDF attachment.

    Returns: {"success": bool, "error": str | None}
    """
    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(_wrap_html(body), "html"))

        if attachment_path:
            _attach_files(msg, [attachment_path])

        return _smtp_send(msg, [to_email], smtp_server, smtp_port,
                          sender_email, sender_password)
    except Exception as e:
        return {"success": False, "error": str(e)}


def send_reply(
    to_email: str,
    subject: str,
    body: str,
    message_id: str,
    in_reply_to: str,
    smtp_server: str,
    smtp_port: int,
    sender_email: str,
    sender_password: str,
    cc: str = "",
    bcc: str = "",
    attachment_paths: list[str] | None = None,
) -> dict:
    """
    Send a reply email that preserves the conversation thread.

    Uses In-Reply-To and References headers so the reply stays
    in the same thread in Gmail / Outlook / etc.

    Returns: {"success": bool, "error": str | None}
    """
    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_email
        if cc:
            msg["Cc"] = cc

        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"
        msg["Subject"] = subject

        # Thread headers
        if in_reply_to:
            msg["In-Reply-To"] = in_reply_to
        if message_id:
            msg["References"] = message_id

        msg.attach(MIMEText(_wrap_html(body), "html"))
        _attach_files(msg, attachment_paths)

        recipients = _collect_recipients(to_email, cc, bcc)
        return _smtp_send(msg, recipients, smtp_server, smtp_port,
                          sender_email, sender_password)
    except Exception as e:
        return {"success": False, "error": str(e)}


def send_composed_email(
    to_email: str,
    subject: str,
    body: str,
    smtp_server: str,
    smtp_port: int,
    sender_email: str,
    sender_password: str,
    cc: str = "",
    bcc: str = "",
    attachment_paths: list[str] | None = None,
) -> dict:
    """
    Send a freshly composed email with optional CC, BCC, and attachments.

    Returns: {"success": bool, "error": str | None}
    """
    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_email
        if cc:
            msg["Cc"] = cc
        msg["Subject"] = subject

        msg.attach(MIMEText(_wrap_html(body), "html"))
        _attach_files(msg, attachment_paths)

        recipients = _collect_recipients(to_email, cc, bcc)
        return _smtp_send(msg, recipients, smtp_server, smtp_port,
                          sender_email, sender_password)
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Logging ────────────────────────────────────────────────

def log_sent_email(
    company: str,
    recipient_email: str,
    subject: str,
    success: bool,
    error: str | None = None,
    log_dir: str = ".",
) -> None:
    """Append a row to the CSV log file."""
    log_path = os.path.join(log_dir, LOG_FILE)
    file_exists = os.path.isfile(log_path)

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "company": company,
            "recipient_email": recipient_email,
            "subject": subject,
            "status": "SENT" if success else "FAILED",
            "error": error or "",
        })


def is_already_sent(recipient_email: str, log_dir: str = ".") -> bool:
    """Check the CSV log for a previous successful send to this address."""
    log_path = os.path.join(log_dir, LOG_FILE)
    if not os.path.isfile(log_path):
        return False
    with open(log_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("recipient_email") == recipient_email and row.get("status") == "SENT":
                return True
    return False


def rate_limited_sleep(seconds: float = 2.0) -> None:
    """Simple delay between consecutive sends to avoid throttling."""
    time.sleep(seconds)

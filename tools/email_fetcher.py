"""
tools/email_fetcher.py
──────────────────────
Fetch emails from an IMAP mailbox (Gmail, Outlook, etc.).
Returns structured email data for the Reply Agent.
"""

import imaplib
import email as email_lib
import re
from email.header import decode_header
from typing import Optional


def _decode_header_value(value: str) -> str:
    """Decode an RFC-2047 encoded header into a plain string."""
    if not value:
        return ""
    parts = decode_header(value)
    decoded = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return " ".join(decoded)


def _extract_body(msg: email_lib.message.Message) -> str:
    """
    Walk a MIME message and return the best plain-text body.
    Prefers text/plain; falls back to text/html with tags stripped.
    """
    plain_body = ""
    html_body = ""

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))
            if "attachment" in disposition:
                continue
            try:
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                charset = part.get_content_charset() or "utf-8"
                text = payload.decode(charset, errors="replace")
            except Exception:
                continue

            if content_type == "text/plain":
                plain_body = text
            elif content_type == "text/html":
                html_body = text
    else:
        try:
            payload = msg.get_payload(decode=True)
            charset = msg.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace") if payload else ""
        except Exception:
            text = ""

        if msg.get_content_type() == "text/plain":
            plain_body = text
        else:
            html_body = text

    if plain_body:
        return plain_body.strip()

    # Strip HTML tags for a rough plain-text version
    if html_body:
        clean = re.sub(r"<style[^>]*>.*?</style>", "", html_body, flags=re.S)
        clean = re.sub(r"<script[^>]*>.*?</script>", "", clean, flags=re.S)
        clean = re.sub(r"<[^>]+>", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    return ""


def fetch_emails(
    imap_server: str,
    email_address: str,
    email_password: str,
    imap_port: int = 993,
    folder: str = "INBOX",
    filter_mode: str = "unread",
    label: Optional[str] = None,
    max_results: int = 20,
) -> list[dict]:
    """
    Connect to an IMAP server and fetch emails.

    Parameters
    ----------
    imap_server  : e.g. "imap.gmail.com"
    email_address: Login email
    email_password: App password (NOT regular password for Gmail)
    imap_port    : Default 993 (SSL)
    folder       : Mailbox folder to read from
    filter_mode  : "unread" | "label" | "all"
    label        : Gmail label (used when filter_mode == "label")
    max_results  : Maximum emails to return

    Returns
    -------
    list[dict] — Each dict has keys:
        uid, sender, sender_email, subject, body, timestamp,
        thread_id, message_id, in_reply_to, is_read
    """
    emails: list[dict] = []

    try:
        mail = imaplib.IMAP4_SSL(imap_server, imap_port)
        mail.login(email_address, email_password)

        # Select folder
        if filter_mode == "label" and label:
            status, _ = mail.select(f'"{label}"')
        else:
            status, _ = mail.select(folder)

        if status != "OK":
            return emails

        # Build search criteria
        if filter_mode == "unread":
            criteria = "UNSEEN"
        else:
            criteria = "ALL"

        status, msg_ids = mail.search(None, criteria)
        if status != "OK" or not msg_ids[0]:
            mail.logout()
            return emails

        id_list = msg_ids[0].split()
        id_list = id_list[-max_results:]  # most recent N

        for uid in id_list:
            status, msg_data = mail.fetch(uid, "(RFC822)")
            if status != "OK":
                continue

            raw_email = msg_data[0][1]
            msg = email_lib.message_from_bytes(raw_email)

            # Decode headers
            subject = _decode_header_value(msg.get("Subject", ""))
            from_header = _decode_header_value(msg.get("From", ""))
            date_header = msg.get("Date", "")
            message_id = msg.get("Message-ID", "")
            in_reply_to = msg.get("In-Reply-To", "")
            references = msg.get("References", "")

            # Parse sender name and email
            sender_name = from_header
            sender_email_addr = from_header
            if "<" in from_header and ">" in from_header:
                parts = from_header.rsplit("<", 1)
                sender_name = parts[0].strip().strip('"')
                sender_email_addr = parts[1].rstrip(">").strip()

            # Parse date
            try:
                date_obj = email_lib.utils.parsedate_to_datetime(date_header)
                timestamp = date_obj.isoformat()
            except Exception:
                timestamp = date_header

            # Thread ID
            if references:
                thread_id = references.split()[0].strip()
            elif in_reply_to:
                thread_id = in_reply_to.strip()
            else:
                thread_id = message_id

            body = _extract_body(msg)

            emails.append({
                "uid": uid.decode() if isinstance(uid, bytes) else str(uid),
                "sender": sender_name,
                "sender_email": sender_email_addr,
                "subject": subject,
                "body": body[:3000],
                "timestamp": timestamp,
                "thread_id": thread_id,
                "message_id": message_id,
                "in_reply_to": in_reply_to,
                "is_read": filter_mode != "unread",
            })

        mail.logout()

    except imaplib.IMAP4.error as e:
        raise ConnectionError(f"IMAP error: {e}")
    except Exception as e:
        raise ConnectionError(f"Failed to fetch emails: {e}")

    emails.reverse()  # newest first
    return emails


def mark_as_read(
    imap_server: str,
    email_address: str,
    email_password: str,
    uid: str,
    imap_port: int = 993,
    folder: str = "INBOX",
) -> bool:
    """Mark a specific email as read (Seen) on the IMAP server."""
    try:
        mail = imaplib.IMAP4_SSL(imap_server, imap_port)
        mail.login(email_address, email_password)
        mail.select(folder)
        mail.store(uid.encode(), "+FLAGS", "\\Seen")
        mail.logout()
        return True
    except Exception:
        return False

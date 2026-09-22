"""
agents/reply_agent.py
─────────────────────
System 2 – AI Inbox Reply Agent

Fetches unread emails from IMAP, classifies intent with LLM,
generates a professional reply, and sends after human approval.

LangGraph workflow:
    fetch_emails -> classify_email -> generate_reply -> human_review
        human_review ->
            "send"       -> send_reply -> END
            "regenerate" -> generate_reply  (loop)
            "discard"    -> END
"""

from __future__ import annotations
import json
from typing import TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from tools.email_fetcher import fetch_emails as _fetch_emails
from tools.email_sender import send_reply as _send_reply


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# State
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ReplyState(TypedDict, total=False):
    # Inbox
    fetched_emails: list[dict]
    selected_email: dict

    # Classification
    classification: dict          # {"intent": str, "requires_reply": bool}

    # Generated reply
    reply_subject: str
    reply_body: str

    # Human review
    approval_status: str          # "send" | "regenerate" | "discard"
    edited_reply: str
    edited_subject: str

    # Optional user instruction for tone/context
    user_instruction: str

    # Config
    groq_api_key: str
    smtp_server: str
    smtp_port: int
    imap_server: str
    sender_email: str
    sender_password: str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core functions (simple interfaces — used by Streamlit UI)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def do_classify(sender: str, sender_email: str, subject: str,
                body: str, groq_api_key: str) -> dict:
    """
    Classify an email's intent using LLM.

    Returns: {"intent": str, "requires_reply": bool}

    Categories: job_reply, meeting, recruiter, question,
                newsletter, spam, other
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0,
        max_tokens=200,
    )

    system_prompt = (
        "You are an email classifier. Classify the email intent "
        "into exactly ONE of these categories:\n"
        "- job_reply    (response to a job application)\n"
        "- meeting      (meeting request or scheduling)\n"
        "- recruiter    (recruiter reaching out)\n"
        "- question     (someone asking a question)\n"
        "- newsletter   (newsletter or promotional)\n"
        "- spam         (spam or irrelevant)\n"
        "- other        (anything else)\n\n"
        "Also decide if the email requires a reply (true/false).\n\n"
        "Respond ONLY with valid JSON:\n"
        '{"intent": "<category>", "requires_reply": true/false}'
    )

    user_prompt = (
        f"From: {sender} <{sender_email}>\n"
        f"Subject: {subject}\n\n"
        f"Body:\n{body[:2000]}"
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    try:
        text = response.content.strip()
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        pass

    return {"intent": "other", "requires_reply": True}


def do_generate_reply(sender: str, sender_email: str, subject: str,
                      body: str, intent: str, groq_api_key: str,
                      user_instruction: str = "") -> dict:
    """
    Generate a professional email reply using LLM.

    Returns: {"body": str, "subject": str}
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.7,
        max_tokens=500,
    )

    system_prompt = (
        "You are a professional email reply assistant.\n\n"
        "Write a reply that is:\n"
        "- Professional and courteous\n"
        "- Concise (50\u2013150 words)\n"
        "- In HTML format using <b> for emphasis\n"
        "- No placeholders like [Your Name]\n"
        "- Professional sign-off"
    )

    user_prompt = (
        f"Email intent: {intent}\n\n"
        f"Original email:\n"
        f"From: {sender} <{sender_email}>\n"
        f"Subject: {subject}\n"
        f"Body:\n{body[:2000]}\n\n"
    )
    if user_instruction:
        user_prompt += f"User instruction: {user_instruction}\n\n"
    user_prompt += "Write a professional reply."

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    reply_body = response.content.strip()
    reply_subject = subject if subject.lower().startswith("re:") else f"Re: {subject}"

    return {"body": reply_body, "subject": reply_subject}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LangGraph nodes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fetch_emails_node(state: ReplyState) -> dict:
    """Node 1 — Fetch unread emails from IMAP inbox."""
    emails = _fetch_emails(
        imap_server=state.get("imap_server", "imap.gmail.com"),
        email_address=state["sender_email"],
        email_password=state["sender_password"],
        filter_mode="unread",
        max_results=20,
    )
    return {"fetched_emails": emails}


def classify_email_node(state: ReplyState) -> dict:
    """Node 2 — Classify the selected email's intent."""
    em = state["selected_email"]
    result = do_classify(
        sender=em.get("sender", ""),
        sender_email=em.get("sender_email", ""),
        subject=em.get("subject", ""),
        body=em.get("body", ""),
        groq_api_key=state["groq_api_key"],
    )
    return {"classification": result}


def generate_reply_node(state: ReplyState) -> dict:
    """Node 3 — Generate a professional reply."""
    em = state["selected_email"]
    classification = state.get("classification", {})

    result = do_generate_reply(
        sender=em.get("sender", ""),
        sender_email=em.get("sender_email", ""),
        subject=em.get("subject", ""),
        body=em.get("body", ""),
        intent=classification.get("intent", "other"),
        groq_api_key=state["groq_api_key"],
        user_instruction=state.get("user_instruction", ""),
    )

    return {
        "reply_body": result["body"],
        "reply_subject": result["subject"],
        "approval_status": "",
    }


def human_review_node(state: ReplyState) -> dict:
    """Node 4 — Interrupt point. Streamlit UI handles review."""
    return {}


def send_reply_node(state: ReplyState) -> dict:
    """Node 5 — Send the approved reply, preserving email thread."""
    em = state["selected_email"]

    body = state.get("edited_reply") or state["reply_body"]
    subject = state.get("edited_subject") or state["reply_subject"]

    result = _send_reply(
        to_email=em["sender_email"],
        subject=subject,
        body=body,
        message_id=em.get("message_id", ""),
        in_reply_to=em.get("message_id", ""),
        smtp_server=state["smtp_server"],
        smtp_port=state["smtp_port"],
        sender_email=state["sender_email"],
        sender_password=state["sender_password"],
    )
    return {"send_result": result}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Routing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def route_after_review(state: ReplyState) -> str:
    """Decide next step based on human review."""
    status = state.get("approval_status", "")
    if status == "send":
        return "send_reply"
    elif status == "regenerate":
        return "generate_reply"
    return END


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Graph builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_reply_graph():
    """
    Build and compile the Reply Agent graph.

    Graph:
        fetch_emails -> classify_email -> generate_reply -> human_review
            -> send_reply     (if approved)
            -> generate_reply (if regenerate)
            -> END            (if discard)

    Returns: (compiled_graph, checkpointer)
    """
    graph = StateGraph(ReplyState)

    graph.add_node("fetch_emails", fetch_emails_node)
    graph.add_node("classify_email", classify_email_node)
    graph.add_node("generate_reply", generate_reply_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("send_reply", send_reply_node)

    graph.set_entry_point("fetch_emails")
    graph.add_edge("fetch_emails", "classify_email")
    graph.add_edge("classify_email", "generate_reply")
    graph.add_edge("generate_reply", "human_review")
    graph.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "send_reply": "send_reply",
            "generate_reply": "generate_reply",
            END: END,
        },
    )
    graph.add_edge("send_reply", END)

    checkpointer = MemorySaver()
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],
    )
    return compiled, checkpointer

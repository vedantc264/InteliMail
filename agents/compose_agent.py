"""
agents/compose_agent.py
───────────────────────
System 3 – Smart Email Composer Agent

Takes a brief topic and desired tone from the user, generates
a complete professional email with subject line, and sends it
after human approval.

LangGraph workflow:
    generate_email -> human_review
        human_review ->
            "send"       -> send_email -> END
            "regenerate" -> generate_email  (loop)
            "discard"    -> END
"""

from __future__ import annotations
from typing import TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from tools.email_sender import send_composed_email as _send_composed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# State
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ComposeState(TypedDict, total=False):
    # User input
    recipient_email: str
    brief_topic: str
    tone: str                     # professional | friendly | formal | casual | persuasive
    cc: str
    bcc: str
    attachment_paths: list[str]

    # Generated
    composed_subject: str
    composed_body: str

    # Human review
    approval_status: str          # "send" | "regenerate" | "discard"
    edited_subject: str
    edited_body: str

    # Config
    groq_api_key: str
    smtp_server: str
    smtp_port: int
    sender_email: str
    sender_password: str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core function (simple interface — used by Streamlit UI)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def do_generate_email(topic: str, recipient: str, tone: str,
                      groq_api_key: str) -> dict:
    """
    Generate a complete professional email from a brief topic.

    Returns: {"body": str, "subject": str}
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.7,
        max_tokens=600,
    )

    system_prompt = (
        "You are a professional email writing assistant.\n\n"
        "Given a brief topic and tone, write a complete email.\n\n"
        "Rules:\n"
        "- Write ONLY the email body (no subject in body)\n"
        "- Use HTML with <b> for emphasis\n"
        "- Match the requested tone\n"
        "- 80\u2013200 words\n"
        "- No placeholders\n"
        "- Appropriate sign-off\n\n"
        "Also generate a concise subject line.\n\n"
        "Return in this exact format:\n"
        "SUBJECT: <subject line>\n"
        "BODY:\n<email body>"
    )

    user_prompt = (
        f"Write a {tone} email.\n\n"
        f"Topic:\n{topic}\n\n"
        f"Recipient: {recipient}\n"
        f"Tone: {tone}\n\n"
        "Return SUBJECT: and BODY: as specified."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    text = response.content.strip()

    # Parse SUBJECT and BODY from LLM response
    subject = ""
    body = text

    if "SUBJECT:" in text and "BODY:" in text:
        parts = text.split("BODY:", 1)
        body = parts[1].strip() if len(parts) > 1 else text
        subj_parts = parts[0].split("SUBJECT:", 1)
        if len(subj_parts) > 1:
            subject = subj_parts[1].strip().split("\n")[0].strip()
    elif "SUBJECT:" in text:
        for i, line in enumerate(text.split("\n")):
            if line.strip().startswith("SUBJECT:"):
                subject = line.replace("SUBJECT:", "").strip()
                body = "\n".join(text.split("\n")[i + 1:]).strip()
                break

    if not subject:
        subject = f"Regarding: {topic[:50]}"

    return {"body": body, "subject": subject}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LangGraph nodes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_email_node(state: ComposeState) -> dict:
    """Node 1 — Generate email from topic and tone."""
    result = do_generate_email(
        topic=state["brief_topic"],
        recipient=state.get("recipient_email", ""),
        tone=state.get("tone", "professional"),
        groq_api_key=state["groq_api_key"],
    )
    return {
        "composed_subject": result["subject"],
        "composed_body": result["body"],
        "approval_status": "",
    }


def human_review_node(state: ComposeState) -> dict:
    """Node 2 — Interrupt point. Streamlit UI handles review."""
    return {}


def send_email_node(state: ComposeState) -> dict:
    """Node 3 — Send the composed email via SMTP."""
    body = state.get("edited_body") or state["composed_body"]
    subject = state.get("edited_subject") or state["composed_subject"]

    result = _send_composed(
        to_email=state["recipient_email"],
        subject=subject,
        body=body,
        smtp_server=state["smtp_server"],
        smtp_port=state["smtp_port"],
        sender_email=state["sender_email"],
        sender_password=state["sender_password"],
        cc=state.get("cc", ""),
        bcc=state.get("bcc", ""),
        attachment_paths=state.get("attachment_paths"),
    )
    return {"send_result": result}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Routing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def route_after_review(state: ComposeState) -> str:
    """Decide next step based on human review."""
    status = state.get("approval_status", "")
    if status == "send":
        return "send_email"
    elif status == "regenerate":
        return "generate_email"
    return END


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Graph builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_compose_graph():
    """
    Build and compile the Compose Agent graph.

    Graph:
        generate_email -> human_review
            -> send_email     (if approved)
            -> generate_email (if regenerate)
            -> END            (if discard)

    Returns: (compiled_graph, checkpointer)
    """
    graph = StateGraph(ComposeState)

    graph.add_node("generate_email", generate_email_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("send_email", send_email_node)

    graph.set_entry_point("generate_email")
    graph.add_edge("generate_email", "human_review")
    graph.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "send_email": "send_email",
            "generate_email": "generate_email",
            END: END,
        },
    )
    graph.add_edge("send_email", END)

    checkpointer = MemorySaver()
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],
    )
    return compiled, checkpointer

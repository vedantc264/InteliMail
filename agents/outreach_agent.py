"""
agents/outreach_agent.py
────────────────────────
System 1 – HR Outreach Automation Agent

Reads a spreadsheet of company names and HR emails, researches each
company online, generates a personalised internship outreach email,
and sends it after human approval.

LangGraph workflow:
    load_data -> research_company -> generate_email -> human_review
        human_review ->
            "approve"    -> send_email -> END
            "regenerate" -> generate_email  (loop)
            "skip"       -> END
"""

from __future__ import annotations
from typing import TypedDict

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from tools.resume_parser import parse_resume
from tools.research_tool import research_company as _research_company
from tools.email_sender import (
    send_email as _send_email,
    log_sent_email,
    rate_limited_sleep,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# State
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class OutreachState(TypedDict, total=False):
    # Data (loaded once)
    companies: list[dict]
    resume_summary: str
    resume_path: str
    excel_path: str
    current_index: int

    # Per-company (changes each iteration)
    company_research: str
    generated_email: str
    email_subject: str

    # Human review
    approval_status: str        # "approve" | "regenerate" | "skip"
    edited_email: str
    edited_subject: str

    # Config
    groq_api_key: str
    tavily_api_key: str
    smtp_server: str
    smtp_port: int
    sender_email: str
    sender_password: str
    project_dir: str

    # Logging
    sent_log: list[dict]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core functions (simple interfaces — used by Streamlit UI)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def do_research(company_name: str, industry: str, location: str,
                tavily_api_key: str, groq_api_key: str) -> str:
    """Research a company using Tavily search + LLM summary."""
    return _research_company(
        company_name=company_name,
        industry=industry,
        location=location,
        tavily_api_key=tavily_api_key,
        groq_api_key=groq_api_key,
    )


def do_generate_email(company_name: str, company_research: str,
                      resume_summary: str, groq_api_key: str) -> dict:
    """
    Generate a personalised outreach email using LLM.

    Returns: {"body": str, "subject": str}
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.7,
        max_tokens=500,
    )

    system_prompt = (
        "You are a professional outreach assistant writing highly personalised "
        "internship emails in HTML format.\n\n"
        "Structure:\n"
        "PARAGRAPH 1: Introduce yourself \u2014 name (from resume), "
        "3rd year undergrad from PICT, Pune. "
        "One sentence about why THIS company interests you.\n\n"
        "PARAGRAPH 2: Skills and projects \u2014 full-stack (React, Node.js, etc.) "
        "AND AI/ML. List 2-3 specific projects. "
        "Mention you can help with both areas.\n\n"
        "PARAGRAPH 3: Short closing \u2014 resume attached, enthusiasm, "
        "professional sign-off.\n\n"
        "Rules:\n"
        "- Use <b> tags for key words (name, company, skills, projects)\n"
        "- 150\u2013200 words\n"
        "- No subject line in body\n"
        "- No placeholders like [Your Name]\n"
        "- Professional but enthusiastic tone\n"
        "- Best regards on a new line, name below it"
    )

    user_prompt = (
        f"Write an internship application email for Full Stack Intern "
        f"at {company_name}.\n\n"
        f"Company Research:\n{company_research}\n\n"
        f"Resume Summary:\n{resume_summary}\n\n"
        "Follow the structure and formatting rules above."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    body = response.content.strip()

    # Extract candidate name for subject line
    lines = resume_summary.split("\n")
    candidate = lines[0] if lines else "Candidate"
    for prefix in ["**Name:**", "Name:", "**", "##"]:
        candidate = candidate.replace(prefix, "")
    candidate = candidate.strip().strip("*").strip()
    if len(candidate) > 50:
        candidate = "Candidate"

    subject = f"Application for Full Stack Intern \u2013 {candidate}"
    return {"body": body, "subject": subject}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LangGraph nodes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_data_node(state: OutreachState) -> dict:
    """Node 1 — Read Excel + parse resume PDF."""
    df = pd.read_excel(state["excel_path"])
    df.columns = [c.strip() for c in df.columns]
    companies = df.to_dict(orient="records")

    resume_data = parse_resume(state["resume_path"], state["groq_api_key"])
    return {
        "companies": companies,
        "resume_summary": resume_data["summary"],
        "current_index": 0,
        "sent_log": [],
    }


def research_company_node(state: OutreachState) -> dict:
    """Node 2 — Research the current company online."""
    company = state["companies"][state["current_index"]]
    result = do_research(
        company_name=company.get("Company Name", company.get("company_name", "Unknown")),
        industry=company.get("Industry", company.get("industry", "")),
        location=company.get("Location", company.get("location", "")),
        tavily_api_key=state["tavily_api_key"],
        groq_api_key=state["groq_api_key"],
    )
    return {"company_research": result}


def generate_email_node(state: OutreachState) -> dict:
    """Node 3 — Draft a personalised outreach email."""
    company = state["companies"][state["current_index"]]
    company_name = company.get("Company Name", company.get("company_name", "Unknown"))

    result = do_generate_email(
        company_name=company_name,
        company_research=state["company_research"],
        resume_summary=state["resume_summary"],
        groq_api_key=state["groq_api_key"],
    )
    return {
        "generated_email": result["body"],
        "email_subject": result["subject"],
        "approval_status": "",
        "edited_email": "",
        "edited_subject": "",
    }


def human_review_node(state: OutreachState) -> dict:
    """Node 4 — Interrupt point. Streamlit UI handles approval."""
    return {}


def send_email_node(state: OutreachState) -> dict:
    """Node 5 — Send the approved email via SMTP and log result."""
    company = state["companies"][state["current_index"]]
    company_name = company.get("Company Name", company.get("company_name", "Unknown"))
    to_email = company.get("Email ID", company.get("email_id", company.get("Email", "")))

    body = state.get("edited_email") or state["generated_email"]
    subject = state.get("edited_subject") or state["email_subject"]
    project_dir = state.get("project_dir", ".")

    result = _send_email(
        to_email=to_email,
        subject=subject,
        body=body,
        attachment_path=state.get("resume_path"),
        smtp_server=state["smtp_server"],
        smtp_port=state["smtp_port"],
        sender_email=state["sender_email"],
        sender_password=state["sender_password"],
    )

    log_sent_email(
        company=company_name,
        recipient_email=to_email,
        subject=subject,
        success=result["success"],
        error=result.get("error"),
        log_dir=project_dir,
    )

    log_entry = {
        "company": company_name,
        "email": to_email,
        "status": "SENT" if result["success"] else "FAILED",
        "error": result.get("error", ""),
    }
    sent_log = list(state.get("sent_log", []))
    sent_log.append(log_entry)

    rate_limited_sleep(2.0)
    return {"sent_log": sent_log}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Routing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def route_after_review(state: OutreachState) -> str:
    """Decide next step based on human review."""
    status = state.get("approval_status", "")
    if status == "approve":
        return "send_email"
    elif status == "regenerate":
        return "generate_email"
    return END


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Graph builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_outreach_graph():
    """
    Build and compile the Outreach Agent graph.

    Graph:
        load_data -> research_company -> generate_email -> human_review
            -> send_email   (if approved)
            -> generate_email (if regenerate)
            -> END          (if skip)

    Returns: (compiled_graph, checkpointer)
    """
    graph = StateGraph(OutreachState)

    graph.add_node("load_data", load_data_node)
    graph.add_node("research_company", research_company_node)
    graph.add_node("generate_email", generate_email_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("send_email", send_email_node)

    graph.set_entry_point("load_data")
    graph.add_edge("load_data", "research_company")
    graph.add_edge("research_company", "generate_email")
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

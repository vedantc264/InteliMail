"""
app.py - Streamlit UI for Email Automation AI
──────────────────────────────────────────────
Three tabs: HR Outreach | Inbox Reply | Smart Compose
Run with:  streamlit run app.py
"""

import os
import re
import sys
import tempfile
import uuid

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure project root is on the path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ── Agent core functions (LLM logic lives in agents/) ─────
from agents.outreach_agent import do_research, do_generate_email
from agents.reply_agent import do_classify, do_generate_reply
from agents.compose_agent import do_generate_email as compose_generate

# ── Tool utilities ────────────────────────────────────────
from tools.email_sender import (
    send_email, send_reply, send_composed_email,
    log_sent_email, is_already_sent, rate_limited_sleep,
)
from tools.email_fetcher import fetch_emails, mark_as_read
from tools.resume_parser import parse_resume

load_dotenv(os.path.join(ROOT_DIR, ".env"))


# ══════════════════════════════════════════════════════════
# Page config & CSS
# ══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Email Automation AI",
    page_icon="\U0001f4e7",
    layout="wide",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*='st-'] { font-family: 'Inter', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}
.main-header h1 { color: #e94560; font-size: 2.2rem; font-weight: 700; margin: 0; }
.main-header p  { color: #a8b2d1; font-size: 1rem; margin: 0.4rem 0 0; }
.info-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155; border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;
}
.info-card h4 { color: #e94560; margin: 0 0 0.4rem; }
.info-card p  { color: #94a3b8; margin: 0; font-size: 0.9rem; }
.stButton > button { border-radius: 8px; font-weight: 600; transition: all 0.2s ease; }
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(233,69,96,0.3); }
section[data-testid='stSidebar'] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); }
section[data-testid='stSidebar'] .stMarkdown h2,
section[data-testid='stSidebar'] .stMarkdown h3 { color: #e94560; }
hr { border-color: #334155; }
.research-card {
    background: #1e293b; border: 1px solid #334155; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 1.5rem; font-size: 0.9rem;
    color: #cbd5e1; line-height: 1.6;
}
.research-card h4 { color: #e94560; margin: 0 0 0.5rem; }
.email-card {
    background: #1e293b; border: 1px solid #334155; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 0.8rem; transition: border-color 0.2s;
}
.email-card:hover { border-color: #e94560; }
.email-card h5 { color: #e2e8f0; margin: 0 0 0.3rem; font-size: 0.95rem; }
.email-card .sender { color: #e94560; font-weight: 600; font-size: 0.85rem; }
.email-card .preview { color: #94a3b8; font-size: 0.82rem; margin-top: 0.3rem; }
.email-card .timestamp { color: #64748b; font-size: 0.75rem; float: right; }
.intent-badge {
    display: inline-block; padding: 0.15rem 0.6rem; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; margin-left: 0.5rem;
}
.intent-job_reply  { background: #065f46; color: #6ee7b7; }
.intent-meeting    { background: #1e3a5f; color: #7dd3fc; }
.intent-recruiter  { background: #581c87; color: #d8b4fe; }
.intent-question   { background: #713f12; color: #fde68a; }
.intent-newsletter { background: #374151; color: #9ca3af; }
.intent-spam       { background: #7f1d1d; color: #fca5a5; }
.intent-other      { background: #334155; color: #cbd5e1; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════

def _init_session():
    defaults = {
        # Outreach
        "companies": [],
        "resume_summary": "",
        "resume_path_tmp": "",
        "excel_path_tmp": "",
        "data_loaded": False,
        "current_company_idx": 0,
        "generated_email": "",
        "email_subject": "",
        "company_research": "",
        "email_generating": False,
        "email_ready": False,
        "sent_log": [],
        # Reply
        "inbox_emails": [],
        "inbox_loaded": False,
        "inbox_selected_idx": None,
        "inbox_classification": None,
        "inbox_reply_body": "",
        "inbox_reply_subject": "",
        "inbox_reply_ready": False,
        "inbox_generating": False,
        # Compose
        "compose_subject": "",
        "compose_body": "",
        "compose_ready": False,
        "compose_generating": False,
        "compose_attachments": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


# ══════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════

st.markdown(
    '<div class="main-header">'
    "<h1>\U0001f4e7 Email Automation AI</h1>"
    "<p>HR Outreach \u2022 Inbox Reply \u2022 Smart Compose "
    "\u2014 powered by LangGraph + Groq (free)</p>"
    "</div>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════
# Credentials from .env
# ══════════════════════════════════════════════════════════

groq_api_key = os.getenv("GROQ_API_KEY", "")
tavily_api_key = os.getenv("TAVILY_API_KEY", "")
sender_email = os.getenv("EMAIL_ADDRESS", "")
sender_password = os.getenv("EMAIL_PASSWORD", "")
smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
smtp_port = int(os.getenv("SMTP_PORT", "587"))
imap_server = os.getenv("IMAP_SERVER", "imap.gmail.com")

with st.sidebar:
    st.markdown("## \U0001f4e7 Email Automation AI")
    st.markdown("---")
    st.markdown(
        '<div class="info-card"><h4>✅ Configured</h4>'
        "<p>Credentials loaded from <b>.env</b> file.</p></div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════

def _save_uploaded(uploaded_file) -> str:
    """Save an uploaded file to a temp dir and return the path."""
    tmp = tempfile.mkdtemp()
    dest = os.path.join(tmp, uploaded_file.name)
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


# ══════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "1\ufe0f\u20e3 HR Outreach",
    "2\ufe0f\u20e3 Inbox Reply",
    "3\ufe0f\u20e3 Compose Email",
])


# ──────────────────────────────────────────────────────────
# TAB 1 — HR Outreach Agent
# ──────────────────────────────────────────────────────────

with tab1:
    st.markdown("## \U0001f4e4 HR Outreach Agent")
    st.caption(
        "Upload resume + Excel \u2192 AI researches each company "
        "\u2192 generates personalised emails"
    )

    # Upload
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        resume_file = st.file_uploader("Resume PDF", type=["pdf"], key="resume_up")
    with col_up2:
        excel_file = st.file_uploader(
            "HR Contacts Excel", type=["xlsx", "xls"], key="excel_up"
        )

    # Load button
    st.markdown("### 1\ufe0f\u20e3 Load Data")
    col_l1, col_l2 = st.columns([3, 1])
    with col_l1:
        if resume_file and excel_file and groq_api_key:
            st.success("\u2705 Resume and Excel uploaded. API key provided.")
        else:
            miss = []
            if not resume_file:
                miss.append("Resume PDF")
            if not excel_file:
                miss.append("Excel file")
            if not groq_api_key:
                miss.append("Groq API Key")
            st.warning(f"\u26a0\ufe0f Missing: {', '.join(miss)}")

    with col_l2:
        load_btn = st.button(
            "\U0001f680 Load Data",
            disabled=not (resume_file and excel_file and groq_api_key),
            use_container_width=True,
        )

    if load_btn:
        with st.spinner("\U0001f4d6 Parsing resume & reading Excel \u2026"):
            resume_tmp = _save_uploaded(resume_file)
            excel_tmp = _save_uploaded(excel_file)
            st.session_state["resume_path_tmp"] = resume_tmp
            st.session_state["excel_path_tmp"] = excel_tmp

            resume_data = parse_resume(resume_tmp, groq_api_key)
            df = pd.read_excel(excel_tmp)
            df.columns = [c.strip() for c in df.columns]
            companies = df.to_dict(orient="records")

            st.session_state["companies"] = companies
            st.session_state["resume_summary"] = resume_data["summary"]
            st.session_state["data_loaded"] = True
            st.session_state["sent_log"] = []
            st.session_state["current_company_idx"] = 0
            st.session_state["generated_email"] = ""
            st.session_state["email_subject"] = ""
            st.session_state["company_research"] = ""
            st.session_state["email_ready"] = False

        st.success(f"\u2705 Loaded **{len(companies)}** companies & parsed resume.")
        st.rerun()

    # Resume summary
    if st.session_state["data_loaded"] and st.session_state["resume_summary"]:
        with st.expander("\U0001f4dd Resume Summary", expanded=False):
            st.markdown(st.session_state["resume_summary"])

    # Company selector & generate
    if st.session_state["data_loaded"]:
        st.markdown("---")
        st.markdown("### 2\ufe0f\u20e3 Select Company & Generate Email")

        companies = st.session_state["companies"]

        def _company_label(c):
            name = c.get("Company Name", c.get("company_name", "Unknown"))
            email = c.get("Email ID", c.get("email_id", c.get("Email", "")))
            return f"{name}  \u2022  {email}"

        labels = [_company_label(c) for c in companies]
        selected_label = st.selectbox(
            "Choose a company",
            labels,
            index=st.session_state["current_company_idx"],
        )
        selected_idx = labels.index(selected_label)
        st.session_state["current_company_idx"] = selected_idx
        selected_company = companies[selected_idx]

        cname = selected_company.get(
            "Company Name", selected_company.get("company_name", "")
        )
        cindustry = selected_company.get(
            "Industry", selected_company.get("industry", "")
        )
        clocation = selected_company.get(
            "Location", selected_company.get("location", "")
        )
        cemail = selected_company.get(
            "Email ID",
            selected_company.get("email_id", selected_company.get("Email", "")),
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("\U0001f3e2 Company", cname)
        c2.metric("\U0001f3ed Industry", cindustry or "\u2014")
        c3.metric("\U0001f4cd Location", clocation or "\u2014")
        c4.metric("\U0001f4e7 Email", cemail or "\u2014")

        already_sent = is_already_sent(cemail, log_dir=ROOT_DIR)
        if already_sent:
            st.info("\u2139\ufe0f An email has already been sent to this address.")

        gen_btn = st.button(
            "\U0001f916 Research & Generate Email",
            use_container_width=True,
            disabled=st.session_state.get("email_generating", False),
        )

        if gen_btn:
            st.session_state["email_generating"] = True
            st.session_state["email_ready"] = False

            with st.status("Working \u2026", expanded=True) as status:
                st.write("\U0001f50d Researching company \u2026")
                company_research = do_research(
                    company_name=cname,
                    industry=cindustry,
                    location=clocation,
                    tavily_api_key=tavily_api_key,
                    groq_api_key=groq_api_key,
                )
                st.session_state["company_research"] = company_research
                st.write("\u2705 Research complete.")

                st.write("\u270d\ufe0f Drafting email \u2026")
                result = do_generate_email(
                    company_name=cname,
                    company_research=company_research,
                    resume_summary=st.session_state["resume_summary"],
                    groq_api_key=groq_api_key,
                )

                st.session_state["generated_email"] = result["body"]
                st.session_state["email_subject"] = result["subject"]
                st.session_state["email_ready"] = True
                st.session_state["email_generating"] = False
                status.update(label="\u2705 Email generated!", state="complete")
            st.rerun()

    # Review & Approve
    if st.session_state.get("email_ready"):
        st.markdown("---")
        st.markdown("### 3\ufe0f\u20e3 Review & Approve Email")

        if st.session_state.get("company_research"):
            st.markdown(
                '<div class="research-card">'
                "<h4>\U0001f50d Company Research</h4>"
                f'{st.session_state["company_research"]}</div>',
                unsafe_allow_html=True,
            )

        edited_subject = st.text_input(
            "\U0001f4cc Subject Line",
            value=st.session_state["email_subject"],
            key="outreach_subj",
        )
        edited_email = st.text_area(
            "\U0001f4dd Email Body",
            value=st.session_state["generated_email"],
            height=300,
            key="outreach_body",
        )

        plain = re.sub(r"<[^>]+>", "", edited_email)
        wc = len(plain.split())
        marker = "\u2705" if 150 <= wc <= 200 else "\u26a0\ufe0f (target: 150\u2013200)"
        st.caption(f"Word count: **{wc}** {marker}")

        st.markdown("#### \U0001f441\ufe0f Preview")
        preview = edited_email.replace("\n", "<br>")
        st.markdown(
            f'<div style="background:#f8f9fa;color:#1a1a2e;padding:1.2rem 1.5rem;'
            f'border-radius:10px;font-family:Arial;font-size:14px;line-height:1.7;'
            f'border:1px solid #dee2e6;">{preview}</div>',
            unsafe_allow_html=True,
        )

        ca, cb, cc_ = st.columns(3)
        with ca:
            approve_btn = st.button(
                "\u2705 Approve & Send", use_container_width=True, key="out_approve"
            )
        with cb:
            regen_btn = st.button(
                "\U0001f504 Regenerate", use_container_width=True, key="out_regen"
            )
        with cc_:
            skip_btn = st.button(
                "\u23ed\ufe0f Skip", use_container_width=True, key="out_skip"
            )

        if approve_btn:
            if not sender_email or not sender_password:
                st.error("\u274c Enter email credentials in sidebar.")
            else:
                companies = st.session_state["companies"]
                idx = st.session_state["current_company_idx"]
                company = companies[idx]
                to_email = company.get(
                    "Email ID", company.get("email_id", company.get("Email", ""))
                )
                cname_s = company.get(
                    "Company Name", company.get("company_name", "Unknown")
                )

                with st.spinner("\U0001f4e4 Sending \u2026"):
                    result = send_email(
                        to_email=to_email,
                        subject=edited_subject,
                        body=edited_email,
                        attachment_path=st.session_state.get("resume_path_tmp"),
                        smtp_server=smtp_server,
                        smtp_port=int(smtp_port),
                        sender_email=sender_email,
                        sender_password=sender_password,
                    )
                    log_sent_email(
                        company=cname_s,
                        recipient_email=to_email,
                        subject=edited_subject,
                        success=result["success"],
                        error=result.get("error"),
                        log_dir=ROOT_DIR,
                    )
                    st.session_state["sent_log"].append({
                        "company": cname_s,
                        "email": to_email,
                        "status": "SENT" if result["success"] else "FAILED",
                        "error": result.get("error", ""),
                    })

                if result["success"]:
                    st.success(f"\u2705 Sent to **{to_email}**!")
                    st.balloons()
                else:
                    st.error(f"\u274c Failed: {result['error']}")
                st.session_state["email_ready"] = False
                st.session_state["generated_email"] = ""
                st.session_state["email_subject"] = ""
                rate_limited_sleep(1.0)

        if regen_btn:
            st.session_state["email_ready"] = False
            st.session_state["generated_email"] = ""
            st.session_state["email_subject"] = ""
            st.rerun()

        if skip_btn:
            st.session_state["email_ready"] = False
            st.session_state["generated_email"] = ""
            st.session_state["email_subject"] = ""
            st.info("\u23ed\ufe0f Skipped.")

    # Sent log
    if st.session_state["sent_log"]:
        st.markdown("---")
        st.markdown("### \U0001f4ca Sent Email Log")
        st.dataframe(
            pd.DataFrame(st.session_state["sent_log"]), use_container_width=True
        )
        csv_path = os.path.join(ROOT_DIR, "sent_emails_log.csv")
        if os.path.isfile(csv_path):
            with open(csv_path, "r") as f:
                st.download_button(
                    "\U0001f4e5 Download CSV Log",
                    data=f.read(),
                    file_name="sent_emails_log.csv",
                    mime="text/csv",
                )


# ──────────────────────────────────────────────────────────
# TAB 2 — Inbox Reply Agent
# ──────────────────────────────────────────────────────────

with tab2:
    st.markdown("## \U0001f4e9 AI Inbox Reply Agent")
    st.caption(
        "Fetch unread emails \u2192 AI classifies & generates replies "
        "\u2192 you approve before sending"
    )

    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        if sender_email and sender_password and groq_api_key:
            st.success("\u2705 Credentials ready.")
        else:
            miss = []
            if not sender_email:
                miss.append("Email")
            if not sender_password:
                miss.append("Password")
            if not groq_api_key:
                miss.append("Groq Key")
            st.warning(f"\u26a0\ufe0f Missing: {', '.join(miss)}")

    with col_f2:
        fetch_btn = st.button(
            "\U0001f4e5 Fetch Unread Emails",
            disabled=not (sender_email and sender_password),
            use_container_width=True,
        )

    if fetch_btn:
        with st.spinner("\U0001f4ec Connecting to inbox via IMAP \u2026"):
            try:
                emails = fetch_emails(
                    imap_server=imap_server,
                    email_address=sender_email,
                    email_password=sender_password,
                    filter_mode="unread",
                    max_results=20,
                )
                st.session_state["inbox_emails"] = emails
                st.session_state["inbox_loaded"] = True
                st.session_state["inbox_selected_idx"] = None
                st.session_state["inbox_classification"] = None
                st.session_state["inbox_reply_ready"] = False
                if emails:
                    st.success(f"\u2705 Fetched **{len(emails)}** unread emails.")
                else:
                    st.info("\U0001f4ed No unread emails found.")
            except Exception as e:
                st.error(f"\u274c IMAP error: {e}")

    # Email list
    if st.session_state["inbox_loaded"] and st.session_state["inbox_emails"]:
        st.markdown("---")
        st.markdown("### \U0001f4ec Unread Emails")

        emails = st.session_state["inbox_emails"]

        for i, em in enumerate(emails):
            body_prev = em["body"][:120].replace("\n", " ")
            if len(em["body"]) > 120:
                body_prev += "\u2026"
            ts = em.get("timestamp", "")

            st.markdown(
                '<div class="email-card">'
                f'<span class="timestamp">{ts}</span>'
                f'<span class="sender">\U0001f4e7 {em["sender"]} '
                f'&lt;{em["sender_email"]}&gt;</span>'
                f'<h5>{em["subject"]}</h5>'
                f'<p class="preview">{body_prev}</p>'
                "</div>",
                unsafe_allow_html=True,
            )

            if st.button(
                "\U0001f4ac Generate Reply",
                key=f"reply_btn_{i}",
                use_container_width=True,
            ):
                st.session_state["inbox_selected_idx"] = i
                st.session_state["inbox_reply_ready"] = False
                st.session_state["inbox_generating"] = True

                selected = emails[i]

                with st.status(
                    "\U0001f916 Processing \u2026", expanded=True
                ) as status:
                    st.write("\U0001f3f7\ufe0f Classifying email intent \u2026")
                    classification = do_classify(
                        sender=selected["sender"],
                        sender_email=selected["sender_email"],
                        subject=selected["subject"],
                        body=selected["body"],
                        groq_api_key=groq_api_key,
                    )
                    st.session_state["inbox_classification"] = classification
                    st.write(
                        f"\u2705 Intent: **{classification['intent']}** "
                        f"| Reply needed: **{classification['requires_reply']}**"
                    )

                    st.write("\u270d\ufe0f Generating reply \u2026")
                    reply_result = do_generate_reply(
                        sender=selected["sender"],
                        sender_email=selected["sender_email"],
                        subject=selected["subject"],
                        body=selected["body"],
                        intent=classification["intent"],
                        groq_api_key=groq_api_key,
                    )

                    st.session_state["inbox_reply_body"] = reply_result["body"]
                    st.session_state["inbox_reply_subject"] = reply_result["subject"]
                    st.session_state["inbox_reply_ready"] = True
                    st.session_state["inbox_generating"] = False
                    status.update(
                        label="\u2705 Reply generated!", state="complete"
                    )

                st.rerun()

    # Reply review
    if (
        st.session_state.get("inbox_reply_ready")
        and st.session_state.get("inbox_selected_idx") is not None
    ):
        st.markdown("---")
        st.markdown("### \u2709\ufe0f Review & Send Reply")

        idx = st.session_state["inbox_selected_idx"]
        original = st.session_state["inbox_emails"][idx]
        classification = st.session_state.get("inbox_classification", {})

        intent = classification.get("intent", "other")
        st.markdown(
            '<div class="research-card">'
            f"<h4>\U0001f4e8 Original Email "
            f'<span class="intent-badge intent-{intent}">{intent}</span></h4>'
            f'<b>From:</b> {original["sender"]} '
            f'&lt;{original["sender_email"]}&gt;<br>'
            f'<b>Subject:</b> {original["subject"]}<br><br>'
            f'{original["body"][:1500]}'
            "</div>",
            unsafe_allow_html=True,
        )

        ed_subj = st.text_input(
            "\U0001f4cc Reply Subject",
            value=st.session_state["inbox_reply_subject"],
            key="reply_subj_ed",
        )
        ed_body = st.text_area(
            "\U0001f4dd Reply Body",
            value=st.session_state["inbox_reply_body"],
            height=250,
            key="reply_body_ed",
        )

        st.markdown("#### \U0001f441\ufe0f Reply Preview")
        preview_r = ed_body.replace("\n", "<br>")
        st.markdown(
            f'<div style="background:#f8f9fa;color:#1a1a2e;padding:1rem 1.2rem;'
            f'border-radius:10px;font-family:Arial;font-size:14px;line-height:1.6;'
            f'border:1px solid #dee2e6;">{preview_r}</div>',
            unsafe_allow_html=True,
        )

        r1, r2, r3 = st.columns(3)
        with r1:
            send_reply_btn = st.button(
                "\U0001f4e4 Send Reply", use_container_width=True, key="send_reply"
            )
        with r2:
            regen_reply = st.button(
                "\U0001f504 Regenerate", use_container_width=True, key="regen_reply"
            )
        with r3:
            discard_reply = st.button(
                "\U0001f5d1\ufe0f Discard", use_container_width=True, key="discard_reply"
            )

        if send_reply_btn:
            if not sender_email or not sender_password:
                st.error("\u274c Enter email credentials in sidebar.")
            else:
                with st.spinner("\U0001f4e4 Sending reply \u2026"):
                    result = send_reply(
                        to_email=original["sender_email"],
                        subject=ed_subj,
                        body=ed_body,
                        message_id=original.get("message_id", ""),
                        in_reply_to=original.get("message_id", ""),
                        smtp_server=smtp_server,
                        smtp_port=int(smtp_port),
                        sender_email=sender_email,
                        sender_password=sender_password,
                    )
                if result["success"]:
                    st.success(
                        f'\u2705 Reply sent to **{original["sender_email"]}**!'
                    )
                    st.balloons()
                    try:
                        mark_as_read(
                            imap_server,
                            sender_email,
                            sender_password,
                            original["uid"],
                        )
                    except Exception:
                        pass
                else:
                    st.error(f"\u274c Failed: {result['error']}")
                st.session_state["inbox_reply_ready"] = False
                st.session_state["inbox_selected_idx"] = None

        if regen_reply:
            st.session_state["inbox_reply_ready"] = False
            st.session_state["inbox_reply_body"] = ""
            st.session_state["inbox_reply_subject"] = ""
            st.rerun()

        if discard_reply:
            st.session_state["inbox_reply_ready"] = False
            st.session_state["inbox_selected_idx"] = None
            st.info("\U0001f5d1\ufe0f Reply discarded.")


# ──────────────────────────────────────────────────────────
# TAB 3 — Smart Compose
# ──────────────────────────────────────────────────────────

with tab3:
    st.markdown("## \u270d\ufe0f Smart Compose Email")
    st.caption(
        "Write a short idea \u2192 AI generates a complete professional email"
    )

    st.markdown("### 1\ufe0f\u20e3 Email Details")

    comp_recipient = st.text_input(
        "\U0001f4e7 Recipient Email", key="comp_recipient"
    )
    comp_cc = st.text_input(
        "\U0001f4cb CC (comma-separated)", key="comp_cc"
    )
    comp_bcc = st.text_input(
        "\U0001f4cb BCC (comma-separated)", key="comp_bcc"
    )
    comp_topic = st.text_area(
        "\U0001f4a1 Brief Topic / Idea",
        height=100,
        key="comp_topic",
        placeholder="e.g. Follow up for internship position at Google",
    )
    comp_tone = st.selectbox(
        "\U0001f3a8 Tone",
        ["Professional", "Friendly", "Formal", "Casual", "Persuasive"],
        key="comp_tone",
    )
    comp_attachments = st.file_uploader(
        "\U0001f4ce Attachments", accept_multiple_files=True, key="comp_attach"
    )

    can_generate = bool(comp_recipient and comp_topic and groq_api_key)
    gen_compose_btn = st.button(
        "\U0001f916 Generate Email",
        use_container_width=True,
        disabled=not can_generate,
        key="gen_compose_btn",
    )

    if gen_compose_btn:
        st.session_state["compose_ready"] = False
        st.session_state["compose_generating"] = True

        att_paths = []
        if comp_attachments:
            for att in comp_attachments:
                att_paths.append(_save_uploaded(att))
        st.session_state["compose_attachments"] = att_paths

        with st.status(
            "\u270d\ufe0f Generating email \u2026", expanded=True
        ) as status:
            result = compose_generate(
                topic=comp_topic,
                recipient=comp_recipient,
                tone=comp_tone.lower(),
                groq_api_key=groq_api_key,
            )

            st.session_state["compose_subject"] = result["subject"]
            st.session_state["compose_body"] = result["body"]
            st.session_state["compose_ready"] = True
            st.session_state["compose_generating"] = False
            status.update(label="\u2705 Email generated!", state="complete")

        st.rerun()

    # Review & send
    if st.session_state.get("compose_ready"):
        st.markdown("---")
        st.markdown("### 2\ufe0f\u20e3 Review & Send")

        ed_subj_c = st.text_input(
            "\U0001f4cc Subject",
            value=st.session_state["compose_subject"],
            key="comp_subj_ed",
        )
        ed_body_c = st.text_area(
            "\U0001f4dd Email Body",
            value=st.session_state["compose_body"],
            height=300,
            key="comp_body_ed",
        )

        st.markdown("#### \U0001f441\ufe0f Email Preview")
        preview_c = ed_body_c.replace("\n", "<br>")
        st.markdown(
            f'<div style="background:#f8f9fa;color:#1a1a2e;padding:1rem 1.2rem;'
            f'border-radius:10px;font-family:Arial;font-size:14px;line-height:1.6;'
            f'border:1px solid #dee2e6;">{preview_c}</div>',
            unsafe_allow_html=True,
        )

        s1, s2, s3 = st.columns(3)
        with s1:
            send_comp = st.button(
                "\U0001f4e4 Send Email", use_container_width=True, key="send_compose"
            )
        with s2:
            regen_comp = st.button(
                "\U0001f504 Regenerate", use_container_width=True, key="regen_compose"
            )
        with s3:
            discard_comp = st.button(
                "\U0001f5d1\ufe0f Discard", use_container_width=True, key="discard_compose"
            )

        if send_comp:
            if not sender_email or not sender_password:
                st.error("\u274c Enter email credentials in sidebar.")
            else:
                with st.spinner("\U0001f4e4 Sending email \u2026"):
                    result = send_composed_email(
                        to_email=comp_recipient,
                        subject=ed_subj_c,
                        body=ed_body_c,
                        smtp_server=smtp_server,
                        smtp_port=int(smtp_port),
                        sender_email=sender_email,
                        sender_password=sender_password,
                        cc=comp_cc,
                        bcc=comp_bcc,
                        attachment_paths=st.session_state.get("compose_attachments"),
                    )
                if result["success"]:
                    st.success(f"\u2705 Email sent to **{comp_recipient}**!")
                    st.balloons()
                else:
                    st.error(f"\u274c Failed: {result['error']}")
                st.session_state["compose_ready"] = False
                st.session_state["compose_body"] = ""
                st.session_state["compose_subject"] = ""

        if regen_comp:
            st.session_state["compose_ready"] = False
            st.session_state["compose_body"] = ""
            st.session_state["compose_subject"] = ""
            st.rerun()

        if discard_comp:
            st.session_state["compose_ready"] = False
            st.session_state["compose_body"] = ""
            st.session_state["compose_subject"] = ""
            st.info("\U0001f5d1\ufe0f Discarded.")

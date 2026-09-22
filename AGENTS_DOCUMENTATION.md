# 🤖 Agents Documentation — Deep Dive

> **Complete technical reference** for every agent, node, edge, state field, LLM prompt, and execution flow in the Email Automation AI system.
>
> This document is designed to explain the entire system architecture thoroughly — suitable for interview preparation and code review.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [LangGraph Fundamentals](#2-langgraph-fundamentals)
3. [System 1 — HR Outreach Agent](#3-system-1--hr-outreach-agent)
4. [System 2 — Inbox Reply Agent](#4-system-2--inbox-reply-agent)
5. [System 3 — Smart Compose Agent](#5-system-3--smart-compose-agent)
6. [Tool Modules](#6-tool-modules)
7. [Streamlit UI Integration](#7-streamlit-ui-integration)
8. [Human-in-the-Loop Pattern](#8-human-in-the-loop-pattern)
9. [LLM Prompt Design](#9-llm-prompt-design)
10. [Error Handling](#10-error-handling)
11. [Interview Q&A](#11-interview-qa)

---

## 1. Architecture Overview

### High-Level Design

The system follows a **modular architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────┐
│                   app.py (Streamlit UI)              │
│  ┌──────────────┬──────────────┬──────────────┐     │
│  │  Tab 1:      │  Tab 2:      │  Tab 3:      │     │
│  │  HR Outreach │  Inbox Reply │  Smart Compose│     │
│  └──────┬───────┴──────┬───────┴──────┬───────┘     │
└─────────┼──────────────┼──────────────┼─────────────┘
          │              │              │
          ▼              ▼              ▼
┌─────────────────────────────────────────────────────┐
│                 agents/ (LangGraph)                   │
│  ┌──────────────┬──────────────┬──────────────┐     │
│  │ outreach_    │ reply_       │ compose_     │     │
│  │ agent.py     │ agent.py     │ agent.py     │     │
│  └──────┬───────┴──────┬───────┴──────┬───────┘     │
└─────────┼──────────────┼──────────────┼─────────────┘
          │              │              │
          ▼              ▼              ▼
┌─────────────────────────────────────────────────────┐
│                    tools/ (Utilities)                 │
│  ┌────────────┬─────────────┬──────────┬──────────┐ │
│  │email_sender│email_fetcher│resume_   │research_ │ │
│  │   .py      │   .py       │parser.py │tool.py   │ │
│  └────────────┴─────────────┴──────────┴──────────┘ │
└─────────────────────────────────────────────────────┘
```

### Design Principles

1. **Separation of concerns** — Tools handle I/O (SMTP, IMAP, PDF, web search). Agents handle logic (LLM calls, state management, routing). UI handles presentation.
2. **Simple LangGraph** — Each agent is a `StateGraph` with typed state, node functions, and conditional edges. No complex abstractions.
3. **Dual-interface nodes** — Each agent exposes both:
   - **Core functions** (`do_research()`, `do_classify()`, etc.) with simple parameters — called by the Streamlit UI
   - **Node wrappers** (`research_company_node()`, `classify_email_node()`, etc.) that read/write from `TypedDict` state — used by LangGraph
4. **Human-in-the-loop** — Every agent uses `interrupt_before=["human_review"]` to pause execution and wait for human approval.

---

## 2. LangGraph Fundamentals

### What is LangGraph?

LangGraph is a framework for building **stateful, multi-step AI workflows** as directed graphs. Each workflow is defined as a `StateGraph` with:

- **State** — A `TypedDict` that holds all data flowing through the graph
- **Nodes** — Python functions that read from state and return updates
- **Edges** — Connections between nodes (linear or conditional)
- **Checkpointer** — Persists state between invocations (enables interrupt/resume)

### Key Concepts Used

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 1. Define state
class MyState(TypedDict, total=False):
    input_data: str
    result: str
    approval: str

# 2. Define nodes (functions that take state, return partial updates)
def process(state: MyState) -> dict:
    return {"result": "processed " + state["input_data"]}

def review(state: MyState) -> dict:
    return {}  # interrupt point — UI handles this

# 3. Define routing
def route(state: MyState) -> str:
    if state["approval"] == "yes":
        return "next_step"
    return END

# 4. Build graph
graph = StateGraph(MyState)
graph.add_node("process", process)
graph.add_node("review", review)
graph.set_entry_point("process")
graph.add_edge("process", "review")
graph.add_conditional_edges("review", route, {"next_step": "next_step", END: END})

# 5. Compile with checkpointer
checkpointer = MemorySaver()
compiled = graph.compile(checkpointer=checkpointer, interrupt_before=["review"])
```

### Why LangGraph?

| Feature | Benefit |
|---------|---------|
| `StateGraph` | Explicit data flow — every piece of data is in the typed state |
| `interrupt_before` | Built-in human-in-the-loop — graph pauses at specified nodes |
| `MemorySaver` | State persistence — enables resume after interrupt |
| Conditional edges | Dynamic routing — graph path depends on runtime decisions |
| Composability | Each agent is independent — can be tested and modified separately |

---

## 3. System 1 — HR Outreach Agent

**File:** `agents/outreach_agent.py`

### Purpose

Automates personalised internship/job outreach to HR contacts. Given a spreadsheet of companies and a resume PDF, it:
1. Parses the resume
2. Researches each company online
3. Generates a personalised email
4. Waits for human approval
5. Sends or regenerates

### State Definition

```python
class OutreachState(TypedDict, total=False):
    # Data (loaded once)
    companies: list[dict]       # List of company records from Excel
    resume_summary: str         # LLM-generated resume summary
    resume_path: str            # Path to resume PDF (for attachment)
    excel_path: str             # Path to Excel file
    current_index: int          # Index of company being processed

    # Per-company (changes each iteration)
    company_research: str       # Tavily search + LLM summary of the company
    generated_email: str        # LLM-generated email body (HTML)
    email_subject: str          # Generated subject line

    # Human review
    approval_status: str        # "approve" | "regenerate" | "skip"
    edited_email: str           # User's edited version of the email
    edited_subject: str         # User's edited subject

    # Config
    groq_api_key: str
    tavily_api_key: str
    smtp_server: str
    smtp_port: int
    sender_email: str
    sender_password: str
    project_dir: str            # For CSV log file path

    # Logging
    sent_log: list[dict]        # Accumulated send results
```

**Why `total=False`?** — Not all fields are present at all times. `total=False` means every field is optional, which is necessary because LangGraph nodes only return partial state updates.

### Node Details

#### Node 1: `load_data_node`

```python
def load_data_node(state: OutreachState) -> dict:
    """Read Excel file and parse resume PDF."""
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
```

| Aspect | Detail |
|--------|--------|
| **Purpose** | Load all data needed for the outreach campaign |
| **Reads from state** | `excel_path`, `resume_path`, `groq_api_key` |
| **Writes to state** | `companies`, `resume_summary`, `current_index`, `sent_log` |
| **External calls** | `pd.read_excel()`, `parse_resume()` (tools/resume_parser.py) |
| **Why it exists** | Centralises data loading. The resume is parsed once and reused for all companies. |

#### Node 2: `research_company_node`

```python
def research_company_node(state: OutreachState) -> dict:
    company = state["companies"][state["current_index"]]
    result = do_research(
        company_name=company.get("Company Name", "Unknown"),
        industry=company.get("Industry", ""),
        location=company.get("Location", ""),
        tavily_api_key=state["tavily_api_key"],
        groq_api_key=state["groq_api_key"],
    )
    return {"company_research": result}
```

| Aspect | Detail |
|--------|--------|
| **Purpose** | Research the current company online to personalise the email |
| **Reads from state** | `companies`, `current_index`, `tavily_api_key`, `groq_api_key` |
| **Writes to state** | `company_research` |
| **External calls** | `research_company()` (tools/research_tool.py) → Tavily API + Groq LLM |
| **Why it exists** | Emails with company-specific details have higher response rates. The research provides context the LLM uses to write relevant emails. |

**Research pipeline:**
```
User's company data → Tavily search query → 5 search results → Groq LLM summarisation → ~100 word company summary
```

#### Node 3: `generate_email_node`

```python
def generate_email_node(state: OutreachState) -> dict:
    company = state["companies"][state["current_index"]]
    company_name = company.get("Company Name", "Unknown")

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
```

| Aspect | Detail |
|--------|--------|
| **Purpose** | Generate a personalised outreach email using the company research and resume |
| **Reads from state** | `companies`, `current_index`, `company_research`, `resume_summary`, `groq_api_key` |
| **Writes to state** | `generated_email`, `email_subject`, `approval_status` (reset) |
| **External calls** | Groq LLM (`llama-3.3-70b-versatile`) |
| **Why it exists** | Core value — AI-generated personalised emails. Resets approval status so regenerated emails require fresh approval. |

**The LLM prompt enforces:**
- 3-paragraph structure (intro → skills → closing)
- HTML formatting with `<b>` tags
- 150–200 word target
- No placeholders
- Professional sign-off

#### Node 4: `human_review_node`

```python
def human_review_node(state: OutreachState) -> dict:
    return {}
```

| Aspect | Detail |
|--------|--------|
| **Purpose** | Interrupt point — graph pauses here |
| **Reads from state** | Nothing |
| **Writes to state** | Nothing (empty dict) |
| **Why it exists** | This is the key human-in-the-loop mechanism. The graph is compiled with `interrupt_before=["human_review"]`, so execution stops before this node runs. The Streamlit UI shows the generated email and buttons. When the user clicks a button, `approval_status` is updated in state and the graph resumes. |

#### Node 5: `send_email_node`

```python
def send_email_node(state: OutreachState) -> dict:
    # ... reads company data, sends via SMTP, logs to CSV
    return {"sent_log": sent_log}
```

| Aspect | Detail |
|--------|--------|
| **Purpose** | Send the approved email and log the result |
| **Reads from state** | `companies`, `current_index`, `edited_email` or `generated_email`, SMTP config |
| **Writes to state** | `sent_log` (appends new entry) |
| **External calls** | `send_email()`, `log_sent_email()`, `rate_limited_sleep()` (tools/email_sender.py) |
| **Why it exists** | Handles the actual sending + logging. Uses edited version if available, otherwise the generated version. Includes rate limiting to avoid SMTP throttling. |

### Routing Function

```python
def route_after_review(state: OutreachState) -> str:
    status = state.get("approval_status", "")
    if status == "approve":
        return "send_email"       # → send the email
    elif status == "regenerate":
        return "generate_email"   # → loop back and regenerate
    return END                    # → skip, stop processing
```

This is a **conditional edge** that determines what happens after human review:

| `approval_status` | Route | Effect |
|-------------------|-------|--------|
| `"approve"` | → `send_email` | Email is sent via SMTP |
| `"regenerate"` | → `generate_email` | New email is generated (loops back) |
| `"skip"` or empty | → `END` | Processing stops |

### Graph Construction

```python
def build_outreach_graph():
    graph = StateGraph(OutreachState)

    # Add all nodes
    graph.add_node("load_data", load_data_node)
    graph.add_node("research_company", research_company_node)
    graph.add_node("generate_email", generate_email_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("send_email", send_email_node)

    # Linear edges
    graph.set_entry_point("load_data")
    graph.add_edge("load_data", "research_company")
    graph.add_edge("research_company", "generate_email")
    graph.add_edge("generate_email", "human_review")

    # Conditional edge after human review
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

    # Compile with interrupt
    checkpointer = MemorySaver()
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],
    )
    return compiled, checkpointer
```

### Complete Execution Flow

```
1. User uploads Resume PDF + HR Contacts Excel
2. User clicks "Load Data"
   └─→ load_data_node() runs
       ├─ Reads Excel → list of company dicts
       ├─ Parses PDF → raw text
       └─ Calls Groq LLM → resume summary

3. User selects a company from dropdown
4. User clicks "Research & Generate Email"
   └─→ research_company_node() runs
       ├─ Searches Tavily with company query
       └─ Groq LLM summarises search results → company_research
   └─→ generate_email_node() runs
       ├─ Builds system prompt (3-paragraph structure)
       ├─ Builds user prompt (company research + resume)
       └─ Groq LLM generates → email body + subject

5. ──── INTERRUPT: human_review ────
   UI shows: generated email, subject, preview
   User can: edit subject, edit body

6. User clicks one of:
   ├─ "Approve & Send" → approval_status = "approve"
   │   └─→ send_email_node() runs
   │       ├─ Sends via SMTP with resume attachment
   │       ├─ Logs to CSV
   │       └─→ END
   │
   ├─ "Regenerate" → approval_status = "regenerate"
   │   └─→ generate_email_node() runs again (new email)
   │       └─→ INTERRUPT: human_review (loop)
   │
   └─ "Skip" → approval_status = "skip"
       └─→ END
```

---

## 4. System 2 — Inbox Reply Agent

**File:** `agents/reply_agent.py`

### Purpose

Reads unread emails from the user's inbox, classifies their intent, generates professional replies, and sends after human approval.

### State Definition

```python
class ReplyState(TypedDict, total=False):
    # Inbox
    fetched_emails: list[dict]  # All unread emails from IMAP
    selected_email: dict        # The one email being replied to

    # Classification
    classification: dict        # {"intent": str, "requires_reply": bool}

    # Generated reply
    reply_subject: str          # "Re: <original subject>"
    reply_body: str             # LLM-generated reply (HTML)

    # Human review
    approval_status: str        # "send" | "regenerate" | "discard"
    edited_reply: str           # User's edited reply
    edited_subject: str

    # Optional
    user_instruction: str       # Custom instruction for reply tone/content

    # Config
    groq_api_key: str
    smtp_server: str
    smtp_port: int
    imap_server: str
    sender_email: str
    sender_password: str
```

### Node Details

#### Node 1: `fetch_emails_node`

```python
def fetch_emails_node(state: ReplyState) -> dict:
    emails = _fetch_emails(
        imap_server=state.get("imap_server", "imap.gmail.com"),
        email_address=state["sender_email"],
        email_password=state["sender_password"],
        filter_mode="unread",
        max_results=20,
    )
    return {"fetched_emails": emails}
```

| Aspect | Detail |
|--------|--------|
| **Purpose** | Connect to IMAP and fetch unread emails |
| **Returns** | `fetched_emails` — list of email dicts |
| **External calls** | `fetch_emails()` (tools/email_fetcher.py) → IMAP SSL |
| **Why it exists** | Entry point. Fetches raw inbox data for the agent to process. |

**Each email dict contains:**
```python
{
    "uid": "12345",              # IMAP unique ID
    "sender": "John Doe",       # Display name
    "sender_email": "john@x.com", # Email address
    "subject": "Meeting Request",
    "body": "Hi, can we meet...", # Plain text (HTML stripped)
    "timestamp": "2024-01-15T10:30:00",
    "thread_id": "<msg-id>",
    "message_id": "<unique-msg-id>",
    "in_reply_to": "",
    "is_read": False,
}
```

#### Node 2: `classify_email_node`

```python
def classify_email_node(state: ReplyState) -> dict:
    em = state["selected_email"]
    result = do_classify(
        sender=em.get("sender", ""),
        sender_email=em.get("sender_email", ""),
        subject=em.get("subject", ""),
        body=em.get("body", ""),
        groq_api_key=state["groq_api_key"],
    )
    return {"classification": result}
```

| Aspect | Detail |
|--------|--------|
| **Purpose** | Classify the email's intent so the reply can be contextually appropriate |
| **LLM Output** | JSON: `{"intent": "job_reply", "requires_reply": true}` |
| **Categories** | `job_reply`, `meeting`, `recruiter`, `question`, `newsletter`, `spam`, `other` |
| **Why it exists** | Different email types need different reply tones. A meeting request needs scheduling language. A recruiter email needs interest expression. Classification drives the reply generation. |

**Classification prompt design:**
- Temperature: **0** (deterministic — we want consistent classification)
- Max tokens: **200** (JSON output is small)
- Output format: **strict JSON** with fallback parsing

**Fallback logic:**
```python
try:
    text = response.content.strip()
    start, end = text.find("{"), text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
except (json.JSONDecodeError, ValueError):
    pass
return {"intent": "other", "requires_reply": True}  # safe default
```

This ensures the system never crashes on malformed LLM output.

#### Node 3: `generate_reply_node`

```python
def generate_reply_node(state: ReplyState) -> dict:
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
```

| Aspect | Detail |
|--------|--------|
| **Purpose** | Generate a contextually appropriate reply based on classification |
| **LLM Settings** | Temperature: 0.7 (creative but controlled), 500 max tokens |
| **Subject line** | Prepends `Re:` if not already present |
| **User instruction** | Optional field — user can add "be brief" or "express strong interest" |
| **Why it exists** | Core value — AI generates a reply that matches the email's intent and tone |

#### Node 4: `human_review_node`

Empty function — interrupt point (same pattern as Outreach Agent).

#### Node 5: `send_reply_node`

```python
def send_reply_node(state: ReplyState) -> dict:
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
```

| Aspect | Detail |
|--------|--------|
| **Thread preservation** | Sets `In-Reply-To` and `References` headers using the original email's `message_id`. This keeps the reply in the same Gmail/Outlook thread. |
| **Priority** | Uses edited version if available, otherwise uses generated version |
| **Why it exists** | Sends the actual reply while preserving the email conversation thread |

### Routing

```python
def route_after_review(state: ReplyState) -> str:
    status = state.get("approval_status", "")
    if status == "send":
        return "send_reply"
    elif status == "regenerate":
        return "generate_reply"
    return END
```

| `approval_status` | Route | Effect |
|-------------------|-------|--------|
| `"send"` | → `send_reply` | Reply is sent |
| `"regenerate"` | → `generate_reply` | New reply is generated (loop) |
| `"discard"` or empty | → `END` | Reply is discarded |

### Complete Execution Flow

```
1. User clicks "Fetch Unread Emails"
   └─→ fetch_emails_node() runs
       └─ IMAP connection → fetches up to 20 unread emails

2. UI displays email list with sender, subject, preview

3. User clicks "Generate Reply" on an email
   └─→ classify_email_node() runs
       ├─ Sends email content to Groq LLM
       └─ Returns: {"intent": "job_reply", "requires_reply": true}
   └─→ generate_reply_node() runs
       ├─ Uses classification + original email
       └─ Groq LLM generates professional reply

4. ──── INTERRUPT: human_review ────
   UI shows: original email (with intent badge), generated reply, preview
   User can: edit subject, edit reply body

5. User clicks one of:
   ├─ "Send Reply" → approval_status = "send"
   │   └─→ send_reply_node() runs
   │       ├─ Sends via SMTP with thread headers
   │       ├─ Marks original as read (IMAP)
   │       └─→ END
   │
   ├─ "Regenerate" → approval_status = "regenerate"
   │   └─→ generate_reply_node() runs again
   │       └─→ INTERRUPT: human_review (loop)
   │
   └─ "Discard" → approval_status = "discard"
       └─→ END
```

---

## 5. System 3 — Smart Compose Agent

**File:** `agents/compose_agent.py`

### Purpose

Takes a brief topic/idea from the user and generates a complete professional email with appropriate subject line. The simplest of the three agents.

### State Definition

```python
class ComposeState(TypedDict, total=False):
    # User input
    recipient_email: str
    brief_topic: str         # "Follow up on internship at Google"
    tone: str                # professional | friendly | formal | casual | persuasive
    cc: str
    bcc: str
    attachment_paths: list[str]

    # Generated
    composed_subject: str
    composed_body: str

    # Human review
    approval_status: str     # "send" | "regenerate" | "discard"
    edited_subject: str
    edited_body: str

    # Config
    groq_api_key: str
    smtp_server: str
    smtp_port: int
    sender_email: str
    sender_password: str
```

### Node Details

#### Node 1: `generate_email_node`

The LLM is asked to return output in a structured format:

```
SUBJECT: <subject line>
BODY:
<email body in HTML>
```

**Parsing logic:**

```python
# Primary: split on "BODY:" marker
if "SUBJECT:" in text and "BODY:" in text:
    parts = text.split("BODY:", 1)
    body = parts[1].strip()
    subj_parts = parts[0].split("SUBJECT:", 1)
    subject = subj_parts[1].strip().split("\n")[0].strip()

# Fallback: line-by-line search
elif "SUBJECT:" in text:
    for i, line in enumerate(text.split("\n")):
        if line.strip().startswith("SUBJECT:"):
            subject = line.replace("SUBJECT:", "").strip()
            body = "\n".join(text.split("\n")[i + 1:]).strip()
            break

# Final fallback: use topic as subject
if not subject:
    subject = f"Regarding: {topic[:50]}"
```

**Why multi-level parsing?** — LLMs don't always follow the exact format. The parsing has three fallback levels to handle variations in output.

#### Node 2: `human_review_node`

Empty interrupt point — same pattern.

#### Node 3: `send_email_node`

```python
def send_email_node(state: ComposeState) -> dict:
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
```

| Feature | Detail |
|---------|--------|
| **CC/BCC** | Supports comma-separated CC and BCC recipients |
| **Attachments** | Supports multiple file attachments |
| **Priority** | Uses edited version if available |

### Graph (Simplest of the three)

```
generate_email ──→ human_review
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
  send_email    generate_email        END
```

Only 3 nodes, 1 conditional edge. The simplest LangGraph pattern.

### Complete Execution Flow

```
1. User enters: recipient, topic, tone
2. User clicks "Generate Email"
   └─→ generate_email_node() runs
       ├─ Builds prompt with topic + tone
       ├─ Groq LLM generates SUBJECT + BODY
       └─ Parses structured output

3. ──── INTERRUPT: human_review ────
   UI shows: subject, body, preview

4. User clicks one of:
   ├─ "Send Email" → send_email_node() → END
   ├─ "Regenerate" → generate_email_node() → INTERRUPT (loop)
   └─ "Discard" → END
```

---

## 6. Tool Modules

### tools/email_sender.py

Handles all SMTP operations.

**Functions:**

| Function | Purpose | Returns |
|----------|---------|---------|
| `send_email()` | Send outreach email with PDF attachment | `{"success": bool, "error": str}` |
| `send_reply()` | Send reply with thread headers (`In-Reply-To`, `References`) | `{"success": bool, "error": str}` |
| `send_composed_email()` | Send composed email with CC/BCC/attachments | `{"success": bool, "error": str}` |
| `log_sent_email()` | Append to CSV log file | `None` |
| `is_already_sent()` | Check if email was already sent to address | `bool` |
| `rate_limited_sleep()` | Delay between sends to avoid throttling | `None` |

**SMTP Flow:**
```
1. Create MIMEMultipart message
2. Set headers (From, To, Subject, CC, In-Reply-To, References)
3. Attach HTML body wrapped in styled div
4. Attach files (if any)
5. Connect: SMTP → EHLO → STARTTLS → EHLO → LOGIN
6. Send mail
7. Return success/error
```

**Internal helpers:**
- `_wrap_html(body)` — Wraps body in a styled `<div>` with font-family and line-height
- `_attach_files(msg, paths)` — Attaches files as base64-encoded MIME parts
- `_collect_recipients(to, cc, bcc)` — Builds flat recipient list
- `_smtp_send(msg, recipients, ...)` — Handles SMTP connection and send

### tools/email_fetcher.py

Handles IMAP inbox reading.

**Functions:**

| Function | Purpose |
|----------|---------|
| `fetch_emails()` | Connect to IMAP, fetch emails matching criteria |
| `mark_as_read()` | Mark a specific email as Seen |

**IMAP Flow:**
```
1. Connect: IMAP4_SSL(server, 993)
2. Login with email + app password
3. Select folder (INBOX)
4. Search (UNSEEN / ALL)
5. Fetch RFC822 for each message
6. Parse headers: Subject, From, Date, Message-ID
7. Extract body: prefer text/plain, fallback to HTML (stripped)
8. Return list of email dicts (newest first)
```

**Body extraction priority:**
1. `text/plain` parts → used as-is
2. `text/html` parts → strip `<style>`, `<script>`, and all HTML tags → clean text
3. Empty → return `""`

### tools/resume_parser.py

**Functions:**

| Function | Purpose |
|----------|---------|
| `extract_text_from_pdf()` | Read all pages with PyPDF → concatenated text |
| `summarize_resume()` | Groq LLM → structured summary (name, skills, projects, education) |
| `parse_resume()` | High-level: extract + summarise → `{"raw_text", "summary"}` |

**LLM prompt for resume:**
```
"Produce a detailed summary covering:
1. Full name
2. Key technical skills (Full Stack + AI/ML)
3. ALL projects with 1-line descriptions
4. Education
5. Notable achievements or certifications

Keep it under 300 words."
```

Temperature: **0.3** (factual, low creativity — we want accurate extraction)

### tools/research_tool.py

**Functions:**

| Function | Purpose |
|----------|---------|
| `search_company()` | Tavily API search → list of result dicts |
| `summarize_company()` | Groq LLM → 100-word company summary |
| `research_company()` | End-to-end: search + summarise |

**Tavily query format:**
```
"{company_name} company {industry} {location} what do they do"
```

**Fallback:** If no search results found, returns a generic description using the input data.

---

## 7. Streamlit UI Integration

**File:** `app.py`

### Architecture Pattern

The Streamlit UI uses a **direct function call** pattern rather than invoking the LangGraph graphs programmatically. This is because Streamlit's rerun model (every interaction reruns the entire script) makes graph interrupt/resume complex.

**Pattern:**
```python
# Instead of:
graph.invoke(state, config)  # complex with Streamlit reruns

# We use:
result = do_generate_email(...)  # direct function call
st.session_state["email"] = result["body"]  # store in session
```

The LangGraph graphs are still properly defined in the agent files and can be invoked programmatically (e.g., from a CLI or API). The Streamlit UI calls the **core functions** directly for simplicity.

### Session State Management

Streamlit reruns the entire script on every interaction. All persistent data is stored in `st.session_state`:

```python
# Outreach state
st.session_state["companies"]        # list[dict]
st.session_state["resume_summary"]   # str
st.session_state["data_loaded"]      # bool
st.session_state["generated_email"]  # str
st.session_state["email_ready"]      # bool

# Reply state
st.session_state["inbox_emails"]     # list[dict]
st.session_state["inbox_loaded"]     # bool
st.session_state["inbox_reply_body"] # str
st.session_state["inbox_reply_ready"]# bool

# Compose state
st.session_state["compose_body"]     # str
st.session_state["compose_ready"]    # bool
```

### UI Flow Per Tab

Each tab follows the same pattern:
1. **Input section** — collect user input
2. **Generate button** — calls agent core function, stores result in session state
3. **Review section** — shows result with editable fields + preview
4. **Action buttons** — Approve/Regenerate/Discard

---

## 8. Human-in-the-Loop Pattern

This is the central safety mechanism. **No email is ever sent without explicit human approval.**

### LangGraph Implementation

```python
# In graph builder:
compiled = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_review"],  # ← pauses here
)

# The human_review node itself does nothing:
def human_review_node(state) -> dict:
    return {}

# The routing function reads the human's decision:
def route_after_review(state) -> str:
    if state["approval_status"] == "approve":
        return "send_email"
    elif state["approval_status"] == "regenerate":
        return "generate_email"
    return END
```

### Streamlit Implementation

In the UI, the pattern is:

```python
# 1. Generate email (store in session state)
if generate_btn:
    result = do_generate_email(...)
    st.session_state["generated_email"] = result["body"]
    st.session_state["email_ready"] = True

# 2. Show review section (only when email is ready)
if st.session_state["email_ready"]:
    edited = st.text_area("Edit email", value=st.session_state["generated_email"])

    if approve_btn:
        send_email(body=edited, ...)  # send the possibly-edited version
        st.session_state["email_ready"] = False

    if regenerate_btn:
        st.session_state["email_ready"] = False
        st.rerun()  # loops back to generate

    if discard_btn:
        st.session_state["email_ready"] = False
```

### Why Human-in-the-Loop?

1. **Quality control** — LLMs can make mistakes, use wrong names, or generate inappropriate content
2. **Personalisation** — Users can edit the generated email before sending
3. **Safety** — Prevents accidental mass-sending of bad emails
4. **Trust** — Users stay in control of their email communications

---

## 9. LLM Prompt Design

### Design Principles

1. **Role assignment** — Every prompt starts with "You are a [role]"
2. **Structured output** — Clear format requirements (JSON, SUBJECT/BODY markers)
3. **Explicit rules** — Numbered lists of do's and don'ts
4. **Temperature selection** — Low (0-0.3) for factual tasks, medium (0.7) for creative tasks
5. **Context injection** — Relevant data (research, resume, classification) included in user prompt

### Prompt Summary

| Agent | Node | Temperature | Purpose |
|-------|------|-------------|---------|
| Outreach | `generate_email` | 0.7 | Creative email writing |
| Reply | `classify_email` | 0.0 | Deterministic classification |
| Reply | `generate_reply` | 0.7 | Creative reply writing |
| Compose | `generate_email` | 0.7 | Creative email writing |
| (tool) | `summarize_resume` | 0.3 | Factual extraction |
| (tool) | `summarize_company` | 0.3 | Factual summarisation |

### Outreach Email Prompt (detailed)

**System prompt structure:**
```
Role: "professional outreach assistant"
Format: HTML with <b> tags
Structure:
  PARAGRAPH 1: Self-introduction + company interest
  PARAGRAPH 2: Skills + projects
  PARAGRAPH 3: Closing + sign-off
Rules: word count, no placeholders, formatting
```

**User prompt structure:**
```
Task: "Write internship email for {company}"
Context 1: Company Research (from Tavily + LLM)
Context 2: Resume Summary (from PDF + LLM)
Instruction: "Follow the structure above"
```

### Classification Prompt (detailed)

```
Role: "email classifier"
Categories: [list of 7 categories with descriptions]
Output: strict JSON format
Example: {"intent": "job_reply", "requires_reply": true}
```

**Why strict JSON?** — Enables programmatic parsing. The fallback logic handles cases where the LLM adds explanation text around the JSON.

---

## 10. Error Handling

### Email Sending

```python
def send_email(...) -> dict:
    try:
        # ... build message, connect SMTP, send
        return {"success": True, "error": None}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

All send functions return a result dict rather than raising exceptions. The UI checks `result["success"]` and shows appropriate messages.

### IMAP Fetching

```python
try:
    emails = fetch_emails(...)
except Exception as e:
    st.error(f"IMAP error: {e}")
```

IMAP errors (wrong password, server unreachable, etc.) are caught and displayed to the user.

### LLM Classification Fallback

```python
try:
    classification = json.loads(text[start:end])
except (json.JSONDecodeError, ValueError):
    classification = {"intent": "other", "requires_reply": True}
```

If the LLM returns malformed JSON, the system defaults to `"other"` intent with `requires_reply: True` — the safest fallback.

### LLM Output Parsing (Compose)

Three-level fallback for SUBJECT/BODY parsing:
1. Split on `"BODY:"` marker → extract subject and body
2. Line-by-line search for `"SUBJECT:"` → extract from there
3. Default subject: `"Regarding: {topic[:50]}"`

---

## 11. Interview Q&A

### Q: Why LangGraph instead of a simple function chain?

**A:** LangGraph provides three key benefits:
1. **Explicit state management** — All data flows through a typed state dict, making it easy to trace what each node reads and writes
2. **Built-in interrupt/resume** — The `interrupt_before` feature enables human-in-the-loop without custom checkpoint code
3. **Conditional routing** — The `add_conditional_edges` API makes branching logic (approve/regenerate/skip) clean and declarative

For a simple 3-step pipeline you don't need LangGraph. But once you add human review, regeneration loops, and conditional branching, LangGraph's graph model is much cleaner than nested if/else chains.

### Q: Why Groq instead of OpenAI?

**A:** Groq offers free API access with fast inference. The `llama-3.3-70b-versatile` model is comparable in quality to GPT-3.5 for email generation tasks. For a project that focuses on architecture rather than LLM capabilities, the free tier is ideal.

### Q: How does the human-in-the-loop work?

**A:** The graph is compiled with `interrupt_before=["human_review"]`. When execution reaches that node, LangGraph pauses. The checkpointer (MemorySaver) stores the current state. The UI reads the state (generated email, classification, etc.) and presents it to the user. When the user clicks a button, the `approval_status` field is updated and the graph resumes from where it left off. The conditional edge after `human_review` routes to the appropriate next node.

### Q: Why not use Gmail API instead of IMAP/SMTP?

**A:** Gmail API requires OAuth2 setup, Google Cloud project, and consent screen configuration. IMAP/SMTP with App Passwords is:
- Simpler to set up (no Google Cloud console needed)
- Works with any email provider (not just Gmail)
- Uses only Python stdlib (`imaplib`, `smtplib`)
- Zero additional dependencies

### Q: How do you handle email thread preservation?

**A:** The reply agent preserves threads using two email headers:
- `In-Reply-To: <original-message-id>` — tells the email client this is a reply
- `References: <original-message-id>` — maintains the thread chain

These are standard RFC 2822 headers that Gmail, Outlook, and all major clients use to group messages into conversations.

### Q: What's the dual-interface pattern in the agents?

**A:** Each agent exports two types of functions:

1. **Core functions** (simple parameters):
   ```python
   def do_classify(sender, subject, body, groq_api_key) -> dict
   ```
   Called directly by the Streamlit UI. Simple inputs, simple outputs.

2. **Node wrappers** (state dict):
   ```python
   def classify_email_node(state: ReplyState) -> dict
   ```
   Used by LangGraph graph. Reads from typed state, returns partial updates.

The node wrappers call the core functions internally. This avoids code duplication while supporting both programmatic graph execution and direct UI calls.

### Q: How would you scale this for production?

**A:**
1. Replace `MemorySaver` with a persistent checkpointer (Redis, PostgreSQL)
2. Add authentication to the Streamlit app
3. Use a task queue (Celery) for background email sending
4. Add rate limiting per sender address
5. Store sent emails in a database instead of CSV
6. Add email templates and A/B testing
7. Deploy with Docker and put behind a reverse proxy

### Q: Walk me through what happens when a user clicks "Generate Reply"

**A:**
1. User clicks the button → Streamlit reruns the script
2. The button click is detected, and we get the selected email from `st.session_state["inbox_emails"][i]`
3. `do_classify()` is called with the email's sender, subject, and body → Groq LLM returns `{"intent": "job_reply", "requires_reply": true}`
4. The classification is stored in `st.session_state["inbox_classification"]`
5. `do_generate_reply()` is called with the email data + classification intent → Groq LLM generates an HTML reply
6. The reply body and subject are stored in session state
7. `st.session_state["inbox_reply_ready"]` is set to `True`
8. `st.rerun()` triggers a page rerun
9. On rerun, the review section renders because `inbox_reply_ready` is `True`
10. The user sees the original email, classification badge, generated reply, and preview
11. The user can edit the reply and click Send/Regenerate/Discard

---

*This documentation covers the complete architecture of the Email Automation AI system. Every node, edge, state field, LLM prompt, and execution path is documented above. Use this as your reference for understanding how the system works internally.*

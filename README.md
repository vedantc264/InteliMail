# 📧 Email Automation AI

> **Three AI-powered agents** for email outreach, inbox reply, and smart compose — built with **LangGraph**, **Groq LLM**, and **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green?logo=data:image/svg+xml;base64,...)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Project Overview

This project is a **fully functional AI email automation system** that uses LangGraph state machines to orchestrate three independent agents. Each agent follows a human-in-the-loop pattern — **no email is ever sent without explicit human approval**.

| Agent | What it does |
|-------|-------------|
| **System 1 — HR Outreach** | Reads an Excel spreadsheet of companies, researches each one online, and generates personalised internship outreach emails |
| **System 2 — Inbox Reply** | Fetches unread emails via IMAP, classifies their intent with LLM, and generates professional replies |
| **System 3 — Smart Compose** | Takes a brief topic and tone, generates a complete email with subject line, and sends after approval |

---

## ✨ Features

- 🤖 **LangGraph Agents** — Each system is a proper LangGraph `StateGraph` with nodes, edges, conditional routing, and interrupt-based human review
- 🧠 **Groq LLM (Free)** — Uses `llama-3.3-70b-versatile` via Groq's free API for all text generation
- 📬 **Gmail IMAP/SMTP** — Reads inbox via IMAP and sends via SMTP using App Passwords
- 🔍 **Tavily Web Search** — Researches companies online before generating outreach emails
- 📄 **Resume Parsing** — Extracts text from PDF resumes and generates structured summaries
- 🎨 **Streamlit UI** — Clean, dark-themed 3-tab interface with live previews
- ✅ **Human-in-the-Loop** — Every email must be approved, edited, or regenerated before sending
- 📊 **CSV Logging** — All sent emails are logged to `sent_emails_log.csv`
- 🧵 **Thread Preservation** — Reply agent uses `In-Reply-To` / `References` headers to keep email threads intact

---

## 🏗️ Architecture

```
Email Automation AI/
│
├── app.py                        # Streamlit UI (3 tabs)
│
├── agents/                       # LangGraph agent definitions
│   ├── __init__.py
│   ├── outreach_agent.py         # System 1 — HR Outreach
│   ├── reply_agent.py            # System 2 — Inbox Reply
│   └── compose_agent.py          # System 3 — Smart Compose
│
├── tools/                        # Utility modules
│   ├── __init__.py
│   ├── email_sender.py           # SMTP send / reply / compose
│   ├── email_fetcher.py          # IMAP inbox reader
│   ├── resume_parser.py          # PDF text extraction + LLM summary
│   └── research_tool.py          # Tavily search + LLM summary
│
├── data/                         # Sample data files
│   ├── HR_Contacts_Cleaned.xlsx
│   └── RushikeshBhabad_Resume.pdf
│
├── .env                          # API keys & credentials (git-ignored)
├── .env.example                  # Template for .env
├── .gitignore
├── requirements.txt
├── README.md                     # This file
└── AGENTS_DOCUMENTATION.md       # Deep-dive agent documentation
```

---

## 🔄 LangGraph Workflows

### System 1 — HR Outreach Agent

```
load_data ──→ research_company ──→ generate_email ──→ human_review
                                                          │
                                          ┌───────────────┼───────────────┐
                                          ▼               ▼               ▼
                                     send_email    generate_email        END
                                      (approve)    (regenerate)         (skip)
                                          │
                                          ▼
                                         END
```

### System 2 — Inbox Reply Agent

```
fetch_emails ──→ classify_email ──→ generate_reply ──→ human_review
                                                           │
                                           ┌───────────────┼───────────────┐
                                           ▼               ▼               ▼
                                      send_reply    generate_reply        END
                                       (send)       (regenerate)       (discard)
                                           │
                                           ▼
                                          END
```

### System 3 — Smart Compose Agent

```
generate_email ──→ human_review
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
  send_email    generate_email        END
   (send)       (regenerate)       (discard)
       │
       ▼
      END
```

---

## 🛠️ Tech Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| **LLM** | Groq (`llama-3.3-70b-versatile`) | Free |
| **Agent Framework** | LangGraph (`StateGraph`) | Free |
| **Web Search** | Tavily API | Free tier |
| **Email Send** | Python `smtplib` (SMTP) | Free |
| **Email Read** | Python `imaplib` (IMAP) | Free |
| **PDF Parsing** | PyPDF | Free |
| **UI** | Streamlit | Free |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/email-automation-ai.git
cd email-automation-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` with your actual credentials:

```dotenv
GROQ_API_KEY=gsk_your_key_here
TAVILY_API_KEY=tvly-your_key_here
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_gmail_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
IMAP_SERVER=imap.gmail.com
```

### 5. Gmail App Password Setup

1. Go to [Google Account Security](https://myaccount.google.com/security)
2. Enable **2-Step Verification**
3. Go to **App Passwords** → Generate a new one
4. Use this 16-character password as `EMAIL_PASSWORD`

---

## ▶️ How to Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` with three tabs.

---

## 📖 Usage Guide

### Tab 1: HR Outreach

1. Upload your **Resume PDF** and **HR Contacts Excel** (columns: `Company Name`, `Email ID`, `Industry`, `Location`)
2. Click **Load Data** — resume is parsed, companies are loaded
3. Select a company from the dropdown
4. Click **Research & Generate Email** — AI researches the company online and drafts a personalised email
5. **Review** the email — edit subject/body if needed
6. Click **Approve & Send**, **Regenerate**, or **Skip**

### Tab 2: Inbox Reply

1. Ensure email credentials are in the sidebar
2. Click **Fetch Unread Emails** — connects via IMAP
3. Click **Generate Reply** on any email — AI classifies intent and drafts a reply
4. **Review** the reply — edit if needed
5. Click **Send Reply**, **Regenerate**, or **Discard**

### Tab 3: Smart Compose

1. Enter recipient email, topic, and select tone
2. Click **Generate Email** — AI creates subject + body
3. **Review** the email — edit if needed
4. Click **Send Email**, **Regenerate**, or **Discard**

---

## 🔑 Environment Variables

| Variable | Description | Required |
|----------|------------|----------|
| `GROQ_API_KEY` | Groq API key for LLM | Yes |
| `TAVILY_API_KEY` | Tavily API key for web search | Yes (Outreach only) |
| `EMAIL_ADDRESS` | Gmail address | Yes |
| `EMAIL_PASSWORD` | Gmail App Password | Yes |
| `SMTP_SERVER` | SMTP server | Yes (default: `smtp.gmail.com`) |
| `SMTP_PORT` | SMTP port | Yes (default: `587`) |
| `IMAP_SERVER` | IMAP server | Yes (default: `imap.gmail.com`) |

---

## 📋 Excel File Format

The HR Contacts Excel file should have these columns:

| Company Name | Email ID | Industry | Location |
|-------------|----------|----------|----------|
| Google | hr@google.com | Technology | Mountain View, CA |
| Microsoft | careers@microsoft.com | Technology | Redmond, WA |

---

## 📚 Documentation

For a deep dive into how each agent works internally — every node, edge, state field, LLM prompt, and execution flow — see:

👉 **[AGENTS_DOCUMENTATION.md](AGENTS_DOCUMENTATION.md)**

This documentation is designed to be comprehensive enough to explain the entire system architecture in an interview setting.

---


## 📝 License

This project is open source under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [LangGraph](https://github.com/langchain-ai/langgraph) — Agent orchestration framework
- [Groq](https://groq.com/) — Free, fast LLM inference
- [Tavily](https://tavily.com/) — AI-powered web search API
- [Streamlit](https://streamlit.io/) — Python web app framework
#   I n t e l i M a i l  
 #   I n t e l i M a i l  
 
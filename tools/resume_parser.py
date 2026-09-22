"""
tools/resume_parser.py
──────────────────────
Extract text from a resume PDF and produce a structured LLM summary
for email personalisation.
"""

from pypdf import PdfReader
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


def extract_text_from_pdf(pdf_path: str) -> str:
    """Read every page of a PDF and return concatenated text."""
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def summarize_resume(raw_text: str, groq_api_key: str) -> str:
    """Send raw resume text to Groq LLM and get a structured summary."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.3,
        max_tokens=600,
    )

    system_prompt = (
        "You are a resume-analysis assistant. Given the raw text of a resume, "
        "produce a detailed summary covering:\n"
        "1. Full name\n"
        "2. Key technical skills (Full Stack + AI/ML)\n"
        "3. ALL projects with 1-line descriptions\n"
        "4. Education\n"
        "5. Notable achievements or certifications\n\n"
        "Keep it under 300 words. Output ONLY the summary."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Resume text:\n\n{raw_text}"),
    ])
    return response.content.strip()


def parse_resume(pdf_path: str, groq_api_key: str) -> dict:
    """
    High-level: extract text from PDF -> summarise with LLM.

    Returns: {"raw_text": str, "summary": str}
    """
    raw_text = extract_text_from_pdf(pdf_path)
    summary = summarize_resume(raw_text, groq_api_key)
    return {"raw_text": raw_text, "summary": summary}

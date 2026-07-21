"""
Applikasi streamlit chatbot

Cara jalankan:
>>> streamlit run app2.py
"""
from dotenv import load_dotenv
import os

load_dotenv()

from pathlib import Path
import streamlit as st

from rag.loader import load_document
from rag.chunking import chunk_documents
from rag.pipeline import build_knowledge_base
from rag.retrieval import get_retriever
from rag.chat import ask_question

from langchain.messages import AIMessage
from langchain_core.messages import HumanMessage, SystemMessage
###from langchain_google_genai import ChatGoogleGenerativeAI

# =========================
# PAGE CONFIG (WAJIB PALING AWAL)
# =========================
st.set_page_config(
    page_title="Product Owner AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# HEADER
# =========================
st.title("📦 IT Solution Lead Tools")
st.caption("Enterprise Requirement & Documentation Assistant")

# =========================
# CREATE UPLOAD DIRECTORY
# =========================
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:

    st.header("📂 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=[
            "pdf",
            "docx",
            "xlsx",
            "csv",
            "pptx",
            "txt",
            "png",
            "jpg",
            "jpeg",
        ],
        accept_multiple_files=True,
    )
    st.divider()

    # ===========================
    #   Build Knowledge Base
    # ===========================

    if st.button("⚙️ Process Knowledge Base", use_container_width=True):

        if not uploaded_files:
            st.warning("⚠️ Silakan upload minimal satu dokumen.")
            st.stop()

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

        if not api_key:
            st.error("""
❌ Google Gemini API Key belum dikonfigurasi.

Silakan tambahkan salah satu environment variable berikut:

• GOOGLE_API_KEY
atau
• GEMINI_API_KEY

Setelah itu restart aplikasi Streamlit.
""")
            st.stop()

        file_paths = [
            str(UPLOAD_DIR / file.name)
            for file in uploaded_files
        ]

        total_chunks = build_knowledge_base(file_paths)

        st.success(
            f"✅ Knowledge Base berhasil dibuat!\n\nTotal Chunk: {total_chunks}"
        )


### client = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

### Kolom Chat ###
# Bikin chat history kosong jika belum ada
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        SystemMessage(
            """
            You are AI, an expert Senior Product Owner with more than 15 years of experience delivering enterprise software products.

You specialize in:
- Retail
- ERP
- Warehouse Management
- Supply Chain
- Finance
- POS
- CRM
- Omnichannel
- Mobile Applications
- Web Applications
- API Integration
- Data Warehouse
- AI Products

Your mission is to transform business ideas into well-structured, implementable product requirements.

--------------------------------------------------
WORKING PRINCIPLES
--------------------------------------------------

Always think before answering.

Never jump directly into a solution.

Follow this framework:

1. Understand Business Objective
2. Identify Stakeholders
3. Understand Current Process
4. Identify Pain Points
5. Perform Root Cause Analysis
6. Identify Constraints
7. Generate Solution Options
8. Recommend Best Solution
9. Analyze Risks
10. Define Functional Requirements
11. Define Non Functional Requirements
12. Define Acceptance Criteria
13. Consider Scalability
14. Consider Future Enhancements

Separate facts from assumptions.

If important information is missing, ask questions before producing documentation.

Never fabricate business rules.

--------------------------------------------------
WHEN ANALYZING A FEATURE
--------------------------------------------------

Always analyze:

Business Goal

Current Process

Pain Points

Root Cause

Stakeholders

User Journey

Business Flow

System Flow

Alternative Solutions

Dependencies

Risks

Edge Cases

Exception Handling

Security Impact

Performance Impact

Data Impact

Reporting Impact

Future Scalability

--------------------------------------------------
DOCUMENTS YOU CAN PRODUCE
--------------------------------------------------

Business Documents

- Business Requirement Document (BRD)
- Functional Specification Document (FSD)
- Software Requirement Specification (SRS)
- Product Requirement Document (PRD)
- Change Request (CR)
- Gap Analysis
- Business Process Flow
- As-Is Process
- To-Be Process
- SWOT Analysis
- Cost Benefit Analysis
- Feasibility Study
- Risk Assessment
- Impact Analysis
- Solution Proposal
- Scope Definition
- Out of Scope
- Release Notes

Agile Documents

- Epic
- Feature
- User Story
- Spike
- Task Breakdown
- Acceptance Criteria
- Definition of Ready
- Definition of Done
- Story Mapping
- Product Backlog
- Sprint Backlog
- Release Planning
- Roadmap
- Prioritization using MoSCoW
- Prioritization using RICE
- Prioritization using WSJF

Technical Documents

- API Specification
- API Contract
- Sequence Diagram
- Activity Diagram
- Flowchart
- ERD
- Database Design
- Integration Flow
- System Architecture
- Microservice Design
- Data Mapping
- Validation Rules
- Error Handling Matrix
- Business Rules
- Configuration Guide

Testing Documents

- Test Scenario
- Test Case
- UAT Scenario
- SIT Scenario
- Regression Checklist
- Smoke Test
- Test Data
- Defect Report
- Bug Report
- Bug Analysis
- RCA (Root Cause Analysis)

Project Documents

- Project Charter
- Project Scope
- RAID Log
- Risk Register
- Dependency List
- Meeting Minutes
- Decision Log
- Timeline
- Milestone Plan
- Deployment Plan
- Rollback Plan
- Go Live Checklist
- Hypercare Checklist

--------------------------------------------------
WHEN WRITING REQUIREMENTS
--------------------------------------------------

Requirements must always be:

Specific

Measurable

Testable

Unambiguous

Consistent

Traceable

Complete

Avoid vague words like:

Easy

Fast

Good

Flexible

User Friendly

Instead use measurable statements.

--------------------------------------------------
WHEN CREATING USER STORIES
--------------------------------------------------

Always use:

As a <Role>

I want <Capability>

So that <Business Value>

Acceptance Criteria using Given When Then.

--------------------------------------------------
WHEN CREATING ACCEPTANCE CRITERIA
--------------------------------------------------

Prefer Gherkin format.

Given ...

When ...

Then ...

Cover:

Happy Path

Validation

Negative Case

Boundary Case

Exception Case

--------------------------------------------------
COMMUNICATION STYLE
--------------------------------------------------

Be structured.

Use Markdown.

Use tables when appropriate.

Use bullets for requirements.

Highlight assumptions.

Highlight risks.

Provide recommendations with clear reasoning.

Think like a Senior Product Owner working with Enterprise Engineering Teams.
            
"""     
        )
    ]

# Welcome Screen
if len(st.session_state["chat_history"]) == 1:

    with st.chat_message("AI"):

        st.markdown("""
# 👋 Welcome to Product Owner AI

I can help you analyze and create:

- 📄 Business Requirement Document (BRD)
- 📑 Product Requirement Document (PRD)
- 📘 Software Requirement Specification (SRS)
- 📚 Functional Specification Document (FSD)
- 👤 User Stories & Acceptance Criteria
- 🔗 API Documentation
- 🔄 Business Process
- 🏗️ System Design
- 📊 Gap Analysis
- 📝 Technical Documentation

### Two ways to use me

### 💬 General AI Assistant
Ask me anything about Product Management, Business Analysis, or Software Development.

### 📂 Knowledge Base Mode
Upload one or more documents, process the Knowledge Base, then ask questions specifically about those documents.
""")

# Tampilkan chat history yang ada selama ini
for chat in st.session_state["chat_history"]:

    if isinstance(chat, SystemMessage):
        continue

    elif isinstance(chat, HumanMessage):
        role = "User"

    else:
        role = "AI"

    with st.chat_message(role):
        st.markdown(chat.content)


# Minta prompt dari user
user_prompt = st.chat_input(
    "How can I help you today as Product Owner?"
)

if user_prompt:

    # Simpan pertanyaan user ke history
    st.session_state["chat_history"].append(
        HumanMessage(content=user_prompt)
    )

    # Tampilkan pertanyaan user
    with st.chat_message("User"):
        st.markdown(user_prompt)

    # Tanya ke AI
    try:

        with st.spinner("🤖 Thinking..."):

            result = ask_question(user_prompt)

            answer = result["answer"]
            sources = result["sources"]

    except Exception as e:

        if "503" in str(e):

            st.warning(
                "⚠️ Gemini is currently experiencing high demand. Please try again in a few moments."
            )

            st.stop()

        raise

    # Simpan jawaban AI ke history
    st.session_state["chat_history"].append(
        AIMessage(content=answer)
    )

    # Tampilkan jawaban AI
    with st.chat_message("AI"):

        st.markdown(answer)

        if sources:

            with st.expander("📄 Sources"):

                for source in sources:
                    st.markdown(f"- {source}")
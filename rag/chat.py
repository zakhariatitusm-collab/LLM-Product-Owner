import os
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

from rag.prompt import get_rag_prompt
from rag.retrieval import get_retriever


def ask_question(question):

    # ==========================
    # Default (General AI Mode)
    # ==========================
    context = ""
    docs = []
    sources = []

    # ==========================
    # Knowledge Base Mode
    # ==========================
    if (
        os.path.exists("vectorstore/index.faiss")
        and os.path.exists("vectorstore/index.pkl")
    ):

        retriever = get_retriever()

        docs = retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        for doc in docs:

            source = Path(
                doc.metadata.get("source", "Unknown")
            ).name

            if source not in sources:
                sources.append(source)

    # ==========================
    # LLM
    # ==========================
    prompt = get_rag_prompt()

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "answer": answer,
        "sources": sources
    }
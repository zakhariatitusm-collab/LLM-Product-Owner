import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

from rag.retrieval import get_retriever
from rag.prompt import get_rag_prompt


def ask_question(question):

    retriever = get_retriever()

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = get_rag_prompt()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer
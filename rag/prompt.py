from langchain_core.prompts import ChatPromptTemplate


def get_rag_prompt():

    prompt = ChatPromptTemplate.from_template(
        """
Kamu adalah AI Assistant.

Jawablah pertanyaan user HANYA berdasarkan informasi pada context.

Jika jawabannya tidak ada di context, katakan:

"Maaf, saya tidak menemukan informasi tersebut di dokumen."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    return prompt
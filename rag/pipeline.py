from langchain_community.vectorstores import FAISS

from rag.loader import load_document
from rag.chunking import chunk_documents
from rag.embedding import get_embedding_model


def build_knowledge_base(file_paths):

    all_documents = []

    for file_path in file_paths:

        documents = load_document(file_path)

        all_documents.extend(documents)

    chunks = chunk_documents(all_documents)

    embeddings = get_embedding_model()

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vectorstore.save_local("vectorstore")

    return len(chunks)
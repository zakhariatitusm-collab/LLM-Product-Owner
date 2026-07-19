from langchain_community.vectorstores import FAISS

from rag.embedding import get_embedding_model


def get_retriever():

    embeddings = get_embedding_model()

    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    return retriever
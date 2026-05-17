from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def get_vectordb():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )


def search_protocol(query: str):
    """
    Searches the Chroma vector database for relevant documents based on the input query.

    Args:
    ----
        query (str): The search query to find relevant documents.

    Returns:
    -------
        A string containing the concatenated content of the most relevant documents found.

    """
    vectordb = get_vectordb()
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    docs = retriever.invoke(query)

    return "\n\n".join(f"Fonte: {d.metadata.get('source', 'desconhecida')}\n{d.page_content}"for d in docs)

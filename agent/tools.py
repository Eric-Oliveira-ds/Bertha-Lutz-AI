from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-base")
vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)


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
    docs = vectordb.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

os.getenv("OPENAI_API_KEY")


def ingest(pdf_path: str):
    """
    Ingest a PDF document, split it into chunks, and store the chunks in a Chroma vector db.

    Args:
    ----
        pdf_path (str): The path to the PDF document to be ingested.

    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
        )
    vectordb.add_documents(chunks)


if __name__ == "__main__":
    ingest(r'pdf\Consensointegra.pdf')
    ingest(r'pdf\relatorio-preliminar-diretrizes-brasileiras-para-o-rastreamento-do-cancer-do-colo-do-utero-parte-i-rastreamento-organizado-utilizando-testes-moleculares-para-deteccao-de-dna-hpv-oncogenico.pdf')
    ingest(r'pdf\Manual da Gestante.pdf')
    ingest(r'pdf\femina-2019-474-241-244.pdf')
    ingest(r'pdf\infeccoes_sexualmente_transmissiveis.pdf')
    ingest(r'pdf\manual_atencao_mulher_climaterio.pdf')
    ingest(r'pdf\manual_suplementacao_ferro_condutas_gerais.pdf')
    ingest(r'pdf\pcdt_endometriose_2016-1.pdf')
    ingest(r'pdf\saude_sexual_saude_reprodutiva.pdf')

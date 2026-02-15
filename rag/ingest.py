from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


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

    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-base")

    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    vectordb.persist()


ingest(r'pdf\Consensointegra.pdf')
ingest(r'pdf\relatorio-preliminar-diretrizes-brasileiras-para-o-rastreamento-do-cancer-do-colo-do-utero-parte-i-rastreamento-organizado-utilizando-testes-moleculares-para-deteccao-de-dna-hpv-oncogenico.pdf')
ingest(r'pdf\Manual da Gestante.pdf')
ingest(r'pdf\femina-2019-474-241-244.pdf')
ingest(r'pdf\infeccoes_sexualmente_transmissiveis.pdf')
ingest(r'pdf\manual_atencao_mulher_climaterio.pdf')
ingest(r'pdf\manual_suplementacao_ferro_condutas_gerais.pdf')
ingest(r'pdf\pcdt_endometriose_2016-1.pdf')
ingest(r'pdf\saude_sexual_saude_reprodutiva.pdf')

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from ingestion.load_documents import load_documents
from ingestion.text_splitter import split_documents
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

DATA_DIR = "data/raw_data"
VECTORSTORE_DIR = "vectorstore/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_faiss():
    print("Loading documents...")
    docs = load_documents(DATA_DIR)

    print("Splitting documents...")
    chunks = split_documents(docs)

    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    Path(VECTORSTORE_DIR).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)

    print(f"FAISS index saved to {VECTORSTORE_DIR}")
    print(f"Total chunks indexed: {len(chunks)}")


if __name__ == "__main__":
    build_faiss()

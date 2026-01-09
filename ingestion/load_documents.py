from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document

from .excel_loader import load_excel


def load_documents(data_dir: str) -> List[Document]:
    """
    Load documents from PDF, DOCX, and Excel files.
    Compatible with LangChain 0.3.x
    """
    documents: List[Document] = []
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for path in data_path.rglob("*"):
        if not path.is_file():
            continue

        try:
            if path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(path))
                docs = loader.load()

            elif path.suffix.lower() == ".docx":
                loader = Docx2txtLoader(str(path))
                docs = loader.load()

            elif path.suffix.lower() in {".xlsx", ".xls"}:
                docs = load_excel(str(path))
                if not isinstance(docs, list):
                    raise TypeError("Excel loader must return List[Document]")

            else:
                continue

            for d in docs:
                d.metadata.update({
                    "source": path.name,
                    "path": str(path),
                    "file_type": path.suffix.lower(),
                })

            documents.extend(docs)
            print(f" Loaded {len(docs)} documents from {path.name}")

        except Exception as e:
            print(f" Error loading {path.name}: {e}")

    return documents

from ingestion.load_documents import load_documents
from ingestion.text_splitter import split_documents

docs = load_documents("data/raw_data")
chunks = split_documents(docs)

print("Sample chunk:")
print(chunks[0].page_content[:300])
print(chunks[0].metadata)

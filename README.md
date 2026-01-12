# Retrievia — Grounded RAG Assistant

I built this RAG (Retrieval-Augmented Generation) system to pull relevant info from different document types and generate answers that are in the source material. 

---

## Check It Out Live

- **Azure App:** https://rag-chatbot-sharang-f9b0d6fhbaa7cxhw.westeurope-01.azurewebsites.net/


---

## Evaluation & Testing

### User Queries
- **User_Queries**  
  Holds the list of user queries used for testing the RAG system. 

### Evaluation Results
- **Evaluation_results**  
    Including system responses as well as metrics of evaluation for each query in the test.

**Links to User Queries and Evaluation Results :**
- Evaluation output: https://ragresultsweu01.blob.core.windows.net/evaluation-results/Evaluation_results.csv?sp=r&st=2026-01-12T10:02:36Z&se=2026-01-12T18:17:36Z&spr=https&sv=2024-11-04&sr=b&sig=UYtdLl44TJ%2FVsldu1SSiefyQ9%2FrTHb63P%2BTVTeUfg5g%3D

- User queries: https://ragresultsweu01.blob.core.windows.net/evaluation-results/User_Queries.xlsx?sp=r&st=2026-01-12T10:02:14Z&se=2026-01-12T18:17:14Z&spr=https&sv=2024-11-04&sr=b&sig=QD5O7DtBxig1ZRMrRCozksUTsrVAQh8rzovv3J1ma9A%3D

---
## Architecture Overview

```text
Data Ingestion (PDF, DOCX, XLSX)
        ↓
Cleaning and normalization
Chunk size: 500 characters
Chunk overlap: 150 characters
        ↓
Embedding Generation
  - all-MiniLM-L6-v2 (384 dimensions)
        ↓
Vector Store
  - FAISS
        ↓
Query Processing
  - Query embedding and similarity search
        ↓
Generation Layer
  - LangChain + Groq (LLaMA 3.3 70B)
  - Strict context grounding
        ↓
Web Interface & Deployment
  - Streamlit UI
  - Dockerized and deployed on Azure App Service
  - Evaluation documents stored in Azure Blob Storage
```
---

## How It Works

Here's the basic flow of what happens when you ask a question:

**Getting the Documents Ready**
I fed the system three different document types to work with:
- A PDF of the famous "Attention Is All You Need" paper and Deepseek-r1
- A DOCX file containing the EU AI Act
- An Excel sheet with CPI/inflation data

The text gets cleaned up and split into manageable chunks that the system can actually work with.

**Creating the Search Index**
Each chunk gets converted into an embedding .These embeddings go into a FAISS vector database.

**Finding Relevant Stuff**
When you ask a question, it gets converted into the embedding. The system then finds the chunks that are most similar to your question and provides you with the answer

**Generating Answers**
The retrieved context goes to an LLM with strict instructions: only answer based on what you found in the documents. If the answer isn't in there, say so. This helps avoid the classic AI problem of confidently making things up.

**The Interface**
Built a Streamlit app to keep things simple, featuring a sidebar to display the data sources. It's deployed on Azure so anyone can access it.

---

## Why I Made These Choices

**FAISS for storage** - It's fast, relatively simple to set up, and works well both locally and in the cloud. 

**MiniLM embeddings** - 384 Dimensions - A good balance between quality and speed. Doesn't require massive computing power.

**Strict grounding rules** - I'd rather have the system say "I don't know" than make something up. Trust is more important than always having an answer.

**Deterministic CPI calculations** - For the inflation queries, I built in specific logic to handle the math directly rather than hoping the LLM would get the numbers right. Numbers need to be exact.

---

## Trade-offs

**Being Conservative vs. Being Helpful**
The system will refuse to answer if it doesn't have good source material. This means sometimes it won't answer questions it theoretically could, but it also means you can trust the answers you do get.

**Chunk Size and How Many to Retrieve**
I had to pick a chunk size and decide how many relevant chunks to pull for each query (the "Top-K" parameter). Bigger K means better chance of finding what you need, bigger K means it will be slow

**Special Handling for CPI Data**
Instead of letting the LLM try to do inflation calculations, I wrote code to handle those queries. More work, but guaranteed accuracy.

**One Big Index vs. Multiple Indexes**
Everything goes into one FAISS index rather than separating by document type. Simpler to maintain, though it means you can't easily filter by source.

---

## Testing & What I Learned

I tested with 8 different questions covering various scenarios. Here's what I found:

- When the system finds relevant context, the answers are solid.
- Out-of-scope questions get properly refused - no hallucinations.
- Policy and conceptual questions worked well.
- Some  cases around formatting (like currency symbols) need better handling.

---


## Limitations:
- Retrieval quality depends a lot on how documents are chunked
- No reranking step to refine what chunks get used
- Have to rebuild the entire index if you want to add new documents

## If I Had More Time:
- Add a reranker to improve precision
- Maybe experiment with different chunking strategies
- Add Memory for better context undersatnding 

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```



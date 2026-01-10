# Retrievia — Grounded RAG Assistant

I built this RAG (Retrieval-Augmented Generation) system to pull relevant info from different document types and generate answers that are in the source material. 

---

## Check It Out Live

- **Azure App:** https://rag-chatbot-sharang-f9b0d6fhbaa7cxhw.westeurope-01.azurewebsites.net/

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

**MiniLM embeddings** - Good balance between quality and speed. Doesn't require massive compute resources and gives good results.

**Strict grounding rules** - I'd rather have the system say "I don't know" than make something up. Trust is more important than always having an answer.

**Deterministic CPI calculations** - For the inflation queries, I built in specific logic to handle the math directly rather than hoping the LLM would get the numbers right. Numbers need to be exact.

---

## Trade-offs I Had to Make

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


**Current Limitations:**
- Retrieval quality depends a lot on how documents are chunked
- No reranking step to refine what chunks get used
- Have to rebuild the entire index if you want to add new documents

**If I Had More Time:**
- Add a reranker to improve precision
- Build better input cleaning/normalization
- Maybe experiment with different chunking strategies
- Add Memory for better context undersatnding 

---

## Running This Yourself

```bash
pip install -r requirements.txt
streamlit run app.py
```



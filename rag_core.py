import re
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
FAISS_PATH = BASE_DIR / "vectorstore" / "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MONTH_MAP = {
    "january": "Jan", "february": "Feb", "march": "Mar",
    "april": "Apr", "may": "May", "june": "Jun",
    "july": "Jul", "august": "Aug", "september": "Sep",
    "october": "Oct", "november": "Nov", "december": "Dec"
}


def load_cpi_dataframe():
    file_path = BASE_DIR / "data" / "raw_data" / "Inflation Calculator (2).xlsx"
    raw_df = pd.read_excel(file_path, header=None)

    header_row = next(
        i for i in range(len(raw_df))
        if raw_df.iloc[i].astype(str).str.strip().eq("Year").any()
    )

    df = pd.read_excel(file_path, header=header_row)
    df["Year"] = df["Year"].astype(int)
    return df


def load_rag_components():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    return retriever, llm


retriever, llm = load_rag_components()

PROMPT = ChatPromptTemplate.from_template(
   """
    You are a Retrieval-Augmented Generation assistant.

    Use ONLY the provided context to answer the question.
    If relevant context is provided, synthesize a clear, concise, and factual answer
    based strictly on that context.

    Do NOT add external knowledge.
    Do NOT make assumptions beyond the context.

    ONLY respond with:
    "I cannot answer this based on the provided information."
    if NO relevant context is provided at all.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

parser = StrOutputParser()


def rag_answer(question: str):
    df = load_cpi_dataframe()

    for month, col in MONTH_MAP.items():
        if month in question.lower():
            years = list(map(int, re.findall(r"\b(?:19|20)\d{2}\b", question)))
            if years:
                row = df[df["Year"] == years[0]]
                if not row.empty and col in row.columns:
                    return f"The CPI in {month.title()} {years[0]} was {row.iloc[0][col]}.", []

    docs = retriever.invoke(question)

    if not docs:
        return "I cannot answer this based on the provided information.", []

    context = "\n\n".join(d.page_content for d in docs)

    answer = (PROMPT | llm | parser).invoke(
        {"context": context, "question": question}
    )

    sources = [
        f"{d.metadata.get('source')} | page {d.metadata.get('page', 'N/A')}"
        for d in docs
    ]

    return answer, sources

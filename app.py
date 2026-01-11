import os
import sys
import re
from pathlib import Path
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()
print("=== ENV DEBUG ===")
print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))


BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))



GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment")

FAISS_PATH = os.getenv(
    "FAISS_PATH",
    str(BASE_DIR / "vectorstore" / "faiss_index")
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"



st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide"
)

st.title(" Retrievia — Grounded RAG Assistant")



with st.sidebar:
    st.header("Knowledge Sources")
    st.markdown(
        """
        Documents used:
        - Attention_is_all_you_need.pdf  
        - Deepseek-r1.pdf  
        - EU AI Act Doc.docx  
        - Inflation Calculator (2).xlsx
        """
    )



@st.cache_resource
def load_cpi_dataframe():
    file_path = BASE_DIR / "data" / "raw_data" / "Inflation Calculator (2).xlsx"

    raw_df = pd.read_excel(file_path, header=None)

    header_row = None
    for i in range(len(raw_df)):
        if raw_df.iloc[i].astype(str).str.strip().eq("Year").any():
            header_row = i
            break

    if header_row is None:
        raise ValueError("Year column not found in CPI file")

    df = pd.read_excel(file_path, header=header_row)
    df.columns = df.columns.astype(str).str.strip()
    df = df[df["Year"].notna()]
    df["Year"] = df["Year"].astype(int)

    return df


@st.cache_resource
def load_cpi_lookup():
    df = load_cpi_dataframe()
    return dict(zip(df["Year"], df["Average"]))


MONTH_MAP = {
    "january": "Jan", "february": "Feb", "march": "Mar",
    "april": "Apr", "may": "May", "june": "Jun",
    "july": "Jul", "august": "Aug", "september": "Sep",
    "october": "Oct", "november": "Nov", "december": "Dec"
}


def is_calculation_question(question: str) -> bool:
    keywords = ["worth", "calculate", "inflation adjusted", "value in"]
    return any(k in question.lower() for k in keywords)


def calculate_cpi_adjusted_value(base_value, base_year, target_year, cpi_lookup):
    if base_year not in cpi_lookup or target_year not in cpi_lookup:
        return None
    return round(base_value * (cpi_lookup[target_year] / cpi_lookup[base_year]), 2)



@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


@st.cache_resource
def load_rag_components():
    embeddings = load_embeddings()

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



answer_prompt = ChatPromptTemplate.from_template(
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

if "messages" not in st.session_state:
    st.session_state.messages = []



def format_docs_with_sources(docs):
    context_chunks = []
    sources = []

    for doc in docs:
        context_chunks.append(doc.page_content)
        meta = doc.metadata
        sources.append(
            f"{meta.get('source')} | page {meta.get('page_label', meta.get('page', 'N/A'))}"
        )

    return "\n\n".join(context_chunks), sorted(set(sources))



def rag_answer(question: str):
    df = load_cpi_dataframe()

    # CPI month lookup
    for month, col in MONTH_MAP.items():
        if month in question.lower():
            years = list(map(int, re.findall(r"\b(?:19|20)\d{2}\b", question)))
            if years:
                row = df[df["Year"] == years[0]]
                if not row.empty and col in row.columns:
                    return f"The CPI in {month.title()} {years[0]} was {row.iloc[0][col]}.", []

    # CPI calculation
    if is_calculation_question(question):
        cpi_lookup = load_cpi_lookup()
        values = list(map(float, re.findall(r"\d+\.?\d*", question)))
        years = list(map(int, re.findall(r"\b(?:19|20)\d{2}\b", question)))

        if len(values) >= 1 and len(years) >= 2:
            result = calculate_cpi_adjusted_value(
                values[0], years[0], years[1], cpi_lookup
            )
            if result is not None:
                return f"${values[0]} in {years[0]} is worth ${result} in {years[1]}.", []

        return "I cannot calculate this with the available data.", []

 
    docs = retriever.invoke(question)

    if not docs:
        return "I cannot answer this based on the provided information.", []

    context, sources = format_docs_with_sources(docs)

    answer = (
        answer_prompt | llm | parser
    ).invoke(
        {"context": context, "question": question}
    )

    return answer, sources


##UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about the documents..."):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    answer, sources = rag_answer(question)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- {s}")

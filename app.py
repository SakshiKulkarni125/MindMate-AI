import streamlit as st
import os

from dotenv import load_dotenv
from openai import OpenAI

# PDF Loader
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vector Database
from langchain_community.vectorstores import FAISS

# -----------------------------------
# LOAD ENV VARIABLES
# -----------------------------------

load_dotenv()

# -----------------------------------
# API CONFIG
# -----------------------------------

api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("MODEL")

# Check if API key exists
if not api_key:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()

# -----------------------------------
# GROQ CLIENT
# -----------------------------------

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

# -----------------------------------
# PAGE SETTINGS
# -----------------------------------

st.set_page_config(page_title="MindMate AI")

st.title("🧠 MindMate AI")

st.write("Mental Health Awareness Assistant")

st.info(
    "This assistant provides mental health awareness information "
    "and is not a replacement for professional medical advice."
)

# -----------------------------------
# LOAD PDFs
# -----------------------------------

with st.spinner("Loading PDFs..."):

    loader = PyPDFDirectoryLoader("data")

    docs = loader.load()

st.success(f"Loaded {len(docs)} pages from PDFs")

# -----------------------------------
# SPLIT TEXT
# -----------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(docs)

st.success(f"Created {len(chunks)} text chunks")

# -----------------------------------
# CREATE EMBEDDINGS
# -----------------------------------

with st.spinner("Creating embeddings..."):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

st.success("Embeddings model loaded")

# -----------------------------------
# CREATE VECTOR DATABASE
# -----------------------------------

with st.spinner("Creating vector database..."):

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

st.success("FAISS vector database ready")

# -----------------------------------
# CREATE RETRIEVER
# -----------------------------------

retriever = vectorstore.as_retriever()

# -----------------------------------
# USER INPUT
# -----------------------------------

question = st.text_input(
    "Ask a question about stress, anxiety, burnout, wellness..."
)

# -----------------------------------
# QUESTION ANSWERING
# -----------------------------------

if question:

    with st.spinner("Thinking..."):

        # Retrieve relevant chunks
        relevant_docs = retriever.invoke(question)

        # Combine retrieved text
        context = "\n\n".join(
            [doc.page_content for doc in relevant_docs]
        )

        # Prompt
        prompt = f"""
        You are a helpful mental health awareness assistant.

        Use ONLY the provided context to answer.

        Context:
        {context}

        Question:
        {question}

        Give a supportive, concise, easy-to-understand answer.
        """

        # Send to model
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        # Extract answer
        answer = response.choices[0].message.content

    # Display answer
    st.subheader("Answer")

    st.write(answer)
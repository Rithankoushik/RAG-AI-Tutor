import os
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings

from langchain_ollama.llms import OllamaLLM    # <- corrected import path
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

from get_embedding_function import get_embedding_function
from PyPDF2 import PdfReader
from langchain_community.vectorstores import Chroma
import chromadb
# DB Path
CHROMA_PATH = "chroma_store"
embedding_function = get_embedding_function()

# Chroma DB (auto-persistent)
db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_function
)

# Prompt template
PROMPT_TEMPLATE = """
You are a fun, Gen-Z style teacher who explains concepts in a chill, engaging, and easy-to-understand way.
Your student is preparing for exams, so be precise and give **correct information from the provided context**.
Your goal is to make students understand the concept clearly like explain to their friend.
Add some emoji and make the content fun to read
In output don't show the <think></think> part

Format the answer in two parts:
1 **2 Mark Answer (quick, crisp 4-5 lines)**  
   - Give the core definition/key point. Keep it short and exam-friendly.

2 **16 Mark Answer (detailed explanation)**  
   - Break into sub-topics if needed.  
   - Give examples, diagrams (described in words), and analogies.  
   - Make it **fun and engaging** like a Gen-Z teacher.  
   - Keep it structured and organized for exam prep.  
Context: {context}

Question: {question}

Answer in the above format:
"""

def extract_text_from_file(uploaded_file):
    """Extract text from PDF or TXT"""
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    else:
        return None
def add_documents(texts: list[str]):
    """Add new documents into Chroma DB"""
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Convert texts into LangChain Documents
    docs = [Document(page_content=text, metadata={"id": f"doc_{i}"}) for i, text in enumerate(texts)]

    # Add docs to DB (auto-persist in langchain_chroma)
    db.add_documents(docs)

    return f"{len(docs)} documents added successfully ✅"


def query_rag(query_text: str):
    # Prepare DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search top-k docs
    results = db.similarity_search_with_score(query_text, k=3)

    # Build context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # LLM
    model = OllamaLLM(model="qwen3:1.7b", options={"num_predict": 800, "temperature": 0.3})
    response_text = model.invoke(prompt)

    # Sources
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    return response_text, sources


# Streamlit App
st.set_page_config(page_title="RAG AI-Chatbot", page_icon="📚", layout="centered")

st.title("📚 RAG-Chatbot")
st.markdown("Chat with your PDFs like a **Gen-Z Teacher**📚")

# Section 1: Upload Documents
st.subheader("📂 Upload New Docs")
uploaded_files = st.file_uploader("Upload your notes (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files and st.button("➕ Add to DB"):
    texts = []
    for uploaded_file in uploaded_files:
        text = extract_text_from_file(uploaded_file)
        if text:
            texts.append(text)
    if texts:
        msg = add_documents(texts)
        st.success(msg)
    else:
        st.error("Unsupported file format da makku ❌")

# Section 2: Ask Questions
st.subheader("💬 Ask Questions")
query_text = st.text_input("Ask me anything from your docs:", "")

if st.button("Ask"):
    if query_text.strip() == "":
        st.warning("Type something da lavdea 🤌")
    else:
        with st.spinner("Thinking... 🧠"):
            response, sources = query_rag(query_text)

        st.subheader("📝 Answer")
        st.write(response)

        st.subheader("📌 Sources")
        for src in sources:
            st.code(src, language="text")

import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate

from get_embedding_function import get_embedding_function

# DB Path
CHROMA_PATH = "chroma"

# Prompt template
PROMPT_TEMPLATE = """
You are a cool, Gen-Z style teacher who explains things in a fun, easy-to-understand way!
Your students are young learners who need clear, engaging explanations.

Use simple language, add some casual expressions, and make learning fun!
If you don't know something, be honest about it.

Context: {context}

Question: {question}

Answer in a Gen-Z teacher style:
"""

def query_rag(query_text: str):
    # Prepare DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search top-k docs
    results = db.similarity_search_with_score(query_text, k=5)

    # Build context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # LLM
    model = OllamaLLM(model="llama2")
    response_text = model.invoke(prompt)

    # Sources
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    return response_text, sources

# Streamlit App
st.set_page_config(page_title="RAG AI-Chatbot", page_icon="📚", layout="centered")

st.title("📚 RAG-Chatbot")
st.markdown("Chat with your PDFs like a **Gen-Z Teacher**📚")

# Text input
query_text = st.text_input("💬 Ask a question:", "")

if st.button("Ask"):
    if query_text.strip() == "":
        st.warning("Type something, da makku 🤌")
    else:
        with st.spinner("Thinking... 🧠"):
            response, sources = query_rag(query_text)

        st.subheader("📝 Answer")
        st.write(response)

        st.subheader("📌 Sources")
        for src in sources:
            st.code(src, language="text")

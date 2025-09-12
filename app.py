import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate

from get_embedding_function import get_embedding_function

# DB Path
CHROMA_PATH = "chroma"

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
    model = OllamaLLM(model="qwen3:1.7b",  options={"num_predict": 800, "temperature": 0.3}
)
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

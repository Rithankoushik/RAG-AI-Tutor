📚 RAG-Tutor: Gen-Z Style AI Teacher with Contextual Memory

A Retrieval-Augmented Generation (RAG) powered learning assistant that lets you chat with your PDFs like a cool teacher 😎. Built with LangChain, Chroma, and Ollama, this project creates a fun, Gen-Z style conversational agent that explains concepts in a chill, easy-to-understand way.

🚀 Features

🔍 Query PDFs directly with natural language questions.

🧠 Retrieval-Augmented Generation (RAG) for accurate, context-aware answers.

🤖 Ollama-powered LLMs for generating responses.

📂 Chroma vector database for efficient document storage and retrieval.

🎭 Custom Prompting for “Gen-Z Teacher” style explanations.

🖥️ Streamlit Web UI for a clean and interactive experience.

🛠️ Tech Stack

LangChain – RAG pipeline + chaining

ChromaDB – Vector database for embeddings

Ollama – Local LLM support (Llama2, Mistral, etc.)

Ollama Embeddings – For semantic document search

Streamlit – Web-based UI for querying PDFs

Python 3.10+

📂 Project Structure
rag-tutorial-v2/
│── data/                     # PDF files for knowledge base
│── chroma/                   # Persistent Chroma database
│── get_embedding_function.py # Embedding setup (Ollama)
│── populate_database.py      # Script to load & split PDFs
│── query_data.py             # CLI for querying PDFs
│── streamlit_app.py          # Web UI for RAG assistant
│── requirements.txt
│── README.md

⚡ Setup Instructions
1️⃣ Clone the repo
git clone https://github.com/your-username/rag-tutor.git
cd rag-tutor

2️⃣ Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Pull Ollama model (example: Llama2 + embeddings)
ollama pull llama2
ollama pull nomic-embed-text

5️⃣ Add your PDFs

Put your PDF files into the data/ folder.

6️⃣ Populate database
python populate_database.py --reset

7️⃣ Run query via CLI
python query_data.py "What are the rules of Ticket to Ride?"

8️⃣ Launch Streamlit Web UI
streamlit run streamlit_app.py

🎮 Demo Flow

Upload PDFs (e.g., board game rules, textbooks, research papers).

Populate the database with embeddings.

Ask natural language questions via CLI or Streamlit UI.

Get Gen-Z style answers with proper context + source references.

🔮 Future Improvements

✅ Add PDF upload feature directly in Streamlit.

✅ Multi-model support (switch between Llama2, Mistral, etc.).

✅ UI for source highlighting & document previews.

✅ Export Q&A sessions as study notes.

✨ Credits

Built with ❤️ using LangChain, Ollama, and Chroma. Inspired by the need for a fun, conversational learning assistant that doesn’t bore you to death.

👉 Bro, sollu — unga README fun swag style la vechuklama illa full corporate serious tone la venuma?
import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

# Friendly Teacher
PROMPT_TEMPLATE = """
You are a friendly teacher who explains concepts in a mix of English and Tamil (Tanglish).
Your goal is to make students understand the concept clearly, like how a caring teacher 
would explain to their classmate/friend.

Format the answer in two parts:
1. **2 Mark Answer (short definition or key point in 2-3 lines)**
2. **15 Mark Answer (detailed explanation in simple English + Tamil mix, with examples where possible)**

Rules:
- Keep the 2-mark answer crisp and direct.
- In the 15-mark answer, use easy English with Tamil phrases here and there, 
  so it feels natural (example: "This means basically... appo namakku puriyum...").
- Don't use too much technical jargon without explaining.
- If you don't know, say honestly "I'm not sure about this".

Context: {context}

Question: {question}

Answer in the above format:
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model="qwen3:1.7b")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()

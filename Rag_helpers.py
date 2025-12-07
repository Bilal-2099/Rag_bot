import random
import os
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("API_KEY")

if not API_KEY:
    raise ValueError("API Key not found! Check your .env file.")

genai.configure(api_key=API_KEY)

EMBED_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-2.5-flash" 

def load_notes(folder="My_notes"):
    """Loads all .txt files from the specified folder."""
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder '{folder}'. Please add .txt files there!")
        return []

    docs = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if text: 
                        docs.append({"file": filename, "text": text})
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return docs

def embed_text(text, task_type="retrieval_query"):
    """Corrected function using genai.embed_content"""
    result = genai.embed_content(
        model=EMBED_MODEL,  
        content=text,       
        task_type=task_type 
    )
    return np.array(result['embedding'])

def build_index(docs):
    """Generates embeddings for all documents."""
    print(f"Indexing {len(docs)} documents...")
    for doc in docs:
        doc["embedding"] = embed_text(doc["text"], task_type="retrieval_document")
    return docs


def cosine_sim(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_context(query, docs, top_k=2):
    if not docs:
        return []
    
    query_embed = embed_text(query, task_type="retrieval_query")
    scored = []

    for doc in docs:
        score = cosine_sim(query_embed, doc["embedding"])
        scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored[:top_k]]

def format_history(history):
    """Formats the chat history into a string."""
    text = ""
    for turn in history:
        text += f"User: {turn['user']}\n"
        text += f"AI: {turn['ai']}\n\n"
    return text

def ask_gemini(question, context_docs, chat_history):
    """Generates an answer using Context + Chat History."""
    
    context_text = "\n\n---\n\n".join(
        [f"From {doc['file']}:\n{doc['text']}" for doc in context_docs]
    )

    history_text = format_history(chat_history)

    prompt = f"""
    You are a helpful assistant. Answer the user's question using the provided context and conversation history.
    If the answer is not in the context, say "I don't know based on the provided notes."

    ---
    Chat History (Past Conversation):
    {history_text}
    ---
    
    Context (Retrieved Notes):
    {context_text}
    
    ---
    Current Question: {question}

    Answer:
    """

    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt)
    return response.text

def generate_question(context_docs):
    context_text = "\n\n---\n\n".join([doc["text"] for doc in context_docs])

    prompt = f"""
    You are a tutor. Create one question based ONLY on the following notes:

    {context_text}

    The question must be short and clear.
    Don't include the answer.
    """

    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()

def check_answer(user_answer, context_docs, question):
    context_text = "\n\n---\n\n".join([doc["text"] for doc in context_docs])

    prompt = f"""
    You're a tutor checking the student's answer.

    Question: {question}
    Student Answer: {user_answer}

    Using ONLY the notes below, decide:
    1. If the student is correct or not.
    2. If incorrect, explain the correct answer clearly.

    Notes:
    {context_text}

    Respond in this format:
    - Correct or Incorrect
    - Short explanation
    """

    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()

def main():
    print("Loading documents...")
    docs = load_notes()
    
    if not docs:
        print("No documents found. Exiting.")
        return

    docs = build_index(docs)
    
    chat_history = []
    
    print("RAG chatbot ready. Type 'exit' to quit.\n")

    while True:
        mode = input("Choose mode: (1) Ask AI | (2) Quiz Me: ").strip()

        if mode == "2":
            while True:
                context = [random.choice(docs)]

                question = generate_question(context)
                print(f"\nAI Question: {question}")

                user_answer = input("Your Answer: ").strip()

                if user_answer.lower() in ["exit", "quit"]:
                    break

                result = check_answer(user_answer, context, question)
                print("\n", result)
                print("-" * 40)
        else:
            try:
                question = input("\nYou: ").strip()
                if not question: 
                    continue
                    
                if question.lower() in ["exit", "quit"]:
                    break

                print("Searching notes...")
                context = search_context(question, docs)
                
                print("Thinking...")
                answer = ask_gemini(question, context, chat_history)

                print(f"\nAI: {answer}")
                print("-" * 30)

                chat_history.append({"user": question, "ai": answer})
                
                if len(chat_history) > 5:
                    chat_history.pop(0)
            except Exception as e:
                print(f"\nAn error occurred: {e}")



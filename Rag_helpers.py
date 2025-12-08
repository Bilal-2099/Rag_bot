import pdfplumber
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

def chunk_text(text, chunk_size=1000, overlap=200):
    """Splits long text into smaller overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap) # Move forward, keeping some overlap
    return chunks

def load_notes(folder="My_notes"):
    """Loads files and splits them into chunks."""
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder '{folder}'. Please add .txt or .pdf files there!")
        return []

    docs = []
    print(f"Reading files from {folder}...")
    
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        file_text = ""
        
        try:
            if filename.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    file_text = f.read().strip()
            
            elif filename.endswith(".pdf"):
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text: # Check if text exists to avoid NoneType error
                            file_text += page_text + "\n"
            else:
                continue

            # If we found text, chunk it
            if file_text:
                chunks = chunk_text(file_text)
                for i, chunk in enumerate(chunks):
                    docs.append({
                        "file": filename,
                        "chunk_id": i,
                        "text": chunk
                    })
                    
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
    return docs

def embed_text(text, task_type="retrieval_query"):
    """Generates embedding for a string."""
    try:
        result = genai.embed_content(
            model=EMBED_MODEL,  
            content=text,      
            task_type=task_type
        )
        return np.array(result['embedding'])
    except Exception as e:
        print(f"Embedding error: {e}")
        return np.zeros(768) # Return zero vector on failure to prevent crash

def build_index(docs):
    """Generates embeddings for all document chunks."""
    print(f"Indexing {len(docs)} text chunks (this may take a moment)...")
    for doc in docs:
        doc["embedding"] = embed_text(doc["text"], task_type="retrieval_document")
    return docs

def cosine_sim(a, b):
    """Calculates cosine similarity."""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_context(query, docs, top_k=3):
    """Finds the most relevant chunks."""
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
    text = ""
    for turn in history:
        text += f"User: {turn['user']}\nAI: {turn['ai']}\n\n"
    return text

def ask_gemini(question, context_docs, chat_history):
    context_text = "\n\n---\n\n".join(
        [f"From {doc['file']} (Part {doc['chunk_id']}):\n{doc['text']}" for doc in context_docs]
    )

    history_text = format_history(chat_history)

    prompt = f"""
    You are a helpful assistant. Answer the user's question using the provided context and conversation history.
    If the answer is not in the context, say "I don't know based on the provided notes."

    ---
    Chat History:
    {history_text}
    ---
    Context:
    {context_text}
    ---
    Question: {question}

    Answer:
    """
    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt)
    return response.text

def generate_question(context_docs):
    context_text = "\n\n---\n\n".join([doc["text"] for doc in context_docs])
    
    prompt = f"""
    You are a tutor. Create ONE short question based ONLY on this text:
    "{context_text}"
    Do not include the answer.
    """
    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()

def check_answer(user_answer, context_docs, question):
    context_text = "\n\n---\n\n".join([doc["text"] for doc in context_docs])
    
    prompt = f"""
    Question: {question}
    Student Answer: {user_answer}
    Context: {context_text}
    
    Is the student correct based on the context? 
    If incorrect, briefly explain why using the context.
    """
    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()

def main():
    print("Initializing RAG System...")
    docs = load_notes()
    
    if not docs:
        print("No documents found. Exiting.")
        return

    docs = build_index(docs)
    chat_history = []
    
    print("\nSystem Ready! Type 'exit' to quit.\n")

    while True:
        mode = input("Choose mode: (1) Ask AI | (2) Quiz Me: ").strip()

        if mode == "2":
            # Quiz Mode
            while True:
                # Pick a random chunk for a focused question
                context = [random.choice(docs)] 
                
                print("\nGenerating question...")
                question = generate_question(context)
                print(f"\nAI Question: {question}")

                user_answer = input("Your Answer: ").strip()
                if user_answer.lower() in ["exit", "quit"]:
                    break

                result = check_answer(user_answer, context, question)
                print(f"\nFeedback: {result}")
                print("-" * 40)
                
        else:
            # Chat Mode
            while True:
                question = input("\nYou: ").strip()
                if not question: continue
                if question.lower() in ["exit", "quit"]: break

                print("Searching notes...")
                context = search_context(question, docs, top_k=3)
                
                print("Thinking...")
                answer = ask_gemini(question, context, chat_history)

                print(f"\nAI: {answer}")
                print("-" * 30)

                chat_history.append({"user": question, "ai": answer})
                if len(chat_history) > 5:
                    chat_history.pop(0)

        if input("Back to main menu? (y/n): ").lower() != 'y':
            break

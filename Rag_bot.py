from API import API_KEY
import os
import google.generativeai as genai
import numpy as np

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
                    if text: # Only add if file is not empty
                        docs.append({"file": filename, "text": text})
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return docs

def embed_text(text):
    """
    Corrected function using genai.embed_content
    """
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_query"
    )
    return np.array(result['embedding'])

def build_index(docs):
    """Generates embeddings for all documents."""
    print(f"Indexing {len(docs)} documents...")
    for doc in docs:
        doc["embedding"] = embed_text(doc["text"])
    return docs

def cosine_sim(a, b):
    """Calculates cosine similarity between two vectors."""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_context(query, docs, top_k=2):
    """Finds the most relevant documents for the query."""
    if not docs:
        return []
        
    query_embed = embed_text(query)
    scored = []

    for doc in docs:
        score = cosine_sim(query_embed, doc["embedding"])
        scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])
    
    top_docs = [doc for _, doc in scored[:top_k]]
    return top_docs

def ask_gemini(question, context_docs):
    """Generates an answer using the retrieved context."""
    
    if not context_docs:
        return "I couldn't find any relevant notes to answer that."

    context_text = "\n\n---\n\n".join(
        [f"From {doc['file']}:\n{doc['text']}" for doc in context_docs]
    )

    prompt = f"""
    You are a helpful assistant. Use ONLY the provided context to answer the question.
    If the answer is not in the context, say "I don't know based on the provided notes."

    Context:
    {context_text}

    Question: {question}

    Answer:
    """

    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt)
    return response.text

def main():
    print("Loading documents...")
    docs = load_notes()
    
    if not docs:
        print("No documents found. Exiting.")
        return

    docs = build_index(docs)
    print("RAG chatbot ready. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("\nYou: ").strip()
            if not question: 
                continue
                
            if question.lower() in ["exit", "quit"]:
                break

            print("Searching notes...")
            context = search_context(question, docs)
            
            print("Thinking...")
            answer = ask_gemini(question, context)

            print(f"\nAI: {answer}")
            print("-" * 30)
            
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
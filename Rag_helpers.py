import random # For choosing a random note in quiz mode
import os # For file system operations (loading notes, creating directory)
import google.generativeai as genai # The Google GenAI SDK for LLM and embedding operations
import numpy as np # For numerical operations, especially vector math
from dotenv import load_dotenv # To load environment variables from .env file

load_dotenv() # Load environment variables
API_KEY = os.environ.get("API_KEY") # Get the API key

if not API_KEY:
    raise ValueError("API Key not found! Check your .env file.")

genai.configure(api_key=API_KEY) # Configure the GenAI client with the API key

EMBED_MODEL = "models/text-embedding-004" # Model name for text embeddings
LLM_MODEL = "gemini-2.5-flash" # Model name for large language model generation

def load_notes(folder="My_notes"):
    """Loads all .txt files from the specified folder."""
    if not os.path.exists(folder):
        os.makedirs(folder) # Create the directory if it doesn't exist
        print(f"Created folder '{folder}'. Please add .txt files there!")
        return []

    docs = [] # List to store document data (filename, text, embedding)
    for filename in os.listdir(folder):
        if filename.endswith(".txt"): # Process only text files
            path = os.path.join(folder, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip() # Read file content and remove surrounding whitespace
                    if text: 
                        docs.append({"file": filename, "text": text}) # Store filename and content
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return docs

def embed_text(text, task_type="retrieval_query"):
    """Corrected function using genai.embed_content"""
    # Use the SDK to generate the embedding vector
    result = genai.embed_content(
        model=EMBED_MODEL,  
        content=text,      
        task_type=task_type # Specify task type for better embedding quality
    )
    return np.array(result['embedding']) # Convert the list embedding to a numpy array

def build_index(docs):
    """Generates embeddings for all documents."""
    print(f"Indexing {len(docs)} documents...")
    for doc in docs:
        # Embed each document for retrieval (task_type="retrieval_document")
        doc["embedding"] = embed_text(doc["text"], task_type="retrieval_document") 
    return docs


def cosine_sim(a, b):
    """Calculates the cosine similarity between two numpy vectors."""
    # Handle zero vectors to prevent division by zero
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    # Formula for cosine similarity
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_context(query, docs, top_k=2):
    """Finds the most relevant documents for a given query using cosine similarity."""
    if not docs:
        return []
    
    # Embed the query (task_type="retrieval_query")
    query_embed = embed_text(query, task_type="retrieval_query")
    scored = []

    # Calculate similarity between query and all document embeddings
    for doc in docs:
        score = cosine_sim(query_embed, doc["embedding"])
        scored.append((score, doc))

    # Sort by score in descending order and return the top_k results
    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored[:top_k]]

def format_history(history):
    """Formats the chat history into a string for the LLM prompt."""
    text = ""
    for turn in history:
        text += f"User: {turn['user']}\n"
        text += f"AI: {turn['ai']}\n\n"
    return text

def ask_gemini(question, context_docs, chat_history):
    """Generates an answer using Context + Chat History (RAG implementation)."""
    
    # Format retrieved context documents into a single string
    context_text = "\n\n---\n\n".join(
        [f"From {doc['file']}:\n{doc['text']}" for doc in context_docs]
    )

    # Format chat history into a string
    history_text = format_history(chat_history)

    # The full prompt template for the LLM
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

    model = genai.GenerativeModel(LLM_MODEL) # Initialize the GenerativeModel
    response = model.generate_content(prompt) # Send the prompt to the model
    return response.text

def generate_question(context_docs):
    """Uses the LLM to generate a question based on the provided notes for quiz mode."""
    # Combine the notes into a single string
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
    """Uses the LLM to check a student's answer against the provided notes."""
    # Combine the notes into a single string
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
    docs = load_notes() # Load notes from the folder
    
    if not docs:
        print("No documents found. Exiting.")
        return

    docs = build_index(docs) # Build the index (generate embeddings)
    
    chat_history = [] # Initialize chat history list
    
    print("RAG chatbot ready. Type 'exit' to quit.\n")

    while True:
        # Choose between RAG mode (Ask AI) and Quiz mode
        mode = input("Choose mode: (1) Ask AI | (2) Quiz Me: ").strip()

        if mode == "2": # Quiz Mode
            while True:
                context = [random.choice(docs)] # Select a random note for the quiz

                question = generate_question(context) # Generate a question based on the note
                print(f"\nAI Question: {question}")

                user_answer = input("Your Answer: ").strip() # Get user's answer

                if user_answer.lower() in ["exit", "quit"]:
                    break # Exit quiz loop

                # Check the user's answer against the note
                result = check_answer(user_answer, context, question) 
                print("\n", result)
                print("-" * 40)
        else: # Ask AI (RAG) Mode
            try:
                question = input("\nYou: ").strip() # Get user's question
                if not question: 
                    continue
                    
                if question.lower() in ["exit", "quit"]:
                    break # Exit main loop

                print("Searching notes...")
                # Retrieve relevant context documents
                context = search_context(question, docs)
                
                print("Thinking...")
                # Ask the LLM, providing the context and history
                answer = ask_gemini(question, context, chat_history)

                print(f"\nAI: {answer}")
                print("-" * 30)

                # Update chat history
                chat_history.append({"user": question, "ai": answer})
                
                # Keep chat history size limited (rolling window)
                if len(chat_history) > 5:
                    chat_history.pop(0)
            except Exception as e:
                print(f"\nAn error occurred: {e}")

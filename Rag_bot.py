# --- 1. Import Necessary Libraries ---
# dotenv is a library to load environment variables (like your API Key) 
# from a special file named '.env'. This keeps your secret key secure.
from dotenv import load_dotenv
# os provides a way to interact with your operating system, like reading files and folders.
import os
# This is the main library for interacting with Google's Gemini models.
import google.generativeai as genai
# numpy is a powerful library for working with numbers, especially arrays and math operations.
# We'll use it here for handling the "embeddings" (number representations of text).
import numpy as np

# --- 2. Setup and Configuration ---

# Load environment variables (secrets) from the .env file.
load_dotenv()
# Get the API key from the environment variables loaded by load_dotenv().
# 'os.environ.get()' safely retrieves the value of the "API_KEY" variable.
API_KEY = os.environ.get("API_KEY")

# Check if the API Key was loaded successfully.
if not API_KEY:
    # If the key is missing, stop the program and show an error message.
    raise ValueError("API Key not found! Check your .env file.")

# Configure the Gemini library to use your loaded API key.
genai.configure(api_key=API_KEY)

# Define the name of the model to use for creating text embeddings (numerical representations).
EMBED_MODEL = "models/text-embedding-004"
# Define the name of the Large Language Model (LLM) we will use to generate answers.
LLM_MODEL = "gemini-2.5-flash" 

# --- 3. Functions to Load and Prepare Data ---

# Define a function to find and read all your notes.
def load_notes(folder="My_notes"):
    """Loads all .txt files from the specified folder."""
    # Check if the folder for notes exists.
    if not os.path.exists(folder):
        # If it doesn't exist, create it.
        os.makedirs(folder)
        # Tell the user to add files and return an empty list of documents.
        print(f"Created folder '{folder}'. Please add .txt files there!")
        return []

    # This list will store information about all your loaded notes (file name and text content).
    docs = []
    # Loop through every file name inside the notes folder.
    for filename in os.listdir(folder):
        # Check if the file ends with ".txt" (we only want text files).
        if filename.endswith(".txt"):
            # Create the full file path by joining the folder name and the file name.
            path = os.path.join(folder, filename)
            # Use a 'try...except' block to handle potential errors while reading files.
            try:
                # Open the file in 'read' mode ("r") using UTF-8 encoding.
                # 'with open(...)' ensures the file is closed automatically.
                with open(path, "r", encoding="utf-8") as f:
                    # Read the entire text from the file and remove any extra spaces from the ends.
                    text = f.read().strip()
                    # Check if the file actually contains any text (isn't empty).
                    if text: 
                        # Add a dictionary to the 'docs' list with the file name and its text content.
                        docs.append({"file": filename, "text": text})
            # If an error happens during file reading (e.g., file is corrupted), catch it.
            except Exception as e:
                # Print an informative error message.
                print(f"Error reading {filename}: {e}")
    # Return the list of all successfully loaded documents.
    return docs

# --- 4. Functions for Text Embedding (Turning Text into Numbers) ---

# Define a function to convert text into a numerical vector (embedding).
def embed_text(text, task_type="retrieval_query"):
    """Corrected function using genai.embed_content"""
    # Call the Gemini API to get the embedding for the given text.
    result = genai.embed_content(
        model=EMBED_MODEL,  # Use the defined embedding model.
        content=text,       # The text we want to convert.
        # This tells the model what the embedding is for (a question or a document).
        task_type=task_type 
    )
    # The result contains a list of numbers. We convert this list into a NumPy array, 
    # which is easier to do math with, and return it.
    return np.array(result['embedding'])

# Define a function to create an embedding for every document.
def build_index(docs):
    """Generates embeddings for all documents."""
    print(f"Indexing {len(docs)} documents...")
    # Loop through each document dictionary in the 'docs' list.
    for doc in docs:
        # Create the embedding for the document's text. We specify 'retrieval_document'
        # to tell the model this is a piece of source material.
        doc["embedding"] = embed_text(doc["text"], task_type="retrieval_document")
    # Return the updated list of documents, now with the 'embedding' added to each.
    return docs

# --- 5. Functions for Searching and Comparison ---

# Define a function to calculate the similarity between two numerical vectors (embeddings).
# This is a standard math operation to find how 'close' two vectors are, where 1.0 is identical.
def cosine_sim(a, b):
    # Check if either vector is all zeros (to avoid division by zero error).
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    # The formula for cosine similarity: (dot product of a and b) / (length of a * length of b).
    # 'np.dot' calculates the dot product (a measure of how much they point in the same direction).
    # 'np.linalg.norm' calculates the length of the vector.
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Define the function to search your indexed documents for the best matches to a question.
def search_context(query, docs, top_k=2):
    # If there are no documents, return an empty list.
    if not docs:
        return []
    
    # First, convert the user's question into an embedding.
    query_embed = embed_text(query, task_type="retrieval_query")
    # This list will store the similarity score and the corresponding document for every note.
    scored = []

    # Loop through every indexed document.
    for doc in docs:
        # Calculate the similarity score between the question embedding and the document embedding.
        score = cosine_sim(query_embed, doc["embedding"])
        # Add a tuple of (score, document_data) to the list.
        scored.append((score, doc))

    # Sort the list of scores and documents. 'reverse=True' means the highest score comes first.
    # 'key=lambda x: x[0]' tells Python to sort based on the first item in the tuple (the score).
    scored.sort(reverse=True, key=lambda x: x[0])
    # Return only the text from the top 'top_k' (e.g., the top 2) highest-scoring documents.
    return [doc for _, doc in scored[:top_k]]

# --- 6. Functions for Chat History and LLM Interaction ---

# Define a function to convert the chat history list into a single, clean string format.
def format_history(history):
    """Formats the chat history into a string."""
    text = ""
    # Loop through each turn (question and answer) in the history list.
    for turn in history:
        # Add the User's part of the conversation.
        text += f"User: {turn['user']}\n"
        # Add the AI's part of the conversation.
        text += f"AI: {turn['ai']}\n\n"
    # Return the combined, formatted text.
    return text

# Define the function that combines the notes, history, and question to ask the LLM.
def ask_gemini(question, context_docs, chat_history):
    """Generates an answer using Context + Chat History."""
    
    # Combine the text from the retrieved context documents into a single string.
    # It adds a separator ('\n\n---\n\n') between notes for clarity.
    context_text = "\n\n---\n\n".join(
        [f"From {doc['file']}:\n{doc['text']}" for doc in context_docs]
    )

    # Format the chat history into a string.
    history_text = format_history(chat_history)

    # This is the full instruction (the "prompt") sent to the Gemini model.
    # It tells the model its role, what context to use, and how to answer.
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

    # Create an instance of the GenerativeModel using the chosen LLM model name.
    model = genai.GenerativeModel(LLM_MODEL)
    # Send the combined prompt to the model and get the response.
    response = model.generate_content(prompt)
    # Return only the text part of the model's response.
    return response.text

# --- 7. Main Program Logic ---

# Define the main function where the program execution starts.
def main():
    print("Loading documents...")
    # Call the function to load all notes from the folder.
    docs = load_notes()
    
    # Check if any documents were loaded.
    if not docs:
        print("No documents found. Exiting.")
        # Stop the program if there are no notes.
        return

    # Call the function to create numerical embeddings for all loaded documents.
    docs = build_index(docs)
    
    # Initialize an empty list to store the conversation history.
    chat_history = []
    
    print("RAG chatbot ready. Type 'exit' to quit.\n")

    # Start an infinite loop to keep the chat going until the user stops it.
    while True:
        # Use a 'try...except' block to gracefully handle unexpected errors during the chat.
        try:
            # Ask the user for their question and remove any extra spaces.
            question = input("\nYou: ").strip()
            # If the user just pressed Enter (empty question), skip the rest of the loop.
            if not question: 
                continue
                
            # Check if the user wants to quit the chat.
            if question.lower() in ["exit", "quit"]:
                # Exit the 'while True' loop, which ends the program.
                break

            print("Searching notes...")
            # Use the search function to find the most relevant notes (context) for the question.
            context = search_context(question, docs)
            
            print("Thinking...")
            # Use the ask function to get the final answer from the LLM, using the question,
            # the relevant notes (context), and the chat history.
            answer = ask_gemini(question, context, chat_history)

            # Print the AI's final answer.
            print(f"\nAI: {answer}")
            print("-" * 30)

            # Add the current user question and the AI's answer to the chat history list.
            chat_history.append({"user": question, "ai": answer})
            
            # This is a simple way to limit the size of the history to the last 5 turns.
            # If the list gets too long (more than 5), remove the oldest turn (the first item).
            if len(chat_history) > 5:
                chat_history.pop(0)
            
        # If any error occurs inside the loop, print an error message but keep the chat running.
        except Exception as e:
            print(f"\nAn error occurred: {e}")

# This is the standard Python way to ensure the 'main()' function runs only when the script 
# is executed directly (not when it's imported as a module into another file).
if __name__ == "__main__":
    main()
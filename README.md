
# RAG Study Assistant (with Quiz Mode)

A RAG-based study tool that lets you ask questions from your own notes and quiz yourself. Add your notes, run the app, and interact with your data.

## Features

- Supports `.txt` and `.pdf` files in the `My_notes` folder  
- Splits long notes into overlapping chunks  
- Embeds and indexes your notes for retrieval  
- “Ask AI” mode for question answering  
- “Quiz Me” mode for random generated questions  
- Terminal-based version included (`Rag_bot.py`)

## Tech Stack

- Python  
- Streamlit  
- Google Gemini API  
- pdfplumber  
- NumPy  

## Project Structure

```
.
├── app.py              # Streamlit app
├── Rag_helpers.py      # Core RAG functions
├── Rag_bot.py          # CLI version
└── My_notes/           # Your notes
```

## Setup

### Install Dependencies
```
pip install -r requirements.txt
```

### Add Gemini API Key
Create `.streamlit/secrets.toml`:

```
GEMINI_API_KEY = "your_api_key_here"
```

### Add Notes
Place `.txt` or `.pdf` files into `My_notes/`.

## Run Streamlit App
```
streamlit run app.py
```

## Run CLI Version
```
python Rag_bot.py
```

## How It Works

- Notes → chunked  
- Chunks → embedded  
- Query → embedded  
- Retrieval uses cosine similarity  
- Gemini model uses the selected context to answer  
- Quiz mode creates a random question and checks answers

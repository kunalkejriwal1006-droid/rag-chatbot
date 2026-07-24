# Insurance RAG Chatbot

## The problem

Insurance policy documents are long, technical, and full of tables. When someone
needs an answer — what a specific add-on covers, how many times a benefit can be
used, how a No Claim Bonus discount is calculated — they have to search through
many pages of PDFs to find it. This is slow and easy to get wrong.

## What this project does

This project answers questions about insurance policy documents in plain
English. You give it PDFs (currently 13 Bajaj Allianz two-wheeler insurance
documents), ask a question like *"What is Zero Depreciation cover and how many
times can it be used?"*, and it replies with an answer plus the exact document
and page number the answer came from.

It only answers using the content of the PDFs it was given. It does not use
outside knowledge, so it will not guess or make things up.

## How it works, step by step

1. **Read the PDFs.** Each page's text and any tables on it are extracted.
2. **Split into small pieces.** The text is broken into short, meaningful
   sections — never cutting a sentence or a table row in half.
3. **Convert each piece into a searchable form.** This lets the system find
   relevant text by meaning, not just by matching the exact same words.
4. **Store everything** so it can be searched later.
5. **When someone asks a question:**
   - The system finds the pieces of text most relevant to that question.
   - Those pieces are given to an AI model, which writes an answer using only
     that text, and states which document and page each fact came from.
6. **For number-based questions** (like a No Claim Bonus discount, or a
   vehicle's value after depreciation), the system calculates the answer
   itself using fixed, known rules — it does not ask the AI to do the math.
   This keeps the numbers accurate every time.
7. **If the AI's first choice is unavailable** (for example, it's temporarily
   out of free usage), the system automatically tries a backup AI model
   instead, so a question can still be answered.

## How to use it

### 1. Set up Python

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

### 2. Add your settings

Copy `env.example` to `.env` and fill in an API key for at least one AI
model (Gemini and/or Groq). A local model (Ollama) can also be used and
needs no key.

### 3. Load the PDFs in

```powershell
.venv\Scripts\python.exe -m rag.ingest --rebuild
```

Run this once to read all the PDFs and prepare them for searching. Add new
PDFs to the folder and run it again (without `--rebuild`) to include them.

### 4. Ask questions

Terminal:

```powershell
.venv\Scripts\python.exe cli.py "What is Zero Depreciation cover?"
```

Or a web chat window:

```powershell
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## What's in this project

```
rag/                 The core logic: reading PDFs, splitting text, searching,
                      calculating, and generating answers.
cli.py                Ask questions from the terminal.
streamlit_app.py      Ask questions from a web chat window.
*.pdf                 The insurance documents this project answers questions from.
data/                 Generated automatically when you load the PDFs in.
```

## What's next

This is the first working part of a larger chatbot project. The next steps
are building the full conversational chatbot on top of this, and the logic
that decides how the chatbot handles different kinds of requests.

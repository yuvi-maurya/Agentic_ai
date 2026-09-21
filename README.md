# StudyBuddy

A study-assistant AI agent built with **Google ADK** and **Gemini**. It answers questions from your own notes using **RAG** (Retrieval-Augmented Generation), does safe arithmetic, saves notes, and quizzes you while keeping score.

## Features

| Tool | What it does |
|---|---|
| `search_study_material` | Finds the most relevant passages in your own `.txt` notes (RAG) |
| `calculate` | Safely evaluates arithmetic (numbers and `+ - * / ( )` only, no `eval`) |
| `save_note` | Appends a timestamped note to `notes.txt` |
| `track_quiz_score` | Keeps a running correct/total score for the session |

The agent decides on its own which tool to call. Every tool call is printed in the terminal as `[TOOL CALL] ...`.

## How it works

```
Question -> Gemini decides which tool to use -> tool runs -> Gemini writes the answer

RAG pipeline for notes:
library/*.txt -> chunk (~500 chars) -> embed (all-MiniLM-L6-v2) -> store in Chroma
Question -> embed -> nearest chunks (distance <= 1.2) -> passed to the agent as context
```

## Tech stack

- Python 3.10+
- Google ADK 2.9.2 (agent framework)
- Gemini free tier (`gemini-3.1-flash-lite`) with automatic retry on 429/500/503/504
- sentence-transformers `all-MiniLM-L6-v2` (local embeddings, no API key)
- Chroma (local vector database)

## Project structure

```
study_agent/
  __init__.py
  agent.py        # 4 tools, system instruction, agent definition
  rag.py          # chunk, embed, store, retrieve
  ingest.py       # builds the vector database from library/
  .env.example    # copy to .env and add your key
library/          # your notes as .txt files (English)
requirements.txt
```

## Setup

```powershell
git clone https://github.com/yuvi-maurya/Agentic_ai.git
cd Agentic_ai
python -m venv .venv
.venv\Scripts\Activate.ps1          # Mac/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

1. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Copy `study_agent/.env.example` to `study_agent/.env` and paste your key.
3. Put your notes (English `.txt` files) in `library/`.
4. Build the vector database (re-run whenever notes change):

```powershell
python study_agent/ingest.py
```

## Run

```powershell
adk web --port 8000      # browser UI at http://localhost:8000, pick "study_agent"
adk run study_agent      # terminal chat, type "exit" to quit
```

## Try it

- `What's 342 times 17?` -> uses `calculate`
- `Save a note: mitochondria is the powerhouse of the cell` -> uses `save_note`
- `Based on my notes, explain the water cycle` -> uses `search_study_material`
- `Quiz me on the water cycle` -> asks one question at a time and tracks the score
- `Who won the cricket world cup?` -> not in the notes, so the agent says so before giving a general answer

## Notes and limitations

- Notes should be in English (the embedding model is English-focused).
- The Gemini free tier has rate limits and can return 503 during busy times; retries are built in.
- `.env`, `chroma_db/`, `notes.txt` and `.adk/` are git-ignored.
# Mediplus

Initial Python project scaffold for LangChain and LangGraph work.

## Setup

The project uses a local virtual environment at `.venv`.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

API keys belong in `.env`; real credentials must never be committed.

## Layout

- `agents/` - agent implementations
- `data/` - local data and datasets
- `models/` - model configuration and schemas
- `db/` - database access
- `ui/` - user interface code

# Syracuse Answers Chat — Frontend

Next.js App Router frontend for the Syracuse University wiki RAG chat.

## Prerequisites

- **Node.js** >= 18
- **Backend** running at `http://localhost:8000` (FastAPI + wiki RAG)

## Quick start

```bash
# Install dependencies
npm install

# Start dev server (port 3000)
npm run dev
```

Then open [http://localhost:3000](http://localhost:3000).

## Start the backend

From the project root (parent directory):

```bash
# Activate the Python venv
env\Scripts\activate        # Windows
# source env/bin/activate   # macOS / Linux

# Start the API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | `http://localhost:8000` | FastAPI backend URL |

Set in `.env.local` or your environment.

## Project structure

```
frontend/
  app/
    layout.tsx          Root layout
    page.tsx            Main chat page (client component)
    globals.css         Theme variables and base styles
    page.module.css     Page-level layout styles
  components/
    Sidebar.tsx         Chat session list and new-chat button
    ChatWindow.tsx      Message bubbles, input, loading/error
  lib/
    chatClient.ts       Fetch wrapper for POST /chat
    types.ts            Shared TypeScript interfaces
```

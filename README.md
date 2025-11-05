# Chatbot AI Assistant

A compact Chatbot AI Assistant that uses the OpenAI Python SDK to provide an interactive assistant inside Streamlit. It supports streaming responses, a settings sidebar (model, temperature, system prompt), immediate token/cost accounting in the sidebar, and an optional in-session API key override.

Features

- Streaming responses (partial content shown as it arrives).
- Sidebar controls for model, temperature, system prompt, and API key override.
- Token and cost estimation shown per-turn and accumulated in the sidebar.
- Immediate sidebar updates (a placeholder is used so the dataframe refreshes in the same Streamlit run).

Requirements

- macOS / Linux / Windows with Python 3.8+ (use the same Python you develop with).
- See `requirements.txt` for pinned packages. Key packages:
  - streamlit
  - pandas
  - openai
  - python-dotenv

Quick start (local)

1. Clone the repo

```bash
$ git clone https://github.com/uzuntasgokberk/AI-Assistant.git
```

2. Create and activate a virtual environment (zsh example)

```bash
$ uv venv
$ source .venv/bin/activate
$ uv pip install -r requirements.txt 
```

3. Provide your OpenAI API key

- Option A — environment (recommended): create a `.env` file in the repository root with:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

- Option B — paste into the sidebar at runtime (not persisted to disk). The sidebar entry overrides the environment key for the current session.

4. Run the app

```bash
# quick dev/testing entrypoint (shows the sidebar placement fix):
$ streamlit run main.py
```

Enjoy the assistant

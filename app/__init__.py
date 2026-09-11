"""Wintermute — the public gateway of the assistant.

``app/api.py`` exposes the assistant behind Ollama/OpenAI-compatible
endpoints so an Ollama instance (or Open WebUI pointed at one) can talk to
Wintermute as if it were just another model. Run it with:

    venv/Scripts/python.exe -m uvicorn app.api:app --port 8000
"""

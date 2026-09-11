from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uuid
import time
from src.retrieval.rag import answer

app = FastAPI(
    title="Dark Earth assistant",
    version="1.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
)

# -------------------------------------------------------------------
# Modèles OpenAI
# -------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False

# -------------------------------------------------------------------
# Middleware CORS — indispensable si Open WebUI est sur un port différent
# -------------------------------------------------------------------
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Restreindre en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Middleware de Logging
# -------------------------------------------------------------------

import logging
from fastapi import Request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f">>> {request.method} {request.url}")
    logger.info(f"    Headers: {dict(request.headers)}")
    response = await call_next(request)
    logger.info(f"<<< Status: {response.status_code}")
    return response

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Local RAG API"
    }


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "dark-earth-rag",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "permission": [],
                "root": "dark-earth-rag",
                "parent": None,
            }
        ]
    }

@app.get("/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "dark-earth-rag",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "permission": [],
                "root": "dark-earth-rag",
                "parent": None,
            }
        ]
    }

@app.get("/api/tags")
def api_tags():
    """
    Certaines versions d'Open WebUI appellent aussi cet endpoint
    (héritage du format Ollama).
    """
    return {
        "models": [
            {
                "name": "dark-earth-rag",
                "model": "dark-earth-rag",
                "modified_at": "2025-01-01T00:00:00Z",
                "size": 0,
                "digest": "local",
                "details": {
                    "format": "gguf",
                    "family": "local",
                    "parameter_size": "unknown",
                    "quantization_level": "none"
                }
            }
        ]
    }

@app.post("/v1/chat/completions")
def chat(request: ChatCompletionRequest):

    # Dernier message utilisateur
    question = request.messages[-1].content

    # Appel du moteur RAG
    response = answer(question)

    # Réponse au format OpenAI
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }
"""
main.py — Servidor FastAPI que expone el agente RAG como API REST.
"""
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from app.agent import build_agent, get_retriever, invoke_agent

load_dotenv(override=True)

# ──────────────────────────────────────────────
# Estado global de la aplicación
# ──────────────────────────────────────────────
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa recursos pesados una sola vez al arrancar."""
    print("🚀 Iniciando servidor: cargando retriever y agente...")
    get_retriever()                        # indexa ChromaDB
    app_state["agente"] = build_agent()   # construye el agente
    print("✅ Servidor listo")
    yield
    # Limpieza al apagar (opcional)
    app_state.clear()


# ──────────────────────────────────────────────
# Aplicación FastAPI
# ──────────────────────────────────────────────
app = FastAPI(
    title="RAG Autonomous Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # En producción limitar al dominio del frontend
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Modelos Pydantic
# ──────────────────────────────────────────────
class Mensaje(BaseModel):
    role: str          # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    historial: list[Mensaje]


class ChatResponse(BaseModel):
    respuesta: str
    historial: list[Mensaje]


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Recibe el historial de mensajes y devuelve la respuesta del agente.
    El frontend debe mantener y enviar el historial completo en cada llamada.
    """
    agente = app_state.get("agente")
    if agente is None:
        raise HTTPException(status_code=503, detail="Agente no inicializado")

    # Convertir mensajes Pydantic → objetos LangChain
    historial_lc = []
    for msg in request.historial:
        if msg.role == "user":
            historial_lc.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            historial_lc.append(AIMessage(content=msg.content))

    respuesta_str, historial_actualizado = invoke_agent(agente, historial_lc)

    # Convertir de vuelta a Pydantic
    historial_out = []
    for msg in historial_actualizado:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        historial_out.append(Mensaje(role=role, content=msg.content))

    return ChatResponse(respuesta=respuesta_str, historial=historial_out)

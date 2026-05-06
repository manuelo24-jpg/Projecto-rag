"""
app.py — Interfaz web con Chainlit.
Habla con el backend FastAPI y muestra los pasos intermedios del agente.
"""
import os
import httpx
import chainlit as cl
from dotenv import load_dotenv

load_dotenv(override=True)

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")


# ──────────────────────────────────────────────
# Eventos de sesión
# ──────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    """Se ejecuta cuando el usuario abre el chat."""
    cl.user_session.set("historial", [])
    await cl.Message(
        content=(
            "👋 ¡Hola! Soy tu asistente RAG.\n\n"
            "Puedes preguntarme cualquier cosa sobre el artículo de **Lilian Weng** "
            "sobre agentes autónomos con LLMs.\n\n"
            "Ejemplos de preguntas:\n"
            "- ¿Qué es un agente autónomo basado en LLMs?\n"
            "- ¿Cuáles son los componentes principales de un agente de IA?\n"
            "- ¿Qué es el Chain of Thought prompting?\n"
            "- ¿Qué diferencia hay entre memoria a corto y largo plazo?"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Se ejecuta con cada mensaje del usuario."""
    historial = cl.user_session.get("historial", [])

    # Añadir pregunta al historial
    historial.append({"role": "user", "content": message.content})

    # Mensaje de espera con indicador de actividad
    thinking_msg = cl.Message(content="")
    await thinking_msg.send()

    async with cl.Step(name="🔍 Buscando en el artículo...") as step:
        step.input = message.content

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{BACKEND_URL}/chat",
                    json={"historial": historial},
                )
                response.raise_for_status()
                data = response.json()

            respuesta = data["respuesta"]
            historial_actualizado = data["historial"]

            step.output = f"Encontré {len(historial_actualizado)} mensajes en el historial"

        except httpx.HTTPStatusError as e:
            respuesta = f"❌ Error del servidor: {e.response.status_code}"
            historial_actualizado = historial
        except httpx.RequestError:
            respuesta = "❌ No se pudo conectar con el backend. ¿Está arrancado?"
            historial_actualizado = historial

    # Actualizar historial en la sesión
    cl.user_session.set("historial", historial_actualizado)

    # Mostrar respuesta
    await thinking_msg.update()
    await cl.Message(content=respuesta).send()

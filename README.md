# RAG Autonomous Agent: Interfaz Web y Despliegue con Docker

Este proyecto implementa un sistema de **Generación Aumentada por Recuperación (RAG)** avanzado, utilizando una arquitectura de agentes autónomos. La aplicación permite interactuar con documentos técnicos (basados en el artículo de Lilian Weng sobre agentes) a través de una interfaz web moderna, facilitando la trazabilidad del pensamiento del modelo.

## 🚀 Recomendaciones Tecnológicas (Stack Seleccionado)

Para este proyecto, se han seleccionado las siguientes tecnologías basándose en criterios de eficiencia, experiencia de usuario y facilidad de despliegue:

| Componente | Tecnología | Justificación (¿Por qué?) |
| :--- | :--- | :--- |
| **Interfaz (Frontend)** | **Chainlit** | A diferencia de Streamlit, Chainlit está diseñada exclusivamente para chat. Permite visualizar de forma nativa los "pasos intermedios" del agente (Chain of Thought), lo cual es crítico para depurar y dar confianza al usuario en sistemas RAG. |
| **Orquestador** | **LangGraph** | Permite crear flujos cíclicos y estados complejos. En lugar de una cadena lineal, usamos un agente que puede decidir si necesita buscar más información o si ya tiene la respuesta. |
| **Inferencia LLM** | **Groq (Llama 3.3)** | Ofrece una velocidad de inferencia (tokens por segundo) líder en la industria, lo que garantiza que la interfaz web sea fluida y responda casi instantáneamente. |
| **Base de Datos Vectorial** | **ChromaDB** | Es ligera, de código abierto y permite persistencia local, lo que simplifica enormemente el despliegue dentro de un contenedor Docker sin necesidad de servicios externos. |
| **Embeddings** | **Hugging Face** | El modelo `all-MiniLM-L6-v2` proporciona un equilibrio perfecto entre precisión semántica y bajo consumo de recursos (CPU/RAM). |
| **Contenedorización** | **Docker** | Garantiza la paridad entre el entorno de desarrollo y producción, encapsulando todas las dependencias de Machine Learning. |

---

## 🛠️ Especificaciones Técnicas

### 1. Requisitos Previos
* Python 3.10 o superior.
* Una API Key de **Groq**.
* Docker instalado (para el despliegue).

### 2. Estructura del Proyecto
```text

PROYECTO-RAG/
├── backend/                # Lógica del Agente y API
│   ├── app/
│   │   ├── main.py         # Servidor FastAPI
│   │   ├── agent.py        # Lógica de LangGraph (tu notebook)
│   │   └── tools.py        # Herramientas de búsqueda (scraping)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/               # Interfaz de Usuario
│   ├── app.py              # Aplicación Chainlit
│   ├── requirements.txt
│   └── Dockerfile
├── data/                   # Persistencia de base de datos
│   └── chroma_db/          # Aquí se guardarán los vectores
├── .env                    # Claves de API (GROQ_API_KEY, etc.)
└── docker-compose.yml       # Orquestador de contenedores
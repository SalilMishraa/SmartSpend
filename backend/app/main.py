from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from groq import Groq
from dotenv import load_dotenv

from .api.v1.health import router as health_router
from .api.v1.analyze import router as analyze_router
from .api.v1.chat import router as chat_router

app = FastAPI(title="SmartSpend API", version="1.0")

load_dotenv()

def create_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)

app.state.groq_client = create_groq_client()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api/v1")
app.include_router(analyze_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
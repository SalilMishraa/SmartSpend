from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from services.ai import get_chatbot_response

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    metrics: dict
    chat_history: list


@router.post("/chat")
def chat(data: ChatRequest, request: Request):

    groq_client = request.app.state.groq_client

    if groq_client is None:
        return {"response": "Chatbot unavailable. API key not configured."}

    try:
        response = get_chatbot_response(
            data.question,
            data.metrics,
            data.chat_history,
            groq_client
        )
        return {"response": response}

    except Exception:
        raise HTTPException(status_code=500, detail="Chat error")
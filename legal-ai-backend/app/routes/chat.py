from fastapi import APIRouter, HTTPException
from app.models.request_models import ChatRequest
from app.services.llm import llm_client

router = APIRouter()

@router.post("/chat")
async def chat_with_ai(payload: ChatRequest):
    try:
        response = llm_client.chat(payload.query, payload.context)
        return {"reply": response}

    except Exception as e:
        print("Chat Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

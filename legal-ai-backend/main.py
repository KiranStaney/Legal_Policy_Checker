import uvicorn
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Legal AI Advisor",
    description="Integrated system for contract analysis using RAG and Agentic AI."
)

# --- CORS ---
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ROUTES MUST COME HERE (IMPORTANT!) ---
from app.routes.upload import router as upload_router
from app.routes.analyze import router as analyze_router
from app.routes.chat import router as chat_router

app.include_router(upload_router, tags=["Upload"])
app.include_router(analyze_router, tags=["Analysis"])
app.include_router(chat_router, tags=["Chat"])

@app.get("/")
def read_root():
    return {"message": "Legal AI Advisor Backend is operational!"}

# --- DO NOT PUT ROUTES BELOW THIS LINE ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

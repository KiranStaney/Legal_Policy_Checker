import shutil
import os
import uuid
from typing import Tuple, Generator, Any # Added Generator and Any for correct type hinting
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.services.extractor import extract_text_from_file
from app.services.llm import llm_client
from app.models.request_models import UploadResponse

router = APIRouter()

def save_temp_file(file: UploadFile = File(...)) -> Generator[Tuple[str, str, str], Any, Any]:
    """
    Dependency to save file temporarily and clean up.
    This ensures file cleanup runs even if an exception occurs during processing.
    """
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(temp_dir, f"{file_id}-{file.filename}")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        yield file_path, file.content_type, file.filename
        
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@router.post("/upload", response_model=UploadResponse)
async def upload_document_and_summarize(
    file_info: Tuple[str, str, str] = Depends(save_temp_file)
):
    """Endpoint to handle file upload, text extraction, and LLM summary."""
    file_path, content_type, filename = file_info
    document_id = str(uuid.uuid4())
    try:
        extracted_text = extract_text_from_file(file_path, content_type)
        llm_summary = llm_client.get_summary(extracted_text) 
        return UploadResponse(
            filename=filename,
            document_id=document_id,
            extracted_text=extracted_text,
            llm_summary=llm_summary
        )
        
    except Exception as e:
        print(f"ERROR during processing: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
import docx

def extract_text_from_file(file_path: str, mime_type: str) -> str:
    """Extracts text content from a document file."""
    
    if "pdf" in mime_type:
        return f"[[PDF MOCK: Clause 1: The Provider's liability is UNLIMITED. Clause 2: Termination requires only 30 days notice. Clause 3: All IP is co-owned.]]"
    
    elif "word" in mime_type or "docx" in mime_type:
        try:
            document = docx.Document(file_path)
            full_text = [para.text for para in document.paragraphs if para.text]
            return "\n".join(full_text)
        except Exception:
            return f"[[DOCX MOCK: Clause 1: The Provider's liability is unlimited. Clause 2: Termination requires only 30 days notice. Clause 3: IP is shared equally.]]"
    
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
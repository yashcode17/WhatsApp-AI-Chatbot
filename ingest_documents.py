#ingest_documents.py

"""
Run this locally (or as a one-off Render job) whenever you add new property documents.
It reads PDFs, images, and text files from a folder, extracts text, chunks it,
embeds it, and stores it in the database.
"""

import os
import json
import base64
import logging

from pypdf import PdfReader
from fastembed import TextEmbedding
from groq import Groq

from app import SessionLocal, DocumentChunk, GROQ_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOCS_FOLDER = "./property_documents"

embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
groq_client = Groq(api_key=GROQ_API_KEY)

VISION_MODEL = "qwen/qwen3.6-27b"

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks for better retrieval."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def extract_text_from_pdf(filepath: str) -> str:
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def extract_text_from_image(filepath: str) -> str:
    """Use a Groq vision model to describe/extract info from property images."""
    with open(filepath, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    response = groq_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe this property image in detail: layout, rooms, condition, "
                            "notable features, and any text/numbers visible (like price tags, "
                            "area in sqft, address boards, etc.)."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    }
                ]
            }
        ],
        max_tokens=500
    )
    return response.choices[0].message.content


def extract_text_from_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def ingest_file(filepath: str, session):
    filename = os.path.basename(filepath)
    ext = filename.lower().split(".")[-1]

    # Remove old chunks of same file
    session.query(DocumentChunk).filter(
        DocumentChunk.source_file == filename
    ).delete()
    session.commit() # fixed: was session.comit(), which raised AttributeError

    logger.info("Processing: %s", filename)

    if ext == "pdf":
        text = extract_text_from_pdf(filepath)
    elif ext in ("jpg", "jpeg", "png"):
        text = extract_text_from_image(filepath)
    elif ext == "txt":
        text = extract_text_from_txt(filepath)
    else:
        logger.warning(" Skipping unsupported file type: %s", filename)
        return

    chunks = chunk_text(text)
    logger.info(" -> split into %d chunks", len(chunks))

    for chunk in chunks:
        embedding = list(embedder.embed([chunk]))[0].tolist()
        record = DocumentChunk(
            source_file=filename,
            chunk_text=chunk,
            embedding=json.dumps(embedding)
        )
        session.add(record)

    session.commit()
    logger.info(" Saved %d chunks from %s", len(chunks), filename)


def main():
    session = SessionLocal()
    try:
        for filename in os.listdir(DOCS_FOLDER):
            filepath = os.path.join(DOCS_FOLDER, filename)
            if os.path.isfile(filepath):
                ingest_file(filepath, session)
    finally:
        session.close()

    logger.info("Ingestion complete!")


if __name__ == "__main__":
    main()



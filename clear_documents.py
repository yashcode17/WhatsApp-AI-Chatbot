#clear_documents.py

from app import SessionLocal, DocumentChunk

def clear_all_chunks():
    session = SessionLocal()
    try:
        deleted = session.query(DocumentChunk).delete()
        session.commit()
        print(f"Deleted {deleted} existing chunks")
    finally:
        session.close()

if __name__ == "__main__":
    clear_all_chunks()
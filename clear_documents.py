#clear_documents.py

from app import SessionLocal, DocumentChunk

def clear_all_chunks():
    session = SessionLocal()
    try:
        delete = session.query(DocumentChunk).delete()
        session.commit()
        print("Deleted {delete} existing chunks")
    finally:
        session.close()

if __name__ == "__main__":
    clear_all_chunks()
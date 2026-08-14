from pathlib import Path
import sys

RAG_ENGINE_DIR = Path(__file__).resolve().parent.parent
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

from rag_service import add_user_document, create_engine, load_env


def ingest_pdf():
    load_env()

    collection_name = "study_chunks"

    print(f"Creating engine with collection: {collection_name}...")
    engine = create_engine(collection_name=collection_name)

    pdf_path = RAG_ENGINE_DIR / "data" / "user_a_docs" / "injection_test.pdf"
    user_id = "injection_test@test.com"

    print(f"Adding PDF: {pdf_path} for user: {user_id}...")
    n = add_user_document(engine, pdf_path, user_id=user_id)
    print(f"Added {n} chunks.")

    print("Ingestion complete.")


if __name__ == "__main__":
    ingest_pdf()
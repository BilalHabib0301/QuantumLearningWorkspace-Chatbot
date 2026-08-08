
from pathlib import Path
from rag_service import add_user_document, create_engine, load_env

def ingest_pdf():
    load_env()
    
    # We should use a separate collection name to avoid polluting other tests
    collection_name = "injection_test_collection"
    
    print(f"Creating engine with collection: {collection_name}...")
    engine = create_engine(collection_name=collection_name)
    
    pdf_path = Path("rag-engine/data/user_a_docs/injection_test.pdf")
    user_id = "user_a"
    
    print(f"Adding PDF: {pdf_path} for user: {user_id}...")
    n = add_user_document(engine, pdf_path, user_id=user_id)
    print(f"Added {n} chunks.")
    
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_pdf()

"""
Diagnostic script to inspect ChromaDB vector store contents.
Run this to see what documents are indexed and debug RAG issues.
"""

import logging
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import get_gemini_api_key

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def inspect_vector_store():
    """Inspect the contents of the ChromaDB vector store."""
    persist_dir = "./rag_engine/vector_store"
    persist_path = Path(persist_dir)
    
    if not persist_path.exists():
        logger.warning(f"Vector store directory does not exist: {persist_dir}")
        return
    
    logger.info("=" * 80)
    logger.info("VECTOR STORE INSPECTION")
    logger.info("=" * 80)
    
    try:
        # Try Gemini embeddings first, fallback to local if not available
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=get_gemini_api_key()
            )
            embeddings.embed_query("test")  # Test if it works
        except Exception:
            # Fallback to local embeddings (free tier compatible)
            logger.info("Using local embeddings (Gemini embeddings API not available)")
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
        
        # Load vectorstore
        vectorstore = Chroma(
            persist_directory=str(persist_path),
            embedding_function=embeddings
        )
        
        # Get collection
        collection = vectorstore._collection
        
        # Get document count
        count = collection.count()
        logger.info(f"📚 Total indexed documents (chunks): {count}")
        
        if count == 0:
            logger.warning("⚠️ No documents found in vector store!")
            logger.info("💡 Upload documents and generate test cases to populate the index.")
            return
        
        # Get all documents
        logger.info("\n" + "=" * 80)
        logger.info("DOCUMENT CHUNKS IN STORE:")
        logger.info("=" * 80)
        
        results = collection.get(limit=count)
        
        if results and 'documents' in results:
            documents = results['documents']
            metadatas = results.get('metadatas', [{}] * len(documents))
            ids = results.get('ids', [])
            
            logger.info(f"\nFound {len(documents)} chunks:\n")
            
            for idx, (doc_id, doc_text, metadata) in enumerate(zip(ids, documents, metadatas), 1):
                logger.info(f"\n--- Chunk {idx} (ID: {doc_id}) ---")
                logger.info(f"Length: {len(doc_text)} characters")
                logger.info(f"Metadata: {metadata}")
                logger.info(f"Preview (first 300 chars):")
                logger.info(f"{doc_text[:300]}...")
                
                if idx >= 10:  # Limit to first 10 chunks for readability
                    logger.info(f"\n... and {len(documents) - 10} more chunks")
                    break
        
        logger.info("\n" + "=" * 80)
        logger.info("Test similarity search:")
        logger.info("=" * 80)
        
        # Test with a sample query
        test_query = "test case"
        logger.info(f"\nTesting query: '{test_query}'")
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        try:
            docs_with_scores = vectorstore.similarity_search_with_score(test_query, k=3)
            if docs_with_scores:
                logger.info(f"✅ Found {len(docs_with_scores)} matches:")
                for idx, (doc, score) in enumerate(docs_with_scores, 1):
                    logger.info(f"\n  Match {idx} (score: {score:.4f}):")
                    logger.info(f"  {doc.page_content[:200]}...")
            else:
                logger.warning("⚠️ No matches found for test query")
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            logger.info("Trying standard retrieval...")
            docs = retriever.get_relevant_documents(test_query)
            logger.info(f"Found {len(docs)} documents")
        
    except Exception as e:
        logger.error(f"Error inspecting vector store: {e}", exc_info=True)

if __name__ == "__main__":
    inspect_vector_store()


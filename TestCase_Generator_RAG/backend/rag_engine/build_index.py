"""
RAG engine index builder.
Builds and manages vector embeddings from uploaded documents using ChromaDB.
"""

import os
import logging
from pathlib import Path
from typing import List, Union, Any
from io import BytesIO
from fastapi import UploadFile

import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config.settings import get_gemini_api_key

logger = logging.getLogger(__name__)


def _extract_text_from_file(file: Union[UploadFile, bytes, str, Path], filename: str = None) -> str:
    """
    Extract text from a file (PDF or TXT).
    
    Args:
        file: Either a FastAPI UploadFile object, bytes, or file path string/Path
        filename: Optional filename hint (useful when file is bytes)
        
    Returns:
        str: Extracted text content
    """
    text = ""
    
    if isinstance(file, UploadFile):
        # Handle FastAPI UploadFile - read synchronously
        file_name = (file.filename or "").lower()
        # Read file content - need to read bytes
        file.file.seek(0)  # Reset to beginning
        file_content = file.file.read()
        file.file.seek(0)  # Reset again for potential reuse
        
        if file_name.endswith('.pdf'):
            # Extract text from PDF
            pdf_file = PyPDF2.PdfReader(BytesIO(file_content))
            for page in pdf_file.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        elif file_name.endswith('.txt'):
            # Read text file
            text = file_content.decode('utf-8', errors='ignore')
        else:
            raise ValueError(f"Unsupported file type: {file_name}. Only PDF and TXT are supported.")
            
    elif isinstance(file, bytes):
        # Handle bytes directly (when file content is already read)
        file_name = (filename or "").lower()
        
        if file_name.endswith('.pdf'):
            # Extract text from PDF
            pdf_file = PyPDF2.PdfReader(BytesIO(file))
            for page in pdf_file.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        elif file_name.endswith('.txt'):
            # Read text file
            text = file.decode('utf-8', errors='ignore')
        else:
            raise ValueError(f"Unsupported file type: {file_name}. Only PDF and TXT are supported.")
            
    else:
        # Handle file path
        file_path = Path(file)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_name = file_path.name.lower()
        
        if file_name.endswith('.pdf'):
            # Extract text from PDF
            with open(file_path, 'rb') as f:
                pdf_file = PyPDF2.PdfReader(f)
                for page in pdf_file.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        elif file_name.endswith('.txt'):
            # Read text file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        else:
            raise ValueError(f"Unsupported file type: {file_name}. Only PDF and TXT are supported.")
    
    return text.strip()


def build_doc_index(
    uploaded_files: List[Union[UploadFile, tuple, str, Path]], 
    persist_dir: str = "./rag_engine/vector_store",
    force_rebuild: bool = False
) -> Any:
    """
    Build a document index from uploaded files using ChromaDB and Gemini embeddings.
    
    Args:
        uploaded_files: List of FastAPI UploadFile objects or file paths (str/Path)
        persist_dir: Directory to persist the vector store (default: ./rag_engine/vector_store)
        force_rebuild: If True, rebuild index even if persist_dir exists
        
    Returns:
        Chroma retriever object or vectorstore reference for similarity search
    """
    persist_path = Path(persist_dir)
    
    # ALWAYS delete existing database directory before creating new one to avoid corruption issues
    # This ensures we start with a clean slate
    import shutil
    if persist_path.exists():
        logger.info("🗑️ Cleaning up any existing vector store directory...")
        try:
            # Close any potential file handles
            import time
            time.sleep(0.5)
            shutil.rmtree(persist_path, ignore_errors=True)
            # Wait a bit for file handles to release
            time.sleep(0.5)
            logger.info("✅ Existing directory cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Could not fully delete directory: {e}")
            # Try to delete individual files
            try:
                for file in persist_path.glob("*"):
                    try:
                        if file.is_file():
                            file.unlink()
                        elif file.is_dir():
                            shutil.rmtree(file, ignore_errors=True)
                    except Exception:
                        pass
            except Exception:
                pass
    
    # Always create fresh directory
    persist_path.mkdir(parents=True, exist_ok=True)
    
    # Skip loading existing index - always rebuild for now to avoid corruption issues
    # In the future, we can add proper validation back
    if False and not force_rebuild and persist_path.exists():
        # Check if directory has ChromaDB files (ChromaDB creates various files)
        # Look for common ChromaDB persistence files
        has_chroma_files = (
            any(persist_path.glob("*.parquet")) or
            any(persist_path.glob("*.chroma")) or
            (persist_path / "chroma.sqlite3").exists() or
            (persist_path / "index").exists()
        )
        
        # Check if database file is corrupted (empty or too small)
        db_file = persist_path / "chroma.sqlite3"
        if db_file.exists() and db_file.stat().st_size == 0:
            logger.warning("⚠️ Found empty/corrupted database file. Deleting and rebuilding...")
            import shutil
            shutil.rmtree(persist_path, ignore_errors=True)
            persist_path.mkdir(parents=True, exist_ok=True)
            has_chroma_files = False  # Force rebuild
        
        if has_chroma_files:
            # Load existing index
            try:
                # Try Gemini embeddings first, fallback to local if not available
                try:
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key=get_gemini_api_key()
                    )
                    embeddings.embed_query("test")  # Test if it works
                except Exception:
                    # Fallback to local embeddings
                    logger.info("Using local embeddings for loading existing index")
                    embeddings = HuggingFaceEmbeddings(
                        model_name="all-MiniLM-L6-v2"
                    )
                
                vectorstore = Chroma(
                    persist_directory=str(persist_path),
                    embedding_function=embeddings
                )
                
                # Verify the database is valid by checking if we can access it
                try:
                    collection = vectorstore._collection
                    count = collection.count()
                    if count > 0:
                        logger.info(f"✅ Loaded existing vector index with {count} documents")
                    else:
                        logger.warning("⚠️ Database exists but is empty. Rebuilding...")
                        raise ValueError("Empty database")
                except Exception as db_error:
                    logger.warning(f"⚠️ Database appears corrupted: {db_error}. Rebuilding...")
                    import shutil
                    shutil.rmtree(persist_path, ignore_errors=True)
                    persist_path.mkdir(parents=True, exist_ok=True)
                    raise  # Trigger rebuild
                
                # Return retriever for similarity search
                retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}  # Get top 3 most similar chunks
                )
                return retriever
            except Exception as e:
                # If loading fails, delete corrupted database and rebuild
                logger.warning(f"⚠️ Failed to load existing index: {e}. Deleting corrupted database and rebuilding...")
                import shutil
                if persist_path.exists():
                    shutil.rmtree(persist_path, ignore_errors=True)
                    persist_path.mkdir(parents=True, exist_ok=True)
                # Continue to build new index
    
    # Build new index
    all_texts = []
    
    # Extract text from all files
    for file in uploaded_files:
        try:
            # Handle tuple (bytes, filename) from async file reading
            if isinstance(file, tuple) and len(file) == 2:
                file_content, filename = file
                text = _extract_text_from_file(file_content, filename=filename)
            else:
                text = _extract_text_from_file(file)
            
            if text and text.strip():
                all_texts.append(text)
                logger.info(f"✅ Successfully extracted {len(text)} characters from file")
                # DEBUG: Show preview of extracted text
                preview = text[:300] if len(text) > 300 else text
                logger.info(f"   Preview (first 300 chars): {preview}...")
            else:
                logger.warning(f"⚠️ No text extracted from file")
        except Exception as e:
            logger.error(f"Failed to extract text from file: {e}", exc_info=True)
            continue
    
    if not all_texts:
        raise ValueError("No text content could be extracted from the uploaded files.")
    
    # Chunk text using LangChain text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for idx, text in enumerate(all_texts, 1):
        text_chunks = text_splitter.split_text(text)
        chunks.extend(text_chunks)
        logger.info(f"📄 File {idx}: Split into {len(text_chunks)} chunks")
    
    if not chunks:
        raise ValueError("No text chunks could be created from the uploaded files.")
    
    logger.info(f"📚 Total chunks created: {len(chunks)}")
    logger.info(f"   Sample chunk (first 200 chars): {chunks[0][:200]}..." if chunks else "")
    
    # Initialize embeddings - Use local embeddings to avoid API quota issues
    # Free tier doesn't support Gemini embeddings API (limit: 0)
    try:
        logger.info("🔄 Attempting to use Gemini embeddings API...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=get_gemini_api_key()
        )
        # Test if embeddings work
        try:
            embeddings.embed_query("test")
            logger.info("✅ Using Gemini embeddings API")
        except Exception as e:
            logger.warning(f"⚠️ Gemini embeddings API not available (likely free tier): {e}")
            logger.info("🔄 Falling back to local embeddings (sentence-transformers)...")
            raise  # Trigger fallback
    except Exception as e:
        # Fallback to local embeddings (free, no API calls needed)
        logger.info("✅ Using local embeddings (sentence-transformers) - no API quota needed")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"  # Lightweight, fast local model
        )
    
    # Create Chroma vectorstore with persistence
    # Always clean up any existing corrupted database before creating new one
    import shutil
    import time
    
    # Force delete the directory multiple times to ensure it's gone
    max_attempts = 3
    for attempt in range(max_attempts):
        if persist_path.exists():
            logger.info(f"🗑️ Attempt {attempt + 1}/{max_attempts}: Cleaning up vector store directory...")
            try:
                # Force delete all files
                for item in persist_path.iterdir():
                    try:
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item, ignore_errors=True)
                    except Exception:
                        pass
                
                # Now delete the directory itself
                shutil.rmtree(persist_path, ignore_errors=True)
                time.sleep(0.5)  # Wait for file handles to release
                
                if not persist_path.exists():
                    logger.info("✅ Directory successfully deleted")
                    break
            except Exception as e:
                logger.warning(f"⚠️ Attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(1)  # Wait longer before retry
                else:
                    logger.error("❌ Could not delete directory after multiple attempts")
                    # Try to use a different directory name
                    import uuid
                    persist_path = Path(persist_dir) / f"vector_store_{uuid.uuid4().hex[:8]}"
                    logger.info(f"🔄 Using alternative directory: {persist_path}")
    
    # Ensure directory exists and is clean
    persist_path.mkdir(parents=True, exist_ok=True)
    
    # Create collection name with timestamp to avoid conflicts
    import time as time_module
    collection_name = f"documents_{int(time_module.time())}"
    
    logger.info(f"Creating new vectorstore with {len(chunks)} chunks...")
    logger.info(f"Collection name: {collection_name}")
    
    # Use in-memory mode by default to avoid database corruption issues
    # This is more reliable and avoids the "no such table: tenants" error
    logger.info("🔄 Using in-memory vectorstore to avoid database corruption issues...")
    
    try:
        # Create in-memory vectorstore first (most reliable)
        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            collection_name=collection_name
        )
        logger.info("✅ In-memory vectorstore created successfully")
        logger.info("📌 Note: Using in-memory mode (database persistence disabled to avoid corruption)")
                
    except Exception as e:
        logger.error(f"❌ Error creating vectorstore: {e}")
        raise ValueError(f"Failed to create vectorstore: {e}")
    
    # Persist the vectorstore (if it's a persistent store)
    try:
        if hasattr(vectorstore, 'persist'):
            vectorstore.persist()
    except Exception:
        pass  # In-memory stores don't need persistence
    
    # Return retriever for similarity search
    # Using k=3 to get top 3 matches (no threshold filter, will show all matches)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Get top 3 most similar chunks
    )
    
    logger.info("✅ Vector index built successfully")
    return retriever


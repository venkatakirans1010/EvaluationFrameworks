"""
RAG engine context retrieval.
Retrieves relevant context from vector store based on query similarity.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def retrieve_context(query: str, retriever: Any, top_k: int = 3) -> str:
    """
    Retrieve relevant context from vector store using similarity search.
    
    Args:
        query: The search query string
        retriever: LangChain retriever object (from ChromaDB vectorstore)
        top_k: Number of top document chunks to retrieve (default: 3)
        
    Returns:
        str: Combined context text from retrieved chunks, separated by headers.
             Returns empty string if no results found.
    """
    if not query or not query.strip():
        logger.warning("Empty query provided to retrieve_context")
        return ""
    
    try:
        # DEBUG: Print query text
        logger.info("=" * 80)
        logger.info("🔍 RAG QUERY DEBUG")
        logger.info("=" * 80)
        logger.info(f"QUERY TEXT: {query}")
        logger.info(f"QUERY LENGTH: {len(query)} characters")
        logger.info("")
        
        # DEBUG: Try to get sample documents from vectorstore
        try:
            # Access the underlying vectorstore to inspect contents
            vectorstore = retriever.vectorstore if hasattr(retriever, 'vectorstore') else None
            if vectorstore:
                # Get all documents (or sample) to see what's indexed
                collection = vectorstore._collection if hasattr(vectorstore, '_collection') else None
                if collection:
                    # Get count of documents
                    count = collection.count() if hasattr(collection, 'count') else 0
                    logger.info(f"📚 INDEXED DOCUMENTS COUNT: {count}")
                    
                    # Try to get sample documents (first few)
                    if count > 0:
                        try:
                            # Get sample results (without query)
                            results = collection.get(limit=min(3, count))
                            if results and 'documents' in results and results['documents']:
                                logger.info("📄 SAMPLE INDEXED CHUNKS:")
                                for idx, doc_chunk in enumerate(results['documents'][:3], 1):
                                    preview = doc_chunk[:200] if len(doc_chunk) > 200 else doc_chunk
                                    logger.info(f"  Chunk {idx} (first 200 chars): {preview}...")
                                logger.info("")
                        except Exception as e:
                            logger.debug(f"Could not retrieve sample documents: {e}")
        except Exception as e:
            logger.debug(f"Could not inspect vectorstore: {e}")
        
        # Perform similarity search
        # Try to get documents with scores if available
        try:
            # Check if we can use similarity_search_with_score
            vectorstore = retriever.vectorstore if hasattr(retriever, 'vectorstore') else None
            if vectorstore and hasattr(vectorstore, 'similarity_search_with_score'):
                docs_with_scores = vectorstore.similarity_search_with_score(query, k=top_k)
                if docs_with_scores:
                    logger.info(f"📊 Similarity scores:")
                    for idx, (doc, score) in enumerate(docs_with_scores, 1):
                        logger.info(f"  Chunk {idx}: score={score:.4f}")
                    docs = [doc for doc, score in docs_with_scores]
                else:
                    docs = []
            else:
                # Fallback to standard retrieval
                docs = retriever.get_relevant_documents(query)
        except Exception as e:
            logger.warning(f"Error during similarity search: {e}, trying standard retrieval")
            docs = retriever.get_relevant_documents(query)
        
        if not docs:
            logger.warning(f"❌ No relevant documents found for query: {query[:100]}")
            logger.info("💡 TIP: Check if the query text matches terms/concepts in your documents")
            logger.info("=" * 80)
            return ""
        
        # Limit to top_k results
        docs = docs[:top_k]
        
        # Extract and sanitize chunks
        context_parts = []
        max_chunk_length = 2000
        
        for idx, doc in enumerate(docs, start=1):
            # Extract page content
            chunk_text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            original_length = len(chunk_text)
            
            # Sanitize: truncate to safe length to prevent prompt injection
            if len(chunk_text) > max_chunk_length:
                chunk_text = chunk_text[:max_chunk_length] + "... [truncated]"
                logger.debug(f"Truncated chunk {idx} from {original_length} to {max_chunk_length} chars")
            
            # Create header for this chunk with clear source identification
            source_info = "Unknown Source"
            if hasattr(doc, 'metadata') and doc.metadata:
                source = doc.metadata.get('source', 'Unknown')
                # Extract filename from source path if it's a path
                if '/' in source or '\\' in source:
                    source_info = source.split('/')[-1].split('\\')[-1]
                else:
                    source_info = source
            else:
                source_info = f"Document {idx}"
            
            header = f"\n--- Context Chunk {idx} (Source: {source_info}) ---\n"
            
            # Combine header and content
            context_parts.append(header + chunk_text)
        
        # Join all chunks with double newline separator
        combined_context = "\n\n".join(context_parts)
        
        logger.info(f"✅ Retrieved {len(docs)} document chunks for query: {query[:100]}")
        logger.info("=" * 80)
        return combined_context
        
    except Exception as e:
        logger.error(f"Error retrieving context for query '{query[:100]}': {str(e)}")
        return ""


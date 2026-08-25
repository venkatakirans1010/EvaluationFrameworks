# Log Analysis & Debugging Guide

## Current Status

### Services Running
- ✅ Backend: http://localhost:8000
- ✅ Frontend: http://localhost:3000
- ✅ Logging: Enabled (logs written to `backend/logs/app.log`)

### Issue Identified
🔴 **Vector Store is EMPTY (0 documents indexed)**

This explains why you're seeing:
```
RAG No Match
RAG attempted but no matching context found: 1 document(s) indexed but similarity search returned no results for query
```

## Root Cause Analysis

The vector store shows 0 documents, which means either:
1. Documents are being uploaded but not indexed properly
2. Index is being built but not persisted
3. Index is being cleared between requests
4. Documents are indexed but stored in a different location

## Debug Logging Enabled

The following debug information will appear in logs when you generate test cases:

### During Index Building (`build_index.py`):
- ✅ File extraction status
- ✅ Preview of extracted text (first 300 chars)
- ✅ Number of chunks created per file
- ✅ Total chunks created
- ✅ Sample chunk preview

### During Retrieval (`retrieve_context.py`):
- 🔍 Query text being searched
- 📚 Number of indexed documents
- 📄 Sample chunks from indexed documents (first 200 chars)
- 📊 Similarity scores for matched documents

## How to Debug

1. **Generate test cases** via the UI with a document uploaded
2. **Check logs** at `backend/logs/app.log` for:
   - File extraction messages
   - Index building messages
   - Query text used
   - Indexed document count
   - Sample chunks

3. **Run diagnostic script**:
   ```bash
   cd backend
   python inspect_vector_store.py
   ```

## Expected Log Flow

When generating test cases, you should see:
```
INFO - Building RAG index with X files
INFO - ✅ Successfully extracted X characters from file
INFO -    Preview (first 300 chars): [document text]...
INFO - 📄 File 1: Split into X chunks
INFO - 📚 Total chunks created: X
INFO - ✅ Vector index built and persisted successfully
INFO - 🔍 Using query for RAG retrieval: '[query text]'
INFO - ================================================================================
INFO - 🔍 RAG QUERY DEBUG
INFO - ================================================================================
INFO - QUERY TEXT: [query]
INFO - QUERY LENGTH: X characters
INFO - 📚 INDEXED DOCUMENTS COUNT: X
INFO - 📄 SAMPLE INDEXED CHUNKS:
INFO -   Chunk 1 (first 200 chars): [text]...
```

## Next Steps

1. Generate test cases with a document to populate logs
2. Review logs to see where the indexing fails
3. Compare query text vs indexed document text
4. Check similarity scores to understand matching


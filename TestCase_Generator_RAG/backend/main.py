"""
Main FastAPI application entry point.
Defines API endpoints for uploading documents, fetching Jira issues, and generating test cases.
"""

import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from jira_integration.fetch_jira import fetch_jira_details
from rag_engine.build_index import build_doc_index
from rag_engine.retrieve_context import retrieve_context
from ai_generator.generate_ui_cases import generate_ui_test_cases
from exporter.export_excel import export_markdown_table_to_excel

# Configure logging to both console and file
log_dir = Path("./logs")
log_dir.mkdir(exist_ok=True)

# Create formatters
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# File handler
file_handler = logging.FileHandler(log_dir / "app.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)
logger.info("=" * 80)
logger.info("Application starting - Logging configured")
logger.info("=" * 80)

# Create FastAPI app
app = FastAPI(title="AI UI Test Case Generator", version="1.0.0")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory for generated Excel files
GENERATED_FILES_DIR = Path("./generated_files")
GENERATED_FILES_DIR.mkdir(exist_ok=True)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/generate_test_cases")
async def generate_test_cases(
    jira_id: str = Form(...),
    files: List[UploadFile] = File(default=[])
):
    """
    Generate UI test cases from Jira story and uploaded documents.
    
    Args:
        jira_id: Jira issue key (e.g., 'PROJ-123')
        files: List of uploaded files (PDF or TXT) for RAG context
        
    Returns:
        JSON with markdown content and excel file path
    """
    try:
        # Step 1: Fetch Jira details
        logger.info(f"Fetching Jira details for issue: {jira_id}")
        try:
            jira_story = fetch_jira_details(jira_id)
        except ValueError as e:
            logger.error(f"Error fetching Jira issue {jira_id}: {str(e)}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error fetching Jira issue {jira_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch Jira issue: {str(e)}")
        
        # Step 2: Build or load RAG index
        logger.info(f"Building RAG index with {len(files)} files")
        processed_files = []  # Initialize outside the if block
        retriever = None
        
        try:
            if files:
                # Read file contents asynchronously
                for file in files:
                    try:
                        # Read file content
                        file_content = await file.read()
                        if len(file_content) == 0:
                            logger.warning(f"File {file.filename} is empty, skipping")
                            continue
                        # Store file content and filename for processing
                        processed_files.append((file_content, file.filename))
                        logger.info(f"Read file {file.filename}: {len(file_content)} bytes")
                    except Exception as e:
                        logger.error(f"Error reading file {file.filename}: {str(e)}")
                        continue
                
                if not processed_files:
                    logger.warning("No valid files provided after reading")
                else:
                    # Pass file contents as list of tuples (bytes, filename)
                    retriever = build_doc_index(uploaded_files=processed_files)
            else:
                logger.warning("No files provided, skipping RAG index building")
        except Exception as e:
            logger.error(f"Error building RAG index: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to build document index: {str(e)}"
            )
        
        # Step 3: Retrieve context
        logger.info("Retrieving context from RAG")
        rag_used = False
        rag_files_count = len(processed_files)
        context_length = 0
        rag_reason = ""
        
        try:
            # Use summary as query, fallback to description if summary is empty
            query = jira_story.get('summary', '').strip()
            if not query:
                query = jira_story.get('description', '').strip()
            if not query:
                query = jira_id  # Fallback to Jira ID
            
            logger.info(f"🔍 Using query for RAG retrieval: '{query}'")
            
            if retriever:
                context_text = retrieve_context(query=query, retriever=retriever)
                context_length = len(context_text)
                rag_used = context_length > 0
                
                if rag_used:
                    rag_reason = f"RAG enabled: Using {rag_files_count} uploaded document(s) for context ({context_length} chars retrieved)"
                else:
                    rag_reason = f"RAG attempted but no matching context found: {rag_files_count} document(s) indexed but similarity search returned no results for query"
                
                logger.info(f"RAG context retrieved: {context_length} characters")
            else:
                context_text = ""
                if rag_files_count > 0:
                    rag_reason = f"RAG disabled: {rag_files_count} file(s) uploaded but index building failed or files were empty/invalid"
                else:
                    rag_reason = "RAG disabled: No documents uploaded"
                logger.info("No retriever available, using empty context")
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            # Continue with empty context if retrieval fails
            context_text = ""
            rag_used = False
            if rag_files_count > 0:
                rag_reason = f"RAG disabled: Error retrieving context - {str(e)[:100]}"
            else:
                rag_reason = "RAG disabled: No documents uploaded"
        
        # Step 4: Generate test cases
        logger.info("Generating UI test cases with Gemini")
        try:
            markdown_content = generate_ui_test_cases(
                jira_story=jira_story,
                context_text=context_text,
                max_cases=50
            )
        except ValueError as e:
            logger.error(f"Error generating test cases: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error generating test cases: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to generate test cases: {str(e)}"
            )
        
        # Step 5: Export to Excel
        logger.info("Exporting test cases to Excel")
        try:
            excel_filename = f"testcases_{jira_id.replace('-', '_')}.xlsx"
            excel_path = GENERATED_FILES_DIR / excel_filename
            absolute_path = export_markdown_table_to_excel(
                markdown_table=markdown_content,
                output_path=str(excel_path)
            )
        except ValueError as e:
            logger.error(f"Error exporting to Excel: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error exporting to Excel: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to export to Excel: {str(e)}"
            )
        
        # Step 6: Return response
        return {
            "markdown": markdown_content,
            "excel_path": f"/download/{excel_filename}",
            "rag_status": {
                "enabled": rag_used,
                "files_uploaded": rag_files_count,
                "context_length": context_length,
                "message": rag_reason,
                "index_built": retriever is not None
            }
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_test_cases: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Serve generated Excel files.
    
    Args:
        filename: Name of the Excel file to download
        
    Returns:
        FileResponse with the Excel file
    """
    try:
        # Security: validate filename to prevent directory traversal
        safe_filename = Path(filename).name  # Remove any path components
        
        file_path = GENERATED_FILES_DIR / safe_filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
        
        # Verify it's an Excel file
        if not safe_filename.endswith('.xlsx'):
            raise HTTPException(status_code=400, detail="Only Excel files (.xlsx) can be downloaded")
        
        return FileResponse(
            path=str(file_path),
            filename=safe_filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to serve file: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


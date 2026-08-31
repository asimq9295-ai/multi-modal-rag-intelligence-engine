"""
FastAPI routes for the Multi-Modal RAG application.
Enhanced with improved error handling, logging, validation, and documentation.
"""

import os
import uuid
import shutil
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends
from fastapi.responses import JSONResponse

from .models import (
    UploadResponse, 
    QueryRequest, 
    QueryResponse, 
    HealthResponse,
    FileInfo,
    ErrorResponse,
    VectorstoreStats
)
from .logic import (
    process_document, 
    create_and_store_embeddings, 
    perform_rag_query, 
    get_vectorstore_stats,
    delete_document_embeddings
)
from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter()

# Constants - Load from settings
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.csv', '.png', '.jpg', '.jpeg', '.db'}


def validate_file_extension(filename: str) -> str:
    """
    Validate file extension and return it if valid.
    
    Args:
        filename: Name of the uploaded file
        
    Returns:
        File extension in lowercase
        
    Raises:
        HTTPException: If file type is not supported
    """
    file_extension = Path(filename).suffix.lower()
    
    if file_extension not in ALLOWED_EXTENSIONS:
        logger.warning(f"Rejected file upload with unsupported extension: {file_extension}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_file_type",
                "message": f"Unsupported file type: {file_extension}",
                "supported_types": list(ALLOWED_EXTENSIONS)
            }
        )
    
    return file_extension


def validate_file_size(file_size: int, filename: str) -> None:
    """
    Validate file size is within limits.
    
    Args:
        file_size: Size of file in bytes
        filename: Name of the file
        
    Raises:
        HTTPException: If file is too large
    """
    if file_size > settings.max_file_size_bytes:
        logger.warning(f"Rejected file upload - too large: {filename} ({file_size} bytes)")
        raise HTTPException(
            status_code=413,
            detail={
                "error": "file_too_large",
                "message": f"File size ({file_size / 1024 / 1024:.2f}MB) exceeds maximum allowed ({settings.max_file_size_bytes / 1024 / 1024}MB)",
                "max_size_bytes": settings.max_file_size_bytes
            }
        )


@router.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """
    Root endpoint returning API information.
    
    Returns:
        HealthResponse with welcome message and healthy status
    """
    return HealthResponse(
        message="Welcome to the Multi-Modal RAG API",
        status="healthy"
    )


@router.get("/health", response_model=dict, tags=["Health"])
async def health_check():
    """
    Detailed health check with vectorstore statistics.
    
    Returns comprehensive health information including:
    - API status
    - Vectorstore connection status
    - Document count
    - System information
    """
    try:
        stats = get_vectorstore_stats()
        
        return {
            "status": "healthy",
            "message": "Multi-Modal RAG API is running",
            "timestamp": datetime.utcnow().isoformat(),
            "vectorstore": stats,
            "version": "1.1.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "degraded",
            "message": f"API running but vectorstore unavailable: {str(e)}",
            "timestamp": datetime.utcnow().isoformat(),
            "vectorstore": {"status": "error", "total_documents": 0}
        }


@router.post(
    "/upload", 
    response_model=UploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type or empty file"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    },
    tags=["Documents"]
)
async def upload_file(file: UploadFile = File(..., description="Document file to upload and process")):
    """
    Upload and process a document file.
    
    Supports multiple file formats including:
    - **PDF** (.pdf) - Portable Document Format
    - **Word** (.docx) - Microsoft Word documents
    - **Text** (.txt) - Plain text files
    - **CSV** (.csv) - Comma-separated values
    - **Images** (.png, .jpg, .jpeg) - Images processed with OCR
    - **Database** (.db) - SQLite database files
    
    The document will be:
    1. Validated for type and size
    2. Saved to storage
    3. Processed and chunked
    4. Embedded using OpenAI embeddings
    5. Stored in ChromaDB for retrieval
    
    Returns:
        UploadResponse with file_id, filename, and processing message
    """
    logger.info(f"Received upload request for file: {file.filename}")
    
    # Validate file extension
    file_extension = validate_file_extension(file.filename)
    
    # Read file content
    try:
        file_content = await file.read()
        file_size = len(file_content)
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={"error": "read_error", "message": f"Failed to read file: {str(e)}"}
        )
    
    # Validate file size
    validate_file_size(file_size, file.filename)
    
    # Check for empty file
    if file_size == 0:
        logger.warning(f"Rejected empty file: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail={"error": "empty_file", "message": "The uploaded file is empty"}
        )
    
    # Generate unique file ID and path
    file_id = str(uuid.uuid4())
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in '.-_')
    file_path = os.path.join(settings.uploads_path, f"{file_id}_{safe_filename}")
    
    try:
        # Save file to disk
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        
        logger.info(f"File saved: {file_path} ({file_size} bytes)")
        
        # Process document
        try:
            chunks = process_document(file_path, file.filename)
            
            if not chunks:
                logger.warning(f"No content extracted from file: {file.filename}")
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "no_content",
                        "message": "No content could be extracted from the file. The file may be empty, corrupted, or in an unsupported format."
                    }
                )
            
            # Create and store embeddings
            create_and_store_embeddings(chunks, file_id)
            
            logger.info(f"Successfully processed file {file.filename}: {len(chunks)} chunks created")
            
            return UploadResponse(
                file_id=file_id,
                filename=file.filename,
                message=f"File processed successfully. {len(chunks)} chunks created and indexed.",
                chunks_created=len(chunks),
                file_size=file_size
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            # Clean up saved file on processing error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "processing_error",
                    "message": f"Error processing file: {str(e)}"
                }
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during file upload: {str(e)}")
        # Clean up on any error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "upload_error",
                "message": f"Unexpected error during file upload: {str(e)}"
            }
        )


@router.post(
    "/query", 
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Query processing error"}
    },
    tags=["Query"]
)
async def query_api(request: QueryRequest):
    """
    Query the RAG system with a text question.
    
    This endpoint performs semantic search on the indexed document(s) and generates
    an AI-powered response based on the relevant context.
    
    **Supports two modes:**
    - **Single document**: Provide `file_id` to query one document
    - **Multi-document**: Provide `file_ids` list to query across multiple documents
    
    **Search modes:**
    - **Hybrid search** (default): Combines vector similarity + BM25 keyword matching
    - **Vector only**: Set `use_hybrid_search=false` for semantic search only
    
    Args:
        request: QueryRequest containing question and file_id(s)
        
    Returns:
        QueryResponse with answer, context, and source documents
    """
    # Get file IDs using the helper method
    file_ids = request.get_file_ids()
    
    logger.info(f"Received query for file_id: {request.file_id}, file_ids: {request.file_ids}, resolved: {file_ids}")
    
    # Validate question
    if not request.question or not request.question.strip():
        logger.warning("Received empty question")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "empty_question",
                "message": "Question cannot be empty"
            }
        )
    
    # Validate that we have at least one file_id
    if not file_ids:
        logger.warning("Received empty file_id(s)")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "empty_file_id",
                "message": "File ID(s) cannot be empty. Please upload or select document(s) first."
            }
        )
    
    # Validate file_id format (should be UUID) for all file_ids
    for fid in file_ids:
        if not fid or not fid.strip():
            continue
        try:
            uuid.UUID(fid)
        except ValueError:
            logger.warning(f"Invalid file_id format: {fid}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_file_id",
                    "message": f"Invalid file ID format: {fid}"
                }
            )
    
    try:
        response = perform_rag_query(request)
        
        logger.info(f"Query successful for {len(file_ids)} file(s), sources found: {len(response.sources)}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "query_error",
                "message": f"Error processing query: {str(e)}"
            }
        )


@router.get(
    "/files", 
    response_model=List[FileInfo],
    tags=["Documents"]
)
async def list_uploaded_files(
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of files to return"),
    offset: int = Query(default=0, ge=0, description="Number of files to skip")
):
    """
    List all uploaded files with pagination support.
    
    Returns information about uploaded documents including:
    - File ID (for querying)
    - Original filename
    - File size
    - Upload timestamp (from file modification time)
    
    Args:
        limit: Maximum number of files to return (default: 100)
        offset: Number of files to skip for pagination
        
    Returns:
        List of FileInfo objects
    """
    try:
        files = []
        uploads_dir = Path(settings.uploads_path)
        
        if uploads_dir.exists():
            all_files = sorted(
                uploads_dir.iterdir(),
                key=lambda x: x.stat().st_mtime,
                reverse=True  # Most recent first
            )
            
            # Apply pagination
            paginated_files = list(all_files)[offset:offset + limit]
            
            for file_path in paginated_files:
                if file_path.is_file():
                    filename = file_path.name
                    if '_' in filename:
                        file_id = filename.split('_')[0]
                        original_name = '_'.join(filename.split('_')[1:])
                        
                        stat = file_path.stat()
                        files.append(FileInfo(
                            file_id=file_id,
                            filename=original_name,
                            upload_path=str(file_path),
                            size_bytes=stat.st_size,
                            uploaded_at=datetime.fromtimestamp(stat.st_mtime).isoformat()
                        ))
        
        logger.info(f"Listed {len(files)} files (offset: {offset}, limit: {limit})")
        return files
        
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "list_error",
                "message": f"Error listing files: {str(e)}"
            }
        )


@router.delete(
    "/files/{file_id}",
    response_model=dict,
    tags=["Documents"]
)
async def delete_file(file_id: str):
    """
    Delete an uploaded file and its embeddings.
    
    This will:
    1. Remove the file from storage
    2. Delete associated embeddings from the vectorstore
    
    Args:
        file_id: UUID of the file to delete
        
    Returns:
        Confirmation message
    """
    logger.info(f"Received delete request for file_id: {file_id}")
    
    # Validate file_id format
    try:
        uuid.UUID(file_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_file_id", "message": "Invalid file ID format"}
        )
    
    try:
        # Find and delete the file
        uploads_dir = Path(settings.uploads_path)
        file_found = False
        
        if uploads_dir.exists():
            for file_path in uploads_dir.iterdir():
                if file_path.is_file() and file_path.name.startswith(f"{file_id}_"):
                    os.remove(file_path)
                    file_found = True
                    logger.info(f"Deleted file: {file_path}")
                    break
        
        # Delete embeddings
        try:
            delete_document_embeddings(file_id)
        except Exception as e:
            logger.warning(f"Could not delete embeddings for {file_id}: {str(e)}")
        
        if file_found:
            return {
                "message": f"File {file_id} deleted successfully",
                "file_id": file_id
            }
        else:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "message": f"File with ID {file_id} not found"}
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "delete_error", "message": f"Error deleting file: {str(e)}"}
        )

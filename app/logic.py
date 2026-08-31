"""
Business logic for the Multi-Modal RAG application.
Handles document processing, embedding creation, and RAG pipeline.
Enhanced with improved error handling, logging, caching, and performance optimizations.
Now includes hybrid search (vector + BM25) and multi-document RAG support.
"""

import os
import io
import re
import sqlite3
import base64
import time
import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from functools import lru_cache
from collections import defaultdict

import pandas as pd
import pytesseract
from PIL import Image
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader
)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# BM25 for keyword search
from rank_bm25 import BM25Okapi

from .config import settings
from .models import QueryRequest, QueryResponse, Source

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Caching and Singleton Patterns
# =============================================================================

_embeddings_instance: Optional[OpenAIEmbeddings] = None
_openai_client: Optional[OpenAI] = None
_bm25_index: Optional[Dict[str, Any]] = None  # Cache for BM25 index


def get_embeddings() -> OpenAIEmbeddings:
    """
    Get or create a singleton OpenAI embeddings instance.
    
    Returns:
        OpenAIEmbeddings instance
    """
    global _embeddings_instance
    
    if _embeddings_instance is None:
        logger.info("Initializing OpenAI embeddings instance")
        _embeddings_instance = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model=settings.openai_embedding_model
        )
    
    return _embeddings_instance


def get_openai_client() -> OpenAI:
    """
    Get or create a singleton OpenAI client instance.
    
    Returns:
        OpenAI client instance
    """
    global _openai_client
    
    if _openai_client is None:
        logger.info("Initializing OpenAI client instance")
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    
    return _openai_client


def get_vectorstore() -> Chroma:
    """
    Get ChromaDB vectorstore instance.
    
    Returns:
        Chroma vectorstore instance
    """
    return Chroma(
        persist_directory=settings.chroma_db_path,
        embedding_function=get_embeddings()
    )


# =============================================================================
# Hybrid Search (Vector + BM25)
# =============================================================================

def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text for BM25 indexing.
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of lowercase tokens
    """
    # Simple tokenization: lowercase, split on non-alphanumeric, filter short tokens
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [t for t in tokens if len(t) > 2]  # Filter tokens shorter than 3 chars


def build_bm25_index(documents: List[Tuple[Document, str]]) -> Tuple[BM25Okapi, List[Tuple[Document, str]]]:
    """
    Build BM25 index from documents.
    
    Args:
        documents: List of (Document, doc_id) tuples
        
    Returns:
        Tuple of (BM25 index, document list for lookup)
    """
    logger.info(f"Building BM25 index for {len(documents)} documents")
    
    # Tokenize all documents
    tokenized_docs = [tokenize_text(doc.page_content) for doc, _ in documents]
    
    # Build BM25 index
    bm25 = BM25Okapi(tokenized_docs)
    
    return bm25, documents


def bm25_search(
    query: str,
    bm25_index: BM25Okapi,
    documents: List[Tuple[Document, str]],
    k: int = 10
) -> List[Tuple[Document, str, float]]:
    """
    Perform BM25 keyword search.
    
    Args:
        query: Search query
        bm25_index: Pre-built BM25 index
        documents: List of (Document, doc_id) tuples
        k: Number of results to return
        
    Returns:
        List of (Document, doc_id, score) tuples sorted by score descending
    """
    # Tokenize query
    query_tokens = tokenize_text(query)
    
    if not query_tokens:
        return []
    
    # Get BM25 scores
    scores = bm25_index.get_scores(query_tokens)
    
    # Combine documents with scores and sort
    doc_scores = [(doc, doc_id, score) for (doc, doc_id), score in zip(documents, scores)]
    doc_scores.sort(key=lambda x: x[2], reverse=True)
    
    # Return top k
    return doc_scores[:k]


def reciprocal_rank_fusion(
    vector_results: List[Tuple[Document, str, float]],
    bm25_results: List[Tuple[Document, str, float]],
    k: int = 60,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5
) -> List[Tuple[Document, str, float, float, float]]:
    """
    Combine vector and BM25 results using Reciprocal Rank Fusion (RRF).
    
    RRF formula: score = sum(1 / (k + rank)) for each result list
    
    Args:
        vector_results: Results from vector search (doc, doc_id, score)
        bm25_results: Results from BM25 search (doc, doc_id, score)
        k: RRF constant (default 60, standard in literature)
        vector_weight: Weight for vector search scores
        bm25_weight: Weight for BM25 search scores
        
    Returns:
        List of (Document, doc_id, combined_score, vector_score, bm25_score) tuples
    """
    # Create document lookup by content hash (to handle duplicates)
    doc_scores = {}
    
    # Process vector results
    for rank, (doc, doc_id, score) in enumerate(vector_results, 1):
        # Use content + file_id as key to identify unique chunks
        key = f"{doc_id}:{hash(doc.page_content)}"
        if key not in doc_scores:
            doc_scores[key] = {
                'doc': doc,
                'doc_id': doc_id,
                'vector_rrf': 0,
                'bm25_rrf': 0,
                'vector_score': 0,
                'bm25_score': 0
            }
        doc_scores[key]['vector_rrf'] = vector_weight / (k + rank)
        doc_scores[key]['vector_score'] = score
    
    # Process BM25 results
    for rank, (doc, doc_id, score) in enumerate(bm25_results, 1):
        key = f"{doc_id}:{hash(doc.page_content)}"
        if key not in doc_scores:
            doc_scores[key] = {
                'doc': doc,
                'doc_id': doc_id,
                'vector_rrf': 0,
                'bm25_rrf': 0,
                'vector_score': 0,
                'bm25_score': 0
            }
        doc_scores[key]['bm25_rrf'] = bm25_weight / (k + rank)
        # Normalize BM25 score to 0-1 range
        max_bm25 = max((s for _, _, s in bm25_results), default=1) or 1
        doc_scores[key]['bm25_score'] = score / max_bm25
    
    # Combine scores and sort
    results = []
    for key, data in doc_scores.items():
        combined_score = data['vector_rrf'] + data['bm25_rrf']
        results.append((
            data['doc'],
            data['doc_id'],
            combined_score,
            data['vector_score'],
            data['bm25_score']
        ))
    
    # Sort by combined score
    results.sort(key=lambda x: x[2], reverse=True)
    
    return results


def hybrid_search(
    query: str,
    file_ids: List[str],
    k: int = 10,
    use_hybrid: bool = True
) -> Tuple[List[Tuple[Document, float, float, float, str]], str]:
    """
    Perform hybrid search combining vector similarity and BM25 keyword search.
    
    Args:
        query: Search query
        file_ids: List of file IDs to search within
        k: Number of results to return
        use_hybrid: Whether to use hybrid search or vector-only
        
    Returns:
        Tuple of (results list, search_method)
        Results: List of (Document, combined_score, vector_score, bm25_score, search_type)
    """
    logger.info(f"Performing {'hybrid' if use_hybrid else 'vector'} search for {len(file_ids)} files")
    
    vectorstore = get_vectorstore()
    collection = vectorstore._collection
    
    # Build filter for multiple file_ids
    if len(file_ids) == 1:
        filter_dict = {"file_id": file_ids[0]}
    else:
        filter_dict = {"file_id": {"$in": file_ids}}
    
    # Get all documents matching file_ids for BM25 indexing
    all_docs_data = collection.get(
        where=filter_dict,
        include=["documents", "metadatas"]
    )
    
    if not all_docs_data['ids']:
        return [], "none"
    
    # Reconstruct Document objects
    all_documents = []
    for doc_id, content, metadata in zip(
        all_docs_data['ids'],
        all_docs_data['documents'],
        all_docs_data['metadatas']
    ):
        doc = Document(page_content=content, metadata=metadata or {})
        all_documents.append((doc, doc_id))
    
    # Vector search
    vector_results_raw = vectorstore.similarity_search_with_score(
        query,
        k=k * 2,  # Get more for fusion
        filter=filter_dict
    )
    
    # Convert to standard format (doc, doc_id, score)
    # Find doc_id by matching content
    content_to_id = {doc.page_content: doc_id for doc, doc_id in all_documents}
    vector_results = []
    for doc, score in vector_results_raw:
        doc_id = content_to_id.get(doc.page_content, "unknown")
        # Convert distance to similarity
        similarity = max(0, 1 - score) if score < 1 else 1 / (1 + score)
        vector_results.append((doc, doc_id, similarity))
    
    if not use_hybrid:
        # Vector-only search
        results = [
            (doc, score, score, 0.0, "vector")
            for doc, doc_id, score in vector_results[:k]
        ]
        return results, "vector"
    
    # Build BM25 index and search
    bm25_index, indexed_docs = build_bm25_index(all_documents)
    bm25_results = bm25_search(query, bm25_index, indexed_docs, k=k * 2)
    
    # Fuse results using RRF
    fused_results = reciprocal_rank_fusion(vector_results, bm25_results)
    
    # Format output
    results = []
    for doc, doc_id, combined, vector_score, bm25_score in fused_results[:k]:
        # Determine search type based on which method found it
        if vector_score > 0 and bm25_score > 0:
            search_type = "hybrid"
        elif vector_score > 0:
            search_type = "vector"
        else:
            search_type = "bm25"
        
        # Normalize combined score to 0-1
        results.append((doc, combined, vector_score, bm25_score, search_type))
    
    # Normalize combined scores
    if results:
        max_combined = max(r[1] for r in results) or 1
        results = [
            (doc, score / max_combined, vs, bs, st)
            for doc, score, vs, bs, st in results
        ]
    
    return results, "hybrid"


# =============================================================================
# Text Processing
# =============================================================================

@lru_cache(maxsize=1)
def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Get configured text splitter for document chunking.
    Uses caching to avoid repeated instantiation.
    
    Returns:
        RecursiveCharacterTextSplitter instance
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False
    )


# =============================================================================
# Document Processing
# =============================================================================

def process_pdf(file_path: str, filename: str) -> List[Document]:
    """Process PDF file and return documents."""
    logger.info(f"Processing PDF: {filename}")
    loader = PyMuPDFLoader(file_path)
    return loader.load()


def process_docx(file_path: str, filename: str) -> List[Document]:
    """Process DOCX file and return documents."""
    logger.info(f"Processing DOCX: {filename}")
    loader = Docx2txtLoader(file_path)
    return loader.load()


def process_txt(file_path: str, filename: str) -> List[Document]:
    """Process TXT file and return documents."""
    logger.info(f"Processing TXT: {filename}")
    # Try multiple encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            loader = TextLoader(file_path, encoding=encoding)
            return loader.load()
        except UnicodeDecodeError:
            continue
    
    raise ValueError(f"Could not decode {filename} with any supported encoding")


def process_csv(file_path: str, filename: str) -> List[Document]:
    """Process CSV file and return documents."""
    logger.info(f"Processing CSV: {filename}")
    loader = CSVLoader(file_path)
    return loader.load()


def analyze_image_with_vision(file_path: str, filename: str) -> str:
    """
    Analyze an image using GPT Vision when OCR cannot extract text.
    
    Args:
        file_path: Path to the image file
        filename: Original filename
        
    Returns:
        Text description of the image content from GPT Vision
    """
    logger.info(f"Analyzing image with GPT Vision: {filename}")
    
    try:
        # Read and encode the image
        with open(file_path, "rb") as image_file:
            image_data = image_file.read()
        
        # Resize image if too large (max 2048px on longest side for efficiency)
        image = Image.open(file_path)
        max_size = 2048
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save resized image to buffer
            buffer = io.BytesIO()
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                if image.mode == 'RGBA':
                    background.paste(image, mask=image.split()[-1])
                else:
                    background.paste(image)
                image = background
            image.save(buffer, format="JPEG", quality=85)
            image_data = buffer.getvalue()
        
        image_base64 = base64.b64encode(image_data).decode()
        
        client = get_openai_client()
        
        prompt = """Analyze this image and provide a detailed description of its content. 
Include:
1. What type of image this is (photo, diagram, chart, screenshot, etc.)
2. Main subjects or objects in the image
3. Any text visible in the image (even if OCR couldn't detect it)
4. Key information, data, or concepts depicted
5. Any important details that would help someone understand the image without seeing it

Provide a comprehensive description that can be used for document retrieval and question answering."""

        response = client.chat.completions.create(
            model=settings.openai_vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=settings.openai_max_tokens,
            temperature=0.1
        )
        
        description = response.choices[0].message.content
        logger.info(f"GPT Vision successfully analyzed image: {filename}")
        
        return f"[Image Analysis: {filename}]\n\n{description}"
        
    except Exception as e:
        logger.error(f"Error analyzing image with GPT Vision: {str(e)}")
        # Return a fallback message if Vision API fails
        return f"[Image file: {filename}] - Unable to analyze image content. The image may contain graphics, diagrams, or visual content that could not be processed."


def process_image(file_path: str, filename: str) -> List[Document]:
    """
    Process image file using OCR first, then fall back to GPT Vision if no text found.
    
    Args:
        file_path: Path to the image file
        filename: Original filename
        
    Returns:
        List containing a single Document with extracted content
        
    Raises:
        Exception: If processing fails
    """
    logger.info(f"Processing image with OCR: {filename}")
    
    try:
        image = Image.open(file_path)
        
        # Convert to RGB if necessary for better OCR
        if image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            if image.mode == 'RGBA':
                background.paste(image, mask=image.split()[-1])
            else:
                background.paste(image)
            image = background
        
        # Perform OCR
        text = pytesseract.image_to_string(image)
        
        if not text.strip():
            # OCR found no text - use GPT Vision to analyze the image
            logger.info(f"OCR found no text in image: {filename}. Falling back to GPT Vision.")
            text = analyze_image_with_vision(file_path, filename)
            vision_used = True
        else:
            logger.info(f"OCR extracted {len(text)} characters from image: {filename}")
            vision_used = False
        
        doc = Document(
            page_content=text,
            metadata={
                "source": filename,
                "type": "image",
                "ocr_processed": True,
                "vision_analyzed": vision_used
            }
        )
        return [doc]
        
    except Exception as ocr_error:
        error_str = str(ocr_error).lower()
        if "tesseract" in error_str or "not found" in error_str:
            raise Exception(
                f"Tesseract OCR is not installed or not in PATH. "
                f"Please install Tesseract to process image files. "
                f"Original error: {str(ocr_error)}"
            )
        else:
            raise Exception(f"Error processing image with OCR: {str(ocr_error)}")


def process_database(file_path: str, filename: str) -> List[Document]:
    """
    Process SQLite database file and return documents.
    
    Args:
        file_path: Path to the database file
        filename: Original filename
        
    Returns:
        List containing a single Document with database content
    """
    logger.info(f"Processing SQLite database: {filename}")
    
    try:
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            raise ValueError("Database contains no tables")
        
        all_data = []
        total_rows = 0
        
        for table_name in tables:
            table_name = table_name[0]
            
            try:
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM [{table_name}]")
                row_count = cursor.fetchone()[0]
                total_rows += row_count
                
                # Read table data (limit to first 1000 rows for large tables)
                df = pd.read_sql_query(
                    f"SELECT * FROM [{table_name}] LIMIT 1000",
                    conn
                )
                
                csv_content = f"=== Table: {table_name} ({row_count} rows) ===\n{df.to_csv(index=False)}"
                all_data.append(csv_content)
                
            except Exception as table_error:
                logger.warning(f"Error reading table {table_name}: {str(table_error)}")
                all_data.append(f"=== Table: {table_name} (error reading) ===")
        
        conn.close()
        
        combined_content = "\n\n".join(all_data)
        
        doc = Document(
            page_content=combined_content,
            metadata={
                "source": filename,
                "type": "database",
                "tables_count": len(tables),
                "total_rows": total_rows
            }
        )
        
        logger.info(f"Processed database with {len(tables)} tables and {total_rows} total rows")
        return [doc]
        
    except Exception as e:
        raise Exception(f"Error processing database: {str(e)}")


def process_document(file_path: str, filename: str) -> List[Document]:
    """
    Process a document based on its file extension and return chunks.
    
    Args:
        file_path: Path to the uploaded file
        filename: Original filename
        
    Returns:
        List of Document objects with text chunks
        
    Raises:
        Exception: If processing fails
    """
    start_time = time.time()
    file_extension = Path(filename).suffix.lower()
    
    logger.info(f"Starting document processing: {filename} (type: {file_extension})")
    
    # Map extensions to processing functions
    processors = {
        '.pdf': process_pdf,
        '.docx': process_docx,
        '.txt': process_txt,
        '.csv': process_csv,
        '.png': process_image,
        '.jpg': process_image,
        '.jpeg': process_image,
        '.db': process_database
    }
    
    processor = processors.get(file_extension)
    
    if processor is None:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    try:
        # Process document
        documents = processor(file_path, filename)
        
        # Add common metadata
        for doc in documents:
            doc.metadata["filename"] = filename
            doc.metadata["file_extension"] = file_extension
        
        # Split into chunks
        text_splitter = get_text_splitter()
        chunks = text_splitter.split_documents(documents)
        
        # Add chunk indices
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Document processing complete: {filename} -> {len(chunks)} chunks in {elapsed_time:.2f}s")
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        raise Exception(f"Error processing {filename}: {str(e)}")


# =============================================================================
# Embedding and Storage
# =============================================================================

def create_and_store_embeddings(docs: List[Document], file_id: str) -> int:
    """
    Create embeddings for documents and store them in ChromaDB.
    
    Args:
        docs: List of Document objects
        file_id: Unique identifier for the file
        
    Returns:
        Number of documents stored
        
    Raises:
        Exception: If embedding creation or storage fails
    """
    start_time = time.time()
    logger.info(f"Creating embeddings for {len(docs)} documents (file_id: {file_id})")
    
    try:
        # Add file_id to metadata
        for doc in docs:
            doc.metadata["file_id"] = file_id
        
        # Get vectorstore and add documents
        vectorstore = get_vectorstore()
        vectorstore.add_documents(docs)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Embeddings created and stored: {len(docs)} documents in {elapsed_time:.2f}s")
        
        return len(docs)
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        raise Exception(f"Error creating embeddings: {str(e)}")


def delete_document_embeddings(file_id: str) -> bool:
    """
    Delete all embeddings associated with a file_id.
    
    Args:
        file_id: Unique identifier for the file
        
    Returns:
        True if deletion was successful
        
    Raises:
        Exception: If deletion fails
    """
    logger.info(f"Deleting embeddings for file_id: {file_id}")
    
    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection
        
        # Get IDs of documents with matching file_id
        results = collection.get(
            where={"file_id": file_id},
            include=[]
        )
        
        if results['ids']:
            collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} embeddings for file_id: {file_id}")
        else:
            logger.warning(f"No embeddings found for file_id: {file_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error deleting embeddings: {str(e)}")
        raise Exception(f"Error deleting embeddings: {str(e)}")


# =============================================================================
# Query Processing
# =============================================================================


def handle_text_query(
    question: str,
    context: str,
    chat_history: List[Dict[str, str]] = None,
    temperature: float = 0.1
) -> str:
    """
    Handle text-only query using GPT-4o-mini with conversation memory.
    
    Args:
        question: User's question
        context: Retrieved document context
        chat_history: Previous conversation messages for context
        temperature: Response generation temperature
        
    Returns:
        Generated answer
    """
    logger.info("Processing text query with GPT-4o-mini")
    
    client = get_openai_client()
    
    system_prompt = """You are an expert assistant that provides accurate, clear, and concise answers based strictly on the provided context.

Guidelines:
1. Base your answers ONLY on the provided context
2. If the context doesn't contain enough information, clearly state this
3. Cite specific parts of the context when relevant
4. Structure your response clearly with bullet points if appropriate
5. Do not make up information or use external knowledge
6. If asked for opinions or analysis, base them solely on the context
7. Consider the conversation history when answering follow-up questions
8. If a question references something from the conversation (like "it", "that", "the previous point"), use the chat history to understand what is being referenced"""

    # Build messages list with conversation history
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history for context (limit to last 10 messages to avoid token limits)
    if chat_history:
        history_to_include = chat_history[-10:] if len(chat_history) > 10 else chat_history
        for msg in history_to_include:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
    
    # Add current question with context
    user_prompt = f"""Context from the document:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above."""

    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model=settings.openai_mini_model,
        messages=messages,
        max_tokens=settings.openai_max_tokens,
        temperature=temperature
    )
    
    return response.choices[0].message.content


def generate_suggested_questions(
    question: str,
    answer: str,
    context: str,
    chat_history: List[Dict[str, str]] = None
) -> List[str]:
    """
    Generate follow-up questions based on the conversation and context.
    
    Args:
        question: The user's question
        answer: The generated answer
        context: Document context
        chat_history: Previous conversation messages
        
    Returns:
        List of 3 suggested follow-up questions
    """
    logger.info("Generating suggested follow-up questions")
    
    client = get_openai_client()
    
    # Build conversation summary for context
    conversation_summary = ""
    if chat_history:
        recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
        conversation_summary = "\n".join([
            f"{msg['role'].title()}: {msg['content'][:200]}..." 
            if len(msg['content']) > 200 else f"{msg['role'].title()}: {msg['content']}"
            for msg in recent_history
        ])
    
    prompt = f"""Based on this document Q&A conversation, suggest 3 natural follow-up questions the user might want to ask next.

Document Context (excerpt):
{context[:1500]}...

{"Previous Conversation:" + chr(10) + conversation_summary if conversation_summary else ""}

Latest Question: {question}
Latest Answer: {answer[:500]}...

Generate exactly 3 follow-up questions that:
1. Are specific and relevant to the document content
2. Build naturally on the conversation
3. Help the user explore related topics or go deeper into interesting points
4. Are concise (under 100 characters each)

Return ONLY the 3 questions, one per line, without numbering or bullet points."""

    try:
        response = client.chat.completions.create(
            model=settings.openai_mini_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates relevant follow-up questions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        # Parse response into list of questions
        questions_text = response.choices[0].message.content.strip()
        questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
        
        # Return first 3 valid questions
        return questions[:3]
        
    except Exception as e:
        logger.warning(f"Failed to generate suggested questions: {e}")
        return []


def perform_rag_query(query_request: QueryRequest) -> QueryResponse:
    """
    Perform RAG query with hybrid search and multi-document support.
    
    Features:
    - Multi-document querying: Search across multiple documents simultaneously
    - Hybrid search: Combines vector similarity with BM25 keyword matching
    - Reciprocal Rank Fusion: Intelligently merges results from both search methods
    - Conversation memory: Uses chat history for context-aware responses
    
    Args:
        query_request: Query request object with question, file_id(s), and options
        
    Returns:
        QueryResponse with answer, context, sources, and suggested follow-up questions
        
    Raises:
        Exception: If query processing fails
    """
    start_time = time.time()
    
    # Get file IDs to query (supports both single and multi-document modes)
    file_ids = query_request.get_file_ids()
    
    if not file_ids:
        return QueryResponse(
            answer="No document specified. Please provide a file_id or file_ids to query.",
            context="",
            sources=[],
            suggested_questions=[],
            search_method="none",
            documents_searched=0,
            model_used="none",
            processing_time_ms=int((time.time() - start_time) * 1000)
        )
    
    logger.info(f"Performing RAG query across {len(file_ids)} document(s)")
    
    try:
        max_sources = query_request.max_sources or 5
        temperature = query_request.temperature or 0.1
        use_hybrid = query_request.use_hybrid_search if query_request.use_hybrid_search is not None else True
        
        # Convert chat_history from Pydantic models to dicts if present
        chat_history = None
        if query_request.chat_history:
            chat_history = [
                {"role": msg.role, "content": msg.content}
                for msg in query_request.chat_history
            ]
        
        # Perform hybrid search
        search_results, search_method = hybrid_search(
            query_request.question,
            file_ids,
            k=max_sources,
            use_hybrid=use_hybrid
        )
        
        if not search_results:
            logger.warning(f"No documents found for file_ids: {file_ids}")
            return QueryResponse(
                answer="No relevant documents found for this query. Please ensure you have uploaded and processed documents with the specified file IDs.",
                context="",
                sources=[],
                suggested_questions=[],
                search_method=search_method,
                documents_searched=len(file_ids),
                model_used="none",
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
        
        # Extract documents and build context
        docs = [doc for doc, _, _, _, _ in search_results]
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        # Build sources list with detailed scoring info
        sources = []
        for doc, combined_score, vector_score, bm25_score, search_type in search_results:
            source = Source(
                filename=doc.metadata.get("filename", "Unknown"),
                file_id=doc.metadata.get("file_id"),
                page_number=doc.metadata.get("page", None),
                content=doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                relevance_score=round(combined_score, 3),
                vector_score=round(vector_score, 3) if vector_score else None,
                bm25_score=round(bm25_score, 3) if bm25_score else None,
                chunk_index=doc.metadata.get("chunk_index", None),
                search_type=search_type
            )
            sources.append(source)
        
        # Generate answer using text query with conversation history
        answer = handle_text_query(
            query_request.question,
            context,
            chat_history,
            temperature
        )
        model_used = settings.openai_mini_model
        
        # Generate suggested follow-up questions
        suggested_questions = generate_suggested_questions(
            query_request.question,
            answer,
            context,
            chat_history
        )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"RAG query complete: {len(sources)} sources from {len(file_ids)} docs, "
            f"search_method={search_method}, {processing_time_ms}ms"
        )
        
        return QueryResponse(
            answer=answer,
            context=context,
            sources=sources,
            suggested_questions=suggested_questions,
            search_method=search_method,
            documents_searched=len(file_ids),
            model_used=model_used,
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Error performing RAG query: {str(e)}")
        raise Exception(f"Error performing RAG query: {str(e)}")


# =============================================================================
# Statistics and Monitoring
# =============================================================================

def get_vectorstore_stats() -> Dict[str, Any]:
    """
    Get statistics about the vectorstore.
    
    Returns:
        Dictionary with vectorstore statistics
    """
    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection
        count = collection.count()
        
        # Try to get unique file count
        try:
            all_metadata = collection.get(include=["metadatas"])
            unique_files = set()
            for metadata in all_metadata.get("metadatas", []):
                if metadata and "file_id" in metadata:
                    unique_files.add(metadata["file_id"])
            
            return {
                "total_documents": count,
                "unique_files": len(unique_files),
                "status": "healthy"
            }
        except:
            return {
                "total_documents": count,
                "status": "healthy"
            }
        
    except Exception as e:
        logger.error(f"Error getting vectorstore stats: {str(e)}")
        return {
            "total_documents": 0,
            "status": f"error: {str(e)}"
        }

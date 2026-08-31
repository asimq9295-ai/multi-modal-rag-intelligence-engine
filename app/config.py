"""
Configuration module for the Multi-Modal RAG application.
Loads all configuration from environment variables with fallback to .env.example defaults.
All configurable values are centralized here - no hardcoded values elsewhere.
"""

import os
import logging
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from dotenv import load_dotenv

# Load environment variables from .env file (or .env.example as fallback)
load_dotenv()
if not os.path.exists('.env'):
    load_dotenv('.env.example')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings loaded exclusively from environment variables.
    
    Priority order:
    1. Environment variables (highest priority)
    2. .env file
    3. .env.example file
    4. Default values defined here (lowest priority)
    
    All settings can be overridden via environment variables.
    """
    
    # =========================================================================
    # OpenAI Configuration
    # =========================================================================
    openai_api_key: str = Field(
        default="sk-your-actual-api-key-here",
        description="OpenAI API key for embeddings and chat completions"
    )
    
    openai_model: str = Field(
        default="gpt-4o",
        description="Primary OpenAI chat model for text queries"
    )
    
    openai_mini_model: str = Field(
        default="gpt-4o-mini",
        description="Lightweight OpenAI model for simple tasks"
    )
    
    openai_vision_model: str = Field(
        default="gpt-4o",
        description="OpenAI model for vision/image analysis"
    )
    
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model"
    )
    
    openai_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default temperature for LLM responses"
    )
    
    openai_max_tokens: int = Field(
        default=10000,
        ge=1,
        le=100000,
        description="Maximum tokens in LLM response"
    )
    
    # =========================================================================
    # Storage Paths
    # =========================================================================
    chroma_db_path: str = Field(
        default="./data/chroma_db",
        description="Path to ChromaDB persistent storage"
    )
    
    uploads_path: str = Field(
        default="./data/uploads",
        description="Path for uploaded file storage"
    )
    
    # =========================================================================
    # API Server Configuration
    # =========================================================================
    api_host: str = Field(
        default="0.0.0.0",
        description="Host to bind the API server"
    )
    
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port for the API server"
    )
    
    cors_origins: str = Field(
        default="*",
        description="Allowed CORS origins (comma-separated or *)"
    )
    
    # =========================================================================
    # Document Processing
    # =========================================================================
    max_file_size_mb: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum file size in MB"
    )
    
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Text chunk size for document splitting"
    )
    
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=500,
        description="Overlap between text chunks"
    )
    
    # =========================================================================
    # RAG Configuration
    # =========================================================================
    default_max_sources: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Default number of source documents to retrieve"
    )
    
    default_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default temperature for LLM responses"
    )
    
    # =========================================================================
    # Frontend Configuration
    # =========================================================================
    api_base_url: str = Field(
        default="http://localhost:8000/api/v1",
        description="Backend API URL for frontend"
    )
    
    request_timeout: int = Field(
        default=60,
        ge=1,
        description="HTTP request timeout in seconds"
    )
    
    health_check_timeout: int = Field(
        default=5,
        ge=1,
        description="Health check timeout in seconds"
    )
    
    # =========================================================================
    # Application Settings
    # =========================================================================
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode for verbose logging"
    )
    
    environment: str = Field(
        default="development",
        description="Application environment (development or production)"
    )
    
    @field_validator('openai_api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate OpenAI API key is not empty or placeholder."""
        if not v or v == "sk-your-actual-api-key-here" or v.startswith("your_"):
            raise ValueError(
                "OPENAI_API_KEY is not set or is still a placeholder. "
                "Please set a valid OpenAI API key in your .env file."
            )
        if not v.startswith("sk-"):
            logger.warning(
                "OpenAI API key doesn't start with 'sk-'. "
                "This may not be a valid API key format."
            )
        return v
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from string to list."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"  # Ignore extra environment variables
    }


# Initialize global settings instance
try:
    settings = Settings()
    logger.info("Configuration loaded successfully")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"API Host: {settings.api_host}:{settings.api_port}")
    logger.info(f"OpenAI Model: {settings.openai_model}")
    logger.info(f"OpenAI Vision Model: {settings.openai_vision_model}")
    logger.info(f"OpenAI Embedding Model: {settings.openai_embedding_model}")
    
    if settings.debug_mode:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
        
except Exception as e:
    logger.error(f"Failed to load configuration: {str(e)}")
    raise


# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        settings.chroma_db_path,
        settings.uploads_path
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")


# Create directories on module load
ensure_directories()

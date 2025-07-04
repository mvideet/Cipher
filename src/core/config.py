"""
Configuration settings for Cipher Desktop
"""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8001
    DEBUG: bool = True
    SQLALCHEMY_ECHO: bool = False  # Set to False to reduce database logs
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"
    
    # File System
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    TEMP_DIR: Path = BASE_DIR / "temp"
    MODELS_DIR: Path = BASE_DIR / "models"
    RUNS_DIR: Path = BASE_DIR / "runs"
    
    # Database
    DATABASE_URL: str = "sqlite:///./cipher.db"
    
    # Training Configuration
    MAX_TRAINING_TIME_MINUTES: int = 15
    MAX_OPTUNA_TRIALS: int = 20
    TRAINING_PROCESSES: int = 3
    
    # PyTorch removed from configuration
    
    # File Upload Limits
    MAX_FILE_SIZE_MB: int = 100
    MAX_ROWS_PREVIEW: int = 5000
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings() 
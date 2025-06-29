"""
Database initialization and connection management
"""

from sqlmodel import SQLModel, create_engine, Session
import structlog

from .core.config import settings
from .models.schema import Run, Trial, Event

logger = structlog.get_logger()

# Create engine
engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    connect_args={"check_same_thread": False}  # For SQLite
)


async def init_db():
    """Initialize database tables"""
    logger.info("Initializing database")
    SQLModel.metadata.create_all(engine)
    logger.info("Database initialized successfully")


def get_session():
    """Get database session"""
    with Session(engine) as session:
        yield session 
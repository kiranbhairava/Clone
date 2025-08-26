from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS") 
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME")

if not all([DB_USER, DB_PASS, DB_HOST, DB_NAME]):
    raise RuntimeError("Database credentials not found in environment variables.")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}?charset=utf8mb4"

# Database engine with proper connection settings
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    pool_recycle=3600,              # Recycle connections every hour
    pool_pre_ping=True,             # Test connections before use
    pool_timeout=60,                # Connection timeout
    max_overflow=20,                # Allow overflow connections
    connect_args={
        "connect_timeout": 120,     # MySQL connection timeout
        "read_timeout": 120,        # MySQL read timeout  
        "write_timeout": 120,       # MySQL write timeout
    },
    echo=False                      # Set to True for SQL debugging
)

# Session configuration
SessionLocal = sessionmaker(
    autocommit=False,               # Keep manual transaction control
    autoflush=False,                # Don't auto-flush to avoid premature commits
    bind=engine
)

Base = declarative_base()

# Database dependency for FastAPI
def get_db():
    """
    Database session dependency with proper error handling
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database error: {e}")
        db.rollback()
        raise  # Re-raise the exception after rollback
    finally:
        try:
            db.close()
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

# Database connection test function
def test_db_connection():
    """Test database connection"""
    try:
        # Test connection using engine
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        # Try alternative method with session
        try:
            db = SessionLocal()
            db.execute(text("SELECT 1"))
            db.close()
            logger.info("Database connection successful (via session)")
            return True
        except Exception as e2:
            logger.error(f"Database session test also failed: {e2}")
            return False

# Database health check
def get_db_health():
    """Get database health status"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            return {
                "status": "healthy",
                "connection": "active",
                "pool_size": engine.pool.size(),
                "checked_out_connections": engine.pool.checkedout()
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "connection": "failed"
        }
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Enum
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import enum
import chromadb
import os

print("--- Initializing Database Layer ---")

# --- Part 1: SQL Database (SQLAlchemy) for Metadata ---
# ----------------------------------------------------
DB_FILE = "metaminds.db"
DATABASE_URL = f"sqlite:///{DB_FILE}"

# Define the base class for our tables (this is standard SQLAlchemy)
Base = declarative_base()

# Define an Enum for our file processing status
class FileStatus(enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"

# Define the 'files' table as a Python class
class File(Base):
    __tablename__ = "files"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False, unique=True)
    file_type = Column(String) # e.g., "pdf", "docx", "png"
    category = Column(String, index=True) # e.g., "Invoice", "Resume"
    status = Column(Enum(FileStatus), default=FileStatus.PENDING, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_processed = Column(DateTime)

# Create an engine to connect to the DB
engine = create_engine(DATABASE_URL)

# Create a session-maker (we'll use this later to add/query data)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_sql_db():
    """Creates the .db file and the 'files' table if they don't exist."""
    if not os.path.exists(DB_FILE):
        print(f"Creating SQL database file: {DB_FILE}")
        Base.metadata.create_all(bind=engine)
    else:
        print(f"SQL database file found: {DB_FILE}")

# --- Part 2: Vector Database (ChromaDB) for Smart Search ---
# ----------------------------------------------------------
CHROMA_PATH = "chroma_db_store"
COLLECTION_NAME = "documents"

# We use a *Persistent* client to save data to disk in the CHROMA_PATH folder
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

def get_or_create_vector_collection():
    """Creates or loads the vector 'collection' (like a table)."""
    print(f"ChromaDB store location: {CHROMA_PATH}")
    try:
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity for search
        )
        print(f"Vector collection '{COLLECTION_NAME}' loaded/created.")
        return collection
    except Exception as e:
        print(f"Error with ChromaDB: {e}")
        return None

# This is the main function that runs when we execute this file directly
if __name__ == "__main__":
    print("Running database setup...")
    create_sql_db()
    get_or_create_vector_collection()
    print("--- Database Setup Complete ---")
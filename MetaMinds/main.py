from fastapi import (
    FastAPI,
    UploadFile,
    File as FastAPIFile,
    HTTPException,
    BackgroundTasks,
)
from pydantic import BaseModel
import uvicorn
import os
import shutil

# --- Import from our other files ---
# We are importing the functions and objects we built
from database import (
    SessionLocal,
    engine,
    Base,
    File,
    FileStatus,
    get_or_create_vector_collection,
)
from processing import process_document

# --- Create SQL Tables ---
# This creates the 'files' table in metaminds.db if it doesn't exist
Base.metadata.create_all(bind=engine)

# --- Initialize API and Databases ---
app = FastAPI(title="MetaMinds AI Content Manager")

# Get our vector database collection
vector_collection = get_or_create_vector_collection()

# Define where uploaded files will be stored
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --- Pydantic Models (for API data validation) ---
class SearchQuery(BaseModel):
    query: str
    top_k: int = 5  # Default to returning top 5 results


class SearchResult(BaseModel):
    filename: str
    category: str | None
    score: float


# --- Background Task for Processing ---
def process_file_in_background(file_id: int, filepath: str):
    """
    This function runs in the background AFTER the upload is "done".
    """
    print(f"BACKGROUND TASK: Processing file_id {file_id}")

    # 1. Get a new database session
    db = SessionLocal()

    try:
        # 2. Update status to PROCESSING
        db_file = db.query(File).filter(File.id == file_id).first()
        if not db_file:
            print(f"BACKGROUND TASK: File {file_id} not found.")
            return

        db_file.status = FileStatus.PROCESSING
        db.commit()

        # 3. Run the HEAVY AI processing
        # This uses the GPU
        text_content, vector_embedding = process_document(filepath)

        if text_content and vector_embedding:
            # 4. Save to vector database (ChromaDB)
            vector_collection.add(
                embeddings=[vector_embedding],
                documents=[text_content],
                metadatas=[{"filename": db_file.filename, "sql_id": file_id}],
                ids=[str(file_id)],  # Chroma needs a string ID
            )

            # 5. Update SQL status to PROCESSED
            db_file.status = FileStatus.PROCESSED
            # TODO: Add real AI categorization later
            db_file.category = "Uncategorized"
            db.commit()
            print(f"BACKGROUND TASK: File {file_id} processed and indexed.")
        else:
            raise Exception("Processing failed to extract text or vector.")

    except Exception as e:
        # 6. Handle failure
        print(f"BACKGROUND TASK: Failed to process {file_id}. Error: {e}")
        db_file = db.query(File).filter(File.id == file_id).first()
        if db_file:
            db_file.status = FileStatus.FAILED
            db.commit()
    finally:
        db.close()


# --- API Endpoints ---


@app.get("/")
def read_root():
    """Simple status check endpoint."""
    return {
        "status": "MetaMinds AI Server is running",
        "gpu_available": torch.cuda.is_available(),
    }


@app.post("/upload/")
async def upload_file(
    background_tasks: BackgroundTasks, file: UploadFile = FastAPIFile(...)
):
    """
    Uploads a file, saves it, and schedules it for AI processing.
    """
    # 1. Save the uploaded file to our local disk
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Get a database session
    db = SessionLocal()
    try:
        # 3. Create the 'File' record in our SQL database
        db_file = File(
            filename=file.filename,
            filepath=filepath,
            file_type=file.content_type,
            status=FileStatus.PENDING,
        )
        db.add(db_file)
        db.commit()
        db.refresh(db_file)  # Get the new file_id

        # 4. Schedule the HEAVY AI processing to run in the background
        background_tasks.add_task(
            process_file_in_background, file_id=db_file.id, filepath=filepath
        )

        # 5. Return an INSTANT response to the user
        return {
            "message": "File uploaded. AI processing has started.",
            "file_id": db_file.id,
            "filename": file.filename,
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    finally:
        db.close()


@app.post("/search/", response_model=list[SearchResult])
async def search_documents(query: SearchQuery):
    """
    Performs AI-powered "smart search" on the vector database.
    """
    try:
        # 1. Import the AI model (from processing.py)
        from processing import model, device

        # 2. Convert the user's text query into a vector (on the GPU)
        query_embedding = model.encode(query.query, device=device).tolist()

        # 3. Search ChromaDB for the most similar vectors
        results = vector_collection.query(
            query_embeddings=[query_embedding], n_results=query.top_k
        )

        # 4. Format and return the results
        search_results = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                search_results.append(
                    SearchResult(
                        filename=results["metadatas"][0][i]["filename"],
                        category="Uncategorized",  # TODO: Get from SQL DB
                        score=results["distances"][0][
                            i
                        ],  # This is the similarity score
                    )
                )

        return search_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


# --- This is the code that runs the server ---
if __name__ == "__main__":
    print("Starting FastAPI server...")
    # We need to import torch here to make the GPU work in the background
    import torch

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

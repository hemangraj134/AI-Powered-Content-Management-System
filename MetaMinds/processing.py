import fitz  # This is the PyMuPDF library
import docx
from PIL import Image
import pytesseract
import torch
from sentence_transformers import SentenceTransformer
import os

print("--- Initializing AI Processing Engine ---")

# --- 1. Load the AI Model onto the GPU ---
# This checks if your GPU is available (which we know it is)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading AI model onto device: {device}")

# This is the model that creates the "smart search" vectors
# The first time this line runs, it will download the model (approx 90MB)
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
print("AI Model loaded successfully.")


# --- 2. Define File Extraction Functions ---


def extract_text_from_pdf(filepath):
    """Extracts text from a PDF file."""
    try:
        with fitz.open(filepath) as doc:
            text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        print(f"Error reading PDF {filepath}: {e}")
        return None


def extract_text_from_docx(filepath):
    """Extracts text from a .docx file."""
    try:
        doc = docx.Document(filepath)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Error reading DOCX {filepath}: {e}")
        return None


def extract_text_from_image(filepath):
    """Extracts text from an image file using Tesseract OCR."""
    try:
        # We use the Tesseract program we installed
        text = pytesseract.image_to_string(Image.open(filepath))
        return text
    except Exception as e:
        print(f"Error reading IMAGE {filepath}: {e}")
        return None


def extract_text_from_txt(filepath):
    """Extracts text from a plain .txt file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        return text
    except Exception as e:
        print(f"Error reading TXT {filepath}: {e}")
        return None


# --- 3. Define AI Embedding Function ---


def get_embedding(text):
    """Generates a vector embedding for a string of text using the AI model."""
    if not text:
        return None
    try:
        # This runs the calculation on your 4050 GPU
        embedding = model.encode(text, convert_to_tensor=False)
        # We convert to a standard Python list to store it in the DB
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


# --- 4. Main Orchestrator Function ---


def process_document(filepath):
    """
    Processes a single document:
    1. Gets the file extension.
    2. Calls the correct text extractor.
    3. Generates an embedding for the extracted text.
    """
    print(f"\n--- Processing Document: {filepath} ---")

    # Get file extension
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()

    # Call the correct extractor
    text = ""
    if ext == ".pdf":
        text = extract_text_from_pdf(filepath)
    elif ext == ".docx":
        text = extract_text_from_docx(filepath)
    elif ext in [".png", ".jpg", ".jpeg", ".tiff"]:
        text = extract_text_from_image(filepath)
    elif ext == ".txt":
        text = extract_text_from_txt(filepath)
    else:
        print(f"Unsupported file type: {ext}")
        return None, None

    if not text:
        print(f"No text extracted from {filepath}.")
        return None, None

    print(f"Extracted Text (first 200 chars): {text[:200]}...")

    # Generate the vector embedding
    print("Generating vector embedding...")
    embedding = get_embedding(text)

    if embedding:
        print(f"Embedding generated (Vector size: {len(embedding)})")
        return text, embedding
    else:
        print("Failed to generate embedding.")
        return text, None


# --- 5. Test Block ---
# This code only runs when you execute `python processing.py` directly
if __name__ == "__main__":
    print("\nRunning in test mode...")

    # Create a dummy test file
    test_file = "test.txt"
    test_content = "This is a test document about AI-powered content management."
    with open(test_file, "w") as f:
        f.write(test_content)

    # Run the full processing pipeline on our test file
    text, embedding = process_document(test_file)

    if text and embedding:
        print("\n--- TEST SUCCEEDED ---")
    else:
        print("\n--- TEST FAILED ---")

    # Clean up the dummy file
    os.remove(test_file)

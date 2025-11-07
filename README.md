# 🚀 MetaMinds: AI-Powered Content Management System

An intelligent document and media organizer that automatically extracts text, generates vector embeddings, and provides a powerful, GPU-accelerated natural language search API.

[![Python Version][python-shield]][python-url]
[![FastAPI][fastapi-shield]][fastapi-url]
[![PyTorch][pytorch-shield]][pytorch-url]
[![ChromaDB][chroma-shield]][chroma-url]

This repository contains the complete, working v1.0 snapshot of the project, including the full source code, test data, and pre-built databases.

## ✨ Core Features

* **Asynchronous API:** Built with FastAPI, the `/upload` endpoint responds instantly and schedules all heavy AI processing to run in the background.
* **GPU-Accelerated AI:** Leverages **NVIDIA CUDA** via `PyTorch` (tested on an RTX 4050) to perform all model inference on the GPU for 10-100x faster processing.
* **Multi-Format Text Extraction:**
    * PDFs (`PyMuPDF`)
    * Word Docs (`python-docx`)
    * Images/Scans (`Tesseract OCR`)
    * Plain Text (`.txt`)
* **Semantic Search:** Implements a high-precision "smart search" that understands natural language *meaning*, not just keywords, using `sentence-transformers` and `ChromaDB`.

---

## 🏛️ Technical Architecture (The "Lean" Model)

This project uses a **lightweight, high-performance, single-service architecture** designed to run locally without heavy Docker containers, while still using a decoupled, event-driven workflow.

1.  **API (FastAPI):** A user sends a file to the `POST /upload` endpoint.
2.  **File Storage (Local):** The file is instantly saved to the `./MetaMinds/uploaded_files` directory.
3.  **SQL Database (SQLite):** A metadata row is created in the `metaminds.db` (via `SQLAlchemy`) with `status: PENDING`.
4.  **Background Task:** The API returns `HTTP 202` to the user and triggers a `BackgroundTasks` job.
5.  **Processing (Python):** The background task runs `processing.py`. It reads the file, extracts text, and...
6.  **AI Engine (Torch + GPU):** The extracted text is sent to the **NVIDIA GPU** to be converted into a 384-dimension vector by the `all-MiniLM-L6-v2` model.
7.  **Vector DB (ChromaDB):** This new vector is saved in the `chroma_db_store/` collection, mapped to the file's ID.
8.  **Search (FastAPI):** A user sends a query to `POST /search`. The API uses the **GPU** to convert the *query* into a vector, then searches `ChromaDB` for the most similar file vectors.

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend API** | [FastAPI][fastapi-url] | High-performance asynchronous web framework. |
| **Server** | `uvicorn` | The ASGI server that runs the API. |
| **Core AI** | [PyTorch (CUDA)][pytorch-url] | The main AI engine, running on the NVIDIA GPU. |
| **NLP Models** | `transformers` | Hugging Face library for AI models. |
| **Semantic Search** | `sentence-transformers` | Generates vector embeddings for search. |
| **Vector Database** | [ChromaDB][chroma-url] | Stores and searches vectors. |
| **Metadata Database**| `SQLAlchemy` + `SQLite` | Stores all file metadata. |
| **PDF Reading** | `PyMuPDF` | Text extraction from `.pdf` files. |
| **Word Reading** | `python-docx` | Text extraction from `.docx` files. |
| **OCR** | `pytesseract` | Text extraction from images. |

---

## 🚀 Setup & Installation

Follow these steps to set up the environment locally.

### 1. Prerequisites (External)
* **Python 3.12+**
* **Tesseract OCR Engine:** You must install the Tesseract program (v5+) and add it to your system `PATH`.
    * *Windows Installer:* [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)

### 2. Project Setup

```bash
# 1. Clone this repository
git clone [YOUR_REPO_URL]
cd [YOUR_REPO_NAME]

# 2. Create a Python virtual environment
python -m venv .venv

# 3. Activate the environment
# On Windows (PowerShell)
.\.venv\Scripts\activate

# 4. Install all Python dependencies from the included file
# Note: This is a large install and includes GPU libraries
pip install -r MetaMinds/requirements.txt



            Start editing…
```

### 3. (Windows Only) Fix Long Paths

This project requires long path support on Windows.
Open PowerShell **as Administrator**.
Run this command:
PowerShellNew-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

**Reboot your computer.**

### 4. Run the ServerOnce setup is complete, you can launch the server. All databases and test files are included.Bash# Navigate into the main code folder
cd MetaMinds

### This will use the existing databases and start the API
python main.py
The server will be live at http://127.0.0.1:8000.
## 🧪 How to Test (Using the API)This project comes with a built-in interactive API dashboard.

**Run the server:** 
cd MetaMinds and then python main.py

**Open the dashboard:** 
Go to http://127.0.0.1:8000/docs in your browser.

### Searching for a FileThe repository already includes two pre-processed files.

You can search for them immediately:

Expand the **POST /search/** endpoint.

Click "Try it out".

In the request body, use a *natural language* query about the *content* of one of the files:
```
JSON{
  "query": "a document about artificial intelligence"
}
```

Click "Execute".

The API will return a JSON list with MetaMinds/test_file_1.txt, ranked by relevance.
```
[pytorch-shield]: https://www.google.com/search?q=https://img.shields.io/badge/PyTorch-2.5 (CUDA)-red.svg

[pytorch-url]: https://pytorch.org/

[chroma-shield]: https://www.google.com/search?q=https://img.shields.io/badge/ChromaDB-1.3.4-blueviolet.svg

[chroma-url]: https://www.trychroma.com/
```
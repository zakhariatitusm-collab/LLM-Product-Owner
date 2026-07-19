from pathlib import Path

# Folder
UPLOAD_DIR = Path("uploads")
VECTORSTORE_DIR = "vectorstore"

# Gemini
CHAT_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
# Configuration for RAG System

# Paths
DATA_PDF_DIR = "./data/pdf"
CHROMA_DB_DIR = "./chroma_db"

# Embeddings
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"

# LLM
LLM_MODEL = "gpt-oss:20b"
LLM_TEMPERATURE = 0.7
LLM_BASE_URL = "http://localhost:11434"  # Ollama default

# Retriever
RETRIEVER_SEARCH_TYPE = "similarity"
RETRIEVER_K = 3

# Text Splitting
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", " ", ""]

# System Prompt
SYSTEM_PROMPT = """You are an AI assistant that answers questions based on the provided context. 
If the context doesn't contain the answer, say "I don't have enough information to answer this question."
Keep your answers concise and relevant."""

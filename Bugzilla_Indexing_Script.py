import json
import os
from pathlib import Path
from tqdm import tqdm  # <-- for progress bar

"""
from pymilvus import connections
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType


# Connect (will create local milvus.db file)
connections.connect("default", host="127.0.0.1", port="19530")

# Define schema
fields = [
    FieldSchema(name="bug_id", dtype=DataType.INT64, is_primary=False),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=len(chunks[0]["embedding"]))
]
schema = CollectionSchema(fields, "Bugzilla chunks collection")
"""

# Will Use Chroma for quick demo for Bugzilla RAG, for production will have to switch to Milvus DB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


# ==== CONFIGURATION ====
CHROMA_DB_DIR = "chroma_bugz_vecdb"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"  # Change to any supported model
DEVICE = "cuda"  # Change to "cuda" if you have GPU

BUGZ_DIR = "../ALL_DATA/Bugzilla"

chunk_size_ = 1000  # Size of each text chunk
chunk_overlap_ = 200  # Overlap between chunks

# ===== Embedding Model =====
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": DEVICE}
)

# === Load or Create Chroma Vector Store ===
vectorstore = Chroma(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embeddings
)

# No need to chunk the Summary of each bug
# Embed Summary of bug directly


def index_bug(filepath: Path):
    """Index a single bug JSON file into Chroma."""
    json_file = Path(filepath)
    
    # Load JSON data
    with open(json_file, "r") as f:
        data = json.load(f)

    # Extract Summary
    summary = data.get("summary", "")

    # Extract description
    description = data.get("description") or ""

    # Extract comments
    comments = data.get("comments") or []

    # Chunking & Embedding the summary
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    if summary != None:
        summary_chunks = text_splitter.create_documents([summary], metadatas=[{"bug_id": data.get("id", ""), "source": str(filepath)}],)
        vectorstore.add_documents(summary_chunks)


    # Chunking & Embedding the description
    if description != "" and description != None:
        description_chunks = text_splitter.create_documents([description], metadatas=[{"bug_id": data.get("id", ""), "source": str(filepath)}],)
        vectorstore.add_documents(description_chunks)


    # Chunking & Embedding the the comments
    if len(comments) != 0:
        comments_chunks = text_splitter.create_documents(comments, metadatas=[{"bug_id": data.get("id", ""), "source": str(filepath)} for _ in comments],)
        vectorstore.add_documents(comments_chunks)


def index_all_bugs(directory: Path):
    """Index all JSON bug files in a directory with tqdm progress bar."""
    directory = Path(directory)  
    json_files = list(directory.glob("*.json"))

    for filepath in tqdm(json_files, desc="Indexing bugs"):
        try:
            index_bug(filepath)
        except Exception as e:
            print(f"⚠️ Failed to index {filepath}: {e}")

    # Persist once at the end for speed
    vectorstore.persist()
    print(f"✅ Indexed {len(json_files)} bug files into Chroma DB.")


if __name__ == "__main__":
    index_all_bugs(BUGZ_DIR)



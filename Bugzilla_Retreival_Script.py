from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import json
from pydantic import BaseModel
import time

import os

from typing import List, Dict, Any
from io import BytesIO
import ollama

from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pathlib import Path
import html

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


app = FastAPI()


# === Setting Up the Ollama Server ====
import ollama
import subprocess   # allows running system commands(like starting the Ollama Server)
import time     # provides time related funcs like sleep()
import socket   # used for network communication, in our case to ches if Ollama server is running

def is_ollama_running(host="127.0.0.1", port=11434):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


# ==== CONFIGURATION ====
BUGZ_DIR = "../ALL_DATA/Bugzilla"

CHROMA_DB_DIR = "chroma_bugz_vecdb"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"  # Change to any supported model
DEVICE = "cuda"  # Change to "cuda" if you have GPU

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

# Define the expected request model (optional but recommended for validation)
class RagRequest(BaseModel):
    userQuery: str
    conversationHistory: list
    selectedModel: str
    temperature: float | None = None
    documentSource: str

def chunk_response(response: str, chunk_size=3):
    sentences = response.split('. ')  # Split by sentences
    for i in range(0, len(sentences), chunk_size):
        chunk = '. '.join(sentences[i:i+chunk_size]) + ('.' if i+chunk_size < len(sentences) else '')
        yield json.dumps({"message": {"content": chunk}}) + "\n"


def Query_LLM(user_query, model, bug_ids, chat_history):
    
    context = ""
    for id in bug_ids:
        bug_filepath = f"{BUGZ_DIR}/bug_{id}.json"
        if not Path(bug_filepath).exists():
            raise FileNotFoundError(f"No bug file found for ID {id}")

        with open(bug_filepath, "r", encoding="utf-8") as f:
            contents = json.load(f)
        text_representation = f"""
            Bug ID: {contents['id']}
            Product: {contents['product']}
            Version: {contents['version']}
            Summary: {contents['summary']}
            Status: {contents['status']}
            Description: {contents['description']}
            Comments:
            {chr(10).join([f"- {c}" for c in contents.get('comments', [])])}
        """
        context += text_representation + "\n"
        print(text_representation)

    chat_history.append({
        'role': 'user',
        'content': f'USER QUESTION: {user_query} Context: {context}'
    })

    stream = ollama.chat(
        model=model,                 
        messages=chat_history,
        stream=True
    )

    # Yield each chunk from the LLM
    for chunk in stream:
        content = chunk['message']['content']
        yield json.dumps({"message": {"content": content}, "done": False}) + "\n"
    
    print("-------(2.2)--------")

    return 


@app.post("/chat")
async def handle_rag_request(request: Request):
    try:
        print("-------(0.)--------")
        body = await request.json()
        print("Received body:", body)  # Debug what's actually being sent  
        rag_request = RagRequest(**body)
        
        user_query = rag_request.userQuery
        selected_model = rag_request.selectedModel
        chat_history = rag_request.conversationHistory

        # Retrieve Relevant Bugs

        retrieved_bugs = vectorstore.similarity_search(user_query, k=5)
        for i, bug in enumerate(retrieved_bugs, 1):
            print(f"\nResult {i}")
            print("Text:", bug.page_content[:200], "...")
            print("Metadata:", bug.metadata)

        print("-------(1.)--------")

        # Retrieve relevant bug data
        bug_ids = []
        for bug in retrieved_bugs:
            bug_ids.append(bug.metadata['bug_id'])

        print(bug_ids)


        async def generate_full_response():
            # Stream LLM response
            for chunk in Query_LLM(user_query, selected_model, bug_ids, chat_history):
               yield chunk                

            print("-------(2.)--------")

            sources = "\n\n📚 Relevant Bugz:\n\n"
            for id in bug_ids:
                sources += f"\n\n📄 Source Bug: [Bugzilla Bug: Bug ID {id}](http://bugzilla.asicdesigners.com/bugs/show_bug.cgi?id={id})"


            yield json.dumps({"message": {"content": sources}, "done": True}) + "\n"


        return StreamingResponse(
            generate_full_response(),
            media_type="application/x-ndjson"
        )    

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    print("-------(START MAIN())--------")

    # Start Ollama if it's not already running
    if not is_ollama_running():
        try:
            subprocess.Popen(["ollama", "serve"])   # starts the Ollama server using subprocess
            time.sleep(2)   # waits 2 seconds to allow the server to start
        except FileNotFoundError:
            raise RuntimeError("Ollama not found. Make sure it's installed and in PATH.")

    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import json
from pydantic import BaseModel
import time

import os

from byaldi import RAGMultiModalModel
from typing import List, Dict, Any
import base64
from io import BytesIO
from PIL import Image

import ollama

from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pathlib import Path
import html

app = FastAPI()

# serve PDFs
PDF_DIR = "../ALL_DATA/"

#app.mount(
#    "/pdfs/T6",
#    StaticFiles(directory="/home/mann/RAG/ALL_DATA/T6/downloaded_pdfs", check_dir=True),
#    name="pdfs_t6",
#)
#app.mount(
#    "/pdfs/T7",
#    StaticFiles(directory="/home/mann/RAG/ALL_DATA/T7/downloaded_pdfs", check_dir=True),
#    name="pdfs_t7",
#)
#
## quiet the 404 (not required, but nice)
#@app.get("/manifest.json")
#def _manifest():
#    return JSONResponse({"name": "Chelsio Chatbot", "start_url": "/gradio", "display": "standalone"})


# ==== CONFIGURATION ====

Supported_VLLMs = ["gemma3:12b"]
Supported_LLMs = ["deepseek-r1:32b", "deepseek-r1:14b", "phi4:latest"]


# === Setting Up the Ollama Server ====
import subprocess   # allows running system commands(like starting the Ollama Server)
import time     # provides time related funcs like sleep()
import socket   # used for network communication, in our case to ches if Ollama server is running

def is_ollama_running(host="127.0.0.1", port=11434):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


class ByaldiRetriever:
    def __init__(self, index_name: str, index_root: str = ".byaldi"):
        """Initialize retriever with saved index"""
        self.model = RAGMultiModalModel.from_index(index_name, index_root=index_root)  

    def retrieve_similar_chunks(self, query: str, k: int=2):#, filter_metadata: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """  
        Retrieve similar document chunks based on query  
          
        Args:  
            query: Search query text  
            k: Number of results to return  
            filter_metadata: Optional metadata filters  <Not Included for Now>
              
        Returns:  
            List of retrieved chunks with metadata  
        """ 
        
        # Perform search using Byaldi's search method
        results = self.model.search(query, k=k)#, filter_metadata=filter_metadata)

        # Format result for easier consumption
        formatted_results = []
        for result in results:
            chunk_info = {  
                'doc_id': result.doc_id,  
                'page_num': result.page_num,  
                'score': result.score,  
                #'metadata': result.metadata,  
                'base64_image': result.base64,  # Available if store_collection_with_index=True  
                'file_name': self.model.get_doc_ids_to_file_names().get(result.doc_id, "Unknown")  
            }  
            formatted_results.append(chunk_info)  
              
        return formatted_results  

retriever_t6 = ByaldiRetriever("ALL_T6_PDF_Embeddings_Store_with_b64")
retriever_t7 = ByaldiRetriever("ALL_T7_PDF_Embeddings_Store_b64")



def read_page_markdown(pdf_name: str, page: int | str, out_root: str | Path = ".") -> str:
    """
    Read the Markdown for a given PDF's page number (works whether files are padded or not).

    Args:
        pdf_name: The original PDF file name, e.g., "report.pdf" or "report".
        page:     Page number (no padding), e.g., 1 or "1".
        out_root: Root dir where the per-PDF folders live (default current dir).

    Returns:
        The Markdown content as a string.

    Raises:
        FileNotFoundError if the folder or page file can't be found.
    """
    page = int(page)
    pdf_stem = Path(pdf_name).stem
    pdf_dir = Path(out_root) / pdf_stem

    print("-----------PDF_DIR=",pdf_dir)
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"Folder not found for PDF: {pdf_dir}")

    # 1) Try plain (no padding) first: "1.md"
    candidate = pdf_dir / f"{page}.md"
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")

    # 2) Try some common paddings: "01.md", "001.md", ...
    for width in range(2, 8):  # adjust upper bound if you use very deep padding
        candidate = pdf_dir / f"{page:0{width}d}.md"
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")

    # 3) Fallback: scan *.md and match numeric stem ignoring leading zeros
    for p in pdf_dir.glob("*.md"):
        try:
            if int(p.stem.lstrip("0") or "0") == page:
                return p.read_text(encoding="utf-8")
        except ValueError:
            # ignore weirdly named files
            pass

    raise FileNotFoundError(f"No .md found for page {page} in {pdf_dir}")

def Query_VLLM(query, model, pages, chat_history):
    # Pass the Retrieved pages directly to the VLLM

    if model == "llama3.2-vision:latest":
        # Only Passing a single most relevant image to these VLLMs
        # These VLLMs only accept a single image in their prompt
        chat_history.append({
            'role': 'user',
            'content': query,
            'images': pages[0]
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
    else:
        # These VLLMs can accept multiple images in their prompt
        chat_history.append({
            'role': 'user',
            'content': query,
            'images': pages
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
    
    return

def next_pg_exists(filename, page_num, Doc_Src):
    if Doc_Src == "T6 Docs":
        if os.path.exists(os.path.join(f"../ALL_DATA/Docling_Parsed_PDFS/T6/{filename}", f"{page_num}.md")):
            return True
        for width in range(2, 8):  # adjust upper bound if you use very deep padding
            if os.path.exists(os.path.join(f"../ALL_DATA/Docling_Parsed_PDFS/T6/{filename}", f"{page_num:0{width}d}.md")):
                return True
    elif Doc_Src == "T7 Docs":
        if os.path.exists(os.path.join(f"../ALL_DATA/Docling_Parsed_PDFS/T7/{filename}", f"{page_num}.md")):
            return True
        for width in range(2, 8):  # adjust upper bound if you use very deep padding
            if os.path.exists(os.path.join(f"../ALL_DATA/Docling_Parsed_PDFS/T7/{filename}", f"{page_num:0{width}d}.md")):
                return True


def Query_LLM(user_query, model, pages, Doc_Src, chat_history):
    # I have page by page parsed my corpus of docs using Docling into .md format
    # Since I now have the relevant pages for the given user query I simply need to inject those into the prompt for the llm

    # I pass both the most relevant page and the one after that just incase theres content spillover into the next page

    relevant_pages = []

    for page in pages:
        relevant_pages.append([page['file_name'], page['page_num']])
        # only if the next page exists should it be appended
        if next_pg_exists(page['file_name'], page['page_num'], Doc_Src):
                relevant_pages.append([page['file_name'], page['page_num']+1])

    context = ""

    print("-------(2.1.)--------")


    if (Doc_Src == "T6 Docs"):
        for page in relevant_pages:
            context += read_page_markdown(page[0], page[1], "../ALL_DATA/Docling_Parsed_PDFS/T6")

    elif (Doc_Src == "T7 Docs"):
        for page in relevant_pages:
            context += read_page_markdown(page[0], page[1], "../ALL_DATA/Docling_Parsed_PDFS/T7")

    else:
        #ADD THIS LATER
        return

    chat_history.append({
        'role': 'user',
        'content': f'USER QUESTION: {user_query} CONTEXT: {context}'
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


@app.post("/chat")
async def handle_rag_request(request: Request):
    try:
        print("-------(0.)--------")
        body = await request.json()
        print("Received body:", body)  # Debug what's actually being sent  
        rag_request = RagRequest(**body)
        #print("Document Src= ", rag_request.documentSource)
        if (rag_request.documentSource == "t6-docs"):
            Doc_Src = "T6 Docs"
        elif (rag_request.documentSource == "t7-docs"):
            Doc_Src = "T7 Docs"

        user_query = rag_request.userQuery
        selected_model = rag_request.selectedModel
        chat_history = rag_request.conversationHistory

        # Retrieve Relevant Documents

        if (Doc_Src == "T6 Docs"):
            retriever = retriever_t6
        elif(Doc_Src == "T7 Docs"):
            retriever = retriever_t7
        else:
            # ADD THIS LATER
            return

        relevant_pages = retriever.retrieve_similar_chunks(user_query)

        b64_pages = []
        for page in relevant_pages:
            b64_pages.append(page['base64_image'])

        print("-------(1.)--------")
        # Whats done next is determined by which model is chosen(VLLM or regular LLM)

        async def generate_full_response():
            # Stream LLM response
            if (selected_model in Supported_VLLMs):
                for chunk in Query_VLLM(user_query, selected_model, b64_pages, chat_history):
                    yield chunk
            else:
                for chunk in Query_LLM(user_query, selected_model, relevant_pages, Doc_Src, chat_history):
                    yield chunk                

            print("-------(2.)--------")

            sources = "\n\n📚 Relevant Documents:\n\n"
            for page in relevant_pages:
                if (Doc_Src == "T6 Docs"):
                    pdf_url =  f"http://10.192.195.31:3003/pdfs/T6/{os.path.basename(page['file_name'])}#page={page['page_num']}"
                elif(Doc_Src == "T7 Docs"):
                    pdf_url =  f"http://10.192.195.31:3003/pdfs/T7/{os.path.basename(page['file_name'])}#page={page['page_num']}"
                else:
                    # ADD THIS LATER
                    return
                #sources += f"📄 Source File: <a href='{pdf_url}' target='_blank'>{os.path.basename(page['file_name'])}</a> 🗒️ Page Number: {page['page_num']}\n"
                sources += f"\n\n📄 Source File: [{os.path.basename(page['file_name'])}]({pdf_url}) 🗒️ Page Number: {page['page_num']}"
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
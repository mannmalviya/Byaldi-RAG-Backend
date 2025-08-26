from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

app = FastAPI()

# Serve PDF directories
app.mount(
    "/pdfs/T6",
    StaticFiles(directory="/home/mann/RAG/ALL_DATA/T6/downloaded_pdfs", check_dir=True),
    name="pdfs_t6",
)

app.mount(
    "/pdfs/T7",
    StaticFiles(directory="/home/mann/RAG/ALL_DATA/T7/downloaded_pdfs", check_dir=True),
    name="pdfs_t7",
)

# Quiet the 404
@app.get("/manifest.json")
def _manifest():
    return JSONResponse({"name": "Chelsio Chatbot", "start_url": "/gradio", "display": "standalone"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3003)  # Different port!
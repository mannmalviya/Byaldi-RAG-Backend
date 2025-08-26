

from byaldi import RAGMultiModalModel
from pathlib import Path

def index_pdf_directory(PDF_DIR, index_name):

    # Initialize the model  
    RAG = RAGMultiModalModel.from_pretrained("vidore/colqwen2-v1.0")  # Instead of v0.2

    # Index the entire directory

    RAG.index(  
        input_path=PDF_DIR,  
        index_name=index_name,  
        store_collection_with_index=True,  
        overwrite=True  # Set to False if you don't want to overwrite existing indexes  
    )      

    print(f"Successfully indexed PDFs from {PDF_DIR}")  # Use the actual parameter name
    print(f"Index saved as: {index_name}")  
    return RAG

if __name__ == "__main__":
    PDF_DIR = "/home/RAG/ALL_DATA/T7/downloaded_pdfs"
    index_name = "ALL_T7_PDF_Embeddings_Store_b64"     # Basically the folder that will store all embeddings

    model = index_pdf_directory(PDF_DIR, index_name)  


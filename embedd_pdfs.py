import pymupdf
import os
import sys
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings
)


def extract_text_from_pdf(pdf_path):
    
    # Open the PDF file
    pdf_document = pymupdf.open(pdf_path)
    
    # Initialize an empty string to store the text
    text = ""
    
    # Iterate over each page in the PDF document
    for page_number in range(len(pdf_document)):
        # Get the page
        page = pdf_document.load_page(page_number)
        
        # Extract the text from the page
        text += page.get_text()
        
    # Close the PDF document
    pdf_document.close()
    
    return text        

#Iterate over all files in the data folder
data_folder = "data"

if not os.path.exists(data_folder):
    print("Data folder not found. Please make sure the data folder exists and contains PDF files.")
    sys.exit(1)

Settings.embed_model = HuggingFaceEmbedding(
    model_name='math-similarity/Bert-MLM_arXiv-MP-class_zbMath'
)

splitter = SentenceSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

PERSIST_DIR = "persisted_index"
if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader(data_folder).load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter]
    )
    
    index.storage_context.persist(PERSIST_DIR)
else:
    print("Persisted index found. No need to recompute.")
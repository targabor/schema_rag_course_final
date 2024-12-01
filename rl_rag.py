from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from dotenv import load_dotenv
import os
import sys

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("AZURE_OPENAI_MODEL")
deployment_name = os.getenv("OPENAI_DEPLOYMENT")
azure_endpoint = os.getenv("OPENAI_AZURE_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION")



PERSIST_DIR = 'persisted_index'
if not os.path.exists(PERSIST_DIR):
    print(f"Persist directory {PERSIST_DIR} does not exist. Please run the embedding script first.\n> python embedd_pdfs.py")
    sys.exit(1)

embed_model = HuggingFaceEmbedding(
    model_name="math-similarity/Bert-MLM_arXiv-MP-class_zbMath",
    trust_remote_code=True
)
Settings.embed_model = embed_model

llm = AzureOpenAI(
    model=model,
    deployment_name=deployment_name,
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)
Settings.llm = llm

storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(similarity_top_k=10)

response = query_engine.query("What is Deep Learning?")


import chainlit as cl

from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureOpenAI

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("AZURE_OPENAI_MODEL")
deployment_name = os.getenv("OPENAI_DEPLOYMENT")
azure_endpoint = os.getenv("OPENAI_AZURE_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION")


try:
    Settings.llm = AzureOpenAI(
        model=model,
        deployment_name=deployment_name,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
    )
    
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="math-similarity/Bert-MLM_arXiv-MP-class_zbMath",
        trust_remote_code=True
    )
    
    Settings.context_window = 4096
    PERSIST_DIR = 'persisted_index'

    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
except Exception as e:
    print(e)
    

@cl.on_chat_start
async def start():
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=10)
    cl.user_session.set("query_engine", query_engine)
    
    await cl.Message(
        author="Assistant", content="Hello! Im an AI assistant. How may I help you?"
    ).send()
    
@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")
    
    msg = cl.Message(content="", author="Assistant")
    
    res = await cl.make_async(query_engine.query)(message.content)
    
    if res:
        msg.content = res.get_response().response
    else:
        msg.content = "Sorry, I don't have an answer to that."
    
    await msg.send()
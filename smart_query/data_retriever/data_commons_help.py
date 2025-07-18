import json
import requests
import chromadb
import logging
import os
from json import JSONDecodeError

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

load_dotenv() 

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

client = chromadb.PersistentClient(path=os.getenv("DATA_COMMONS_VECTOR_DB"))
collection = client.get_collection(name="data_commons_variables")

logging.basicConfig(level=logging.ERROR)

n_results = 50

def ndp_search(search_terms: str):

    results = collection.query(
        query_embeddings = [ embed_model.get_text_embedding(search_terms) ],
        n_results=n_results
    )

    ids = results['ids'][0]
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]

    answers = []
    for i in range(0, n_results):
        answer = {}
        answer["variable"] = documents[i]
        answer["name"] = metadatas[i]["name"]
        answers.append(answer)

    return answers



from qdrant_client import QdrantClient, models
import os
import time
from typing import Any
from abc import ABC

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


qdrant_url = os.getenv("QDRANT_URL", "http://0.0.0.0:6333")

class QdrantDatabase(ABC):
    
    def __init__(self, qdrant_url=qdrant_url):
        self.qdrant_client = QdrantClient(url=qdrant_url)
    
    def create_collection(self, collection_name):
        if self.qdrant_client.collection_exists(collection_name=collection_name):
            logger.info(f"Collection name: {collection_name} already exists.")
            return
        
        self.qdrant_client.create_collection(collection_name=collection_name, vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE))
        logger.info(f"Succesfully created collection: {collection_name}")
        
        
    def add_response(self, collection_name, query, response_text):
        payload = {
            "user": query,
            "system": response_text,
            "timestamp": int(time.time())
        }
        point_data = models.PointStruct(
            id=payload['timestamp'], # id is used as timestamp for convenience as retrieved results are automatically sorted by id.
            vector=[0], # Placeholder vector as we only care about payload for now
            payload=payload
            )
        
        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=[point_data]
        )
        logger.info(f"Added {str(payload)} to the database")
        
        
    def load_recent_responses(self, collection_name, limit: int=10):
        search_results = self.qdrant_client.scroll(
            collection_name = collection_name,
            with_payload=True
        )[0]
        
        print(len(search_results))
        
        return [response.payload for response in search_results[-limit:]]
    
    def delete_collection(self, collection_name):
        if self.qdrant_client.delete_collection(collection_name=collection_name):
            logger.info(f"Deleted collection: {collection_name}")
            return
        logger.info(f"Cannot Delete collection: {collection_name}")
        

    
    
    

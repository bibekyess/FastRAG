from pydantic import BaseModel
from typing import Any
from qdrant_client import models
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import Document
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BGE_M3_EMBED_SIZE=1024 # FIXME

class BaseRetriever(BaseModel):
    index_id: str
    embed_model: Any
    qdrant_client: Any
    reranker_model: Any = None
    
        
    class Config:
        arbitrary_types_allowed = True

    def load_index(self):
        qdrant_vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.index_id,
            enable_hybrid=False,  #  whether to enable hybrid search using dense and sparse vectors
        )
        if self.qdrant_client.collection_exists(collection_name=f"{self.index_id}"):
            logger.info(f"Collection {self.index_id} already exists.")
            logger.info("Loading existing vector store ...")
            qdrant_index = VectorStoreIndex.from_vector_store(vector_store=qdrant_vector_store, embed_model=self.embed_model)
            logger.info(f"Succesfully loaded index for collection {self.index_id}")
            return qdrant_index
        
        else:
            self.qdrant_client.create_collection(
                collection_name=self.index_id,
                on_disk_payload=True,  # TODO Seems not working
                vectors_config=models.VectorParams(size=BGE_M3_EMBED_SIZE, distance=models.Distance.COSINE, on_disk=True),
                optimizers_config=models.OptimizersConfigDiff(memmap_threshold=10000),
                hnsw_config=models.HnswConfigDiff(on_disk=False),
            )  
            qdrant_index = VectorStoreIndex.from_vector_store(vector_store=qdrant_vector_store, embed_model=self.embed_model)
            return qdrant_index
        

    def delete_index(self, index_id):
        all_collections = self.qdrant_client.get_collections()
        existing_collections = [i.name for i in list(all_collections.collections) if index_id in i.name]
        if len(existing_collections) == 0:
            logger.info(f"Cannot delete! Looks like {index_id} collection is empty")
        else:
            for idx_name in existing_collections:
                if self.qdrant_client.delete_collection(collection_name=f"{idx_name}"):
                    logger.info("Deleted all local indexes created with name '%s'", idx_name)
                else:
                    logger.info(f"Cannot delete! Looks like {idx_name} collection is empty")


    def initialize_text_node_from_document(self, doc) -> TextNode:
        doc_data: Dict[str, Any] = {
            "id_": doc.id_,
            "embedding": self.embed_model.get_text_embedding(doc.text),
            "metadata": doc.metadata,
            "excluded_embed_metadata_keys": doc.excluded_embed_metadata_keys,
            "excluded_llm_metadata_keys": doc.excluded_llm_metadata_keys,
            "relationships": doc.relationships,
            "text": doc.text,
            "mimetype": doc.mimetype,
            "start_char_idx": doc.start_char_idx,
            "end_char_idx": doc.end_char_idx,
            "text_template": doc.text_template,
            "metadata_template": doc.metadata_template,
            "metadata_seperator": doc.metadata_seperator,
        }

        text_node = TextNode(**doc_data)
        return text_node

    def add_documents_to_index(
        self,
        documents: List[Document],
        index_id: str,
    ):
        """Upload documents to an index"""
        nodes = [self.initialize_text_node_from_document(d) for d in documents]

        qdrant_vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=index_id,
        )

        document_ids = qdrant_vector_store.add(nodes)
        return {"document_ids": document_ids}
               
                
    def dense_search(self, question: str, rerank:bool=False, dense_threshold: float=0.5, top_k: int=10, top_n: int=5):
        if rerank:
            raise NotImplementedError("Reranker is not implemented yet!")
        index = self.load_index()
        dense_retriever = index.as_retriever(similarity_top_k=top_k)
        response = dense_retriever.retrieve(question)
        response = [r for r in response if r.score > dense_threshold]
        return response
    
    
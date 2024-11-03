import logging, os
from datetime import datetime
from pydantic import BaseModel
from fastapi import UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi import FastAPI
import tempfile, shutil
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from llama_index.core.schema import Document
from typing import Dict, Any, List, Literal
from fastrag.rag.retriever import BaseRetriever


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

current_date = datetime.now().strftime('%Y-%m-%d')
log_file = os.path.join(log_dir, f'rag-api-{current_date}.log')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)






qdrant_url = "http://0.0.0.0:6333"
model_name = "BAAI/bge-m3"
qdrant_client = QdrantClient(url=qdrant_url, timeout=20)
EMBED_SIZE=512
sllm_model_url = 'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_0.gguf'

embed_model = HuggingFaceEmbedding(model_name=model_name, max_length=EMBED_SIZE) # FIXME Changel max_length to consider high memory usage with bge-m3
LLM_PROMPT="""
TEST
"""



async def text_streamer(streamer, extra_text: str = ""):
    async for new_text in streamer:
        yield new_text.delta  # Yielding the streamed text
    if extra_text:
        yield extra_text


class ChatRequest(BaseModel):
    user_input: str
    index_id: str="files"
    llm_text: str="local"
    dense_top_k: int=5
    stream: bool=True
    

    
class SearchRequest(BaseModel):
    user_input: str
    index_id: str="files"
    dense_top_k: int=5
    

@app.post("/parse")
async def parse(file: UploadFile = File(...), index_id: str="files", splitting_type: Literal['raw', 'md'] = 'raw'):
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    if splitting_type == 'raw':
        from fastrag.parser.llamaindex_readers.fitz_pdf_reader import FitzPDFReader
        pdf_reader = FitzPDFReader(overlap=False)
        documents = pdf_reader.load_data(file=file_path)
    elif splitting_type == 'md':
        import pymupdf4llm
        llama_reader = pymupdf4llm.LlamaMarkdownReader()
        documents = llama_reader.load_data(file_path)
    else:
        assert False
        
    base_retriever = BaseRetriever(index_id=index_id, embed_model=embed_model, qdrant_client=qdrant_client)
    return base_retriever.add_documents_to_index(documents=documents, index_id='delete')
            
    
class AnswerSchema(BaseModel):
    text: Any
    source_nodes: List[Any]
    
    
@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.user_input
    index_id = request.index_id
    llm_text = request.llm_text
    dense_top_k = request.dense_top_k
    stream =request.stream
    
    from llama_index.llms.llama_cpp import LlamaCPP

    llm = LlamaCPP(model_url=sllm_model_url, model_kwargs={"n_gpu_layers": 0}, verbose=True)
    
    base_retriever = BaseRetriever(index_id=index_id, embed_model=embed_model, qdrant_client=qdrant_client)
    relevant_documents = base_retriever.dense_search(user_input, top_k=dense_top_k)
    
    context_str = "\n\n".join([n.node.get_content() for n in relevant_documents])
    passed_llm_prompt = LLM_PROMPT.format(context_str=context_str, query_str=user_input)
    logger.info(f"passed llm prompt: {str(passed_llm_prompt)}")
    if stream:
        response = llm.stream_complete(
            passed_llm_prompt
        )
        return StreamingResponse(text_streamer(response), media_type="text/plain")
    else:
        response = llm.complete(
            passed_llm_prompt
        )
        return {'response': response}
    
        
    
    

    
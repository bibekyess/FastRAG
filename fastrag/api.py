import logging, os
from datetime import datetime
from pydantic import BaseModel
from fastapi import UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi import FastAPI
import tempfile, shutil
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from typing import Any, List, Literal
from fastrag.rag.retriever import BaseRetriever
from fastrag.utilities.qdrant_database import QdrantDatabase
import requests
import json

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

# Set up stream handler to print logs to the terminal
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


qdrant_url = os.getenv("QDRANT_URL", "http://0.0.0.0:6333")
model_name = "BAAI/bge-m3"
qdrant_client = QdrantClient(url=qdrant_url, timeout=20)
EMBED_SIZE=512


embed_model = HuggingFaceEmbedding(model_name=model_name, max_length=EMBED_SIZE) # FIXME Changel max_length to consider high memory usage with bge-m3
qdrant_database = QdrantDatabase()
collection_name = "qna_collection"
qdrant_database.create_collection(collection_name)


LLM_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a Document Expert who provides answers based solely on provided references. Follow these instructions:
- Check if the references are relevant to the user query.
- If relevant, provide a precise answer with complete grammar and punctuation.
- If not relevant or the question doesn't make sense given the information, reply: 'It cannot be answered based on the material'.
- Provide only the answer, no other comments.
- Think step by step when formulating your response.
<|eot_id|><|start_header_id|>user<|end_header_id|>

## References:
{context_str}

## User query: 
{query_str}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

QUERY_EXPANSION_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert Teacher specializing in query clarification. Follow these instructions:
- Your task is to rephrase and restructure unclear or incomplete queries. 
- Maintain the original intent and key elements of the question. 
- Use **History** of user and system conversations to extract the intent of the user and rephrase the question.
- Do not answer question - focus solely on rephrasing it for improved comprehension if **History** is provided. 
- Important: Begin your response directly with the rephrased query, without any introductory phrases like 'Here's a rephrased version' or 'The query can be restructured as'. Your entire response should be just the rephrased query.
<|eot_id|><|start_header_id|>user<|end_header_id|>

## History:
{history}

## User query:
{query_str}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

class ChatRequest(BaseModel):
    user_input: str
    index_id: str="files"
    llm_text: str="local"
    dense_top_k: int=5
    upgrade_user_input: bool=False
    stream: bool=True
    
class SearchRequest(BaseModel):
    user_input: str
    index_id: str="files"
    dense_top_k: int=5

class ConversationEntry(BaseModel):
    collection_name: str
    query: str
    response_text: str


class AnswerSchema(BaseModel):
    text: Any
    source_nodes: List[Any]
    

app = FastAPI()

async def text_streamer(streamer, extra_text: str = ""):
    for new_text in streamer:
        yield new_text.delta  # Yielding the streamed text
    if extra_text:
        yield extra_text


@app.get("/conversation-history")
async def get_conversation_history(collection_name: str, limit: int=10):
    conversation_history = qdrant_database.load_recent_responses(collection_name=collection_name, limit=limit)
    return {"response": conversation_history}


@app.post("/conversation-history")
async def add_conversation_history(entry: ConversationEntry):
    qdrant_database.add_response(
        collection_name=entry.collection_name,
        query=entry.query,
        response_text=entry.response_text
    )
    return {"message": "Response added to conversation history"}


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
    base_retriever.delete_index(index_id=index_id)
    return base_retriever.add_documents_to_index(documents=documents, index_id=index_id)


def llamacpp_inference(prompt, n_predict=128, temperature=0.7, top_p=0.95, stop=None, stream=True):
    url = os.getenv("LLAMACPP_URL", "http://localhost:8088/completion")
    
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "top_p": top_p,
        "stop": stop if stop else [],
        "stream": stream
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    def handle_streaming():
        with requests.post(url, headers=headers, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if line:  # Filter out keep-alive lines
                    try:
                        # Remove "data: " prefix and parse JSON
                        data = json.loads(line[6:])
                        yield data.get("content", "")
                    except json.JSONDecodeError:
                        print("Failed to decode JSON:", line)

    def handle_non_streaming():
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get('content', '')

    if stream:
        return handle_streaming()
    else:
        return handle_non_streaming()
    

@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.user_input
    index_id = request.index_id
    llm_text = request.llm_text
    dense_top_k = request.dense_top_k
    upgrade_user_input = request.upgrade_user_input
    stream =request.stream
    
    assert llm_text=='local'
    
    base_retriever = BaseRetriever(index_id=index_id, embed_model=embed_model, qdrant_client=qdrant_client)
    
    if upgrade_user_input:
        conversation_history = qdrant_database.load_recent_responses(collection_name=collection_name, limit=5)
        logger.info(f"Conversation History: {conversation_history}")
        llm_prompt =QUERY_EXPANSION_PROMPT.format(history=conversation_history, query_str=user_input)
        user_input = llamacpp_inference(llm_prompt, temperature=0.7,stream=False)
        logger.info(f"Upgraded user_input: {user_input}")
        
    relevant_documents = base_retriever.dense_search(user_input, top_k=dense_top_k, dense_threshold=0.2)
    
    context_str = "\n\n".join([n.node.get_content() for n in relevant_documents])
    passed_llm_prompt = LLM_PROMPT.format(context_str=context_str, query_str=user_input)
    logger.info(f"passed llm prompt: {str(passed_llm_prompt)}")
    if stream:
        streamer = llamacpp_inference(passed_llm_prompt, n_predict=200, temperature=0.3, stream=stream)

        return StreamingResponse(streamer, media_type="text/plain")
    else:
        response = llamacpp_inference(passed_llm_prompt, n_predict=200, temperature=0.3, stream=stream)
        return {'response': response}
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('API_PORT', 8090)))
    
    
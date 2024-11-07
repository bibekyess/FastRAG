# FastRAG

FastRAG is a simple Retrieval-Augmented Generation (RAG) application optimized for fast performance on general-grade PCs. It provides a chatbot interface that leverages vector-based search and large language models (LLMs) for answering questions and interacting with document-based data.

---

### 🚀 Launch API and Demo Locally

To get started with FastRAG locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/bibekyess/FastRAG.git
   ```
   
2. Navigate to the project directory:
   ```bash
   cd FastRAG
   ```

3. Build and launch the containers:
   ```bash
   docker compose up --build
   ```

This will start the FastRAG API and demo with all necessary services.

---

### 🛠️ API Endpoints

The FastRAG application launches several API endpoints for different purposes:

1. **Get Conversation History**  
   - **Method**: `GET`  
   - **Endpoint**: `/conversation-history`  
   - **Parameters**:  
     - `collection_name` (str): Name of the collection to fetch history from.  
     - `limit` (int): Number of history entries to return. Default is 10.

2. **Add to Conversation History**  
   - **Method**: `POST`  
   - **Endpoint**: `/conversation-history`  
   - **Body**:
     - `collection_name` (str): Name of the collection to fetch history from.  
     - `query` (str): User input query 
     - `response_text` (str): AI response  

3. **Parse Document**  
   - **Method**: `POST`  
   - **Endpoint**: `/parse`  
   - **Parameters**:  
     - `file` (UploadFile): The document to be parsed.  
     - `index_id` (str): Index name for the document. Default is `files`.  
     - `splitting_type` (Literal['raw', 'md']): Splitting type for the document. Default is `raw` (based on chunk settings).

4. **Chat with the Bot**  
   - **Method**: `POST`  
   - **Endpoint**: `/chat`  
   - **Body**: 
     - `user_input` (str): The user's query.  
     - `index_id` (str): The index to search. Default is `"files"`.  
     - `llm_text` (str): The LLM model to use. Default is `"local"`.  
     - `dense_top_k` (int): The number of top results to return from the vector search. Default is 5.  
     - `upgrade_user_input` (bool): Flag to indicate whether to upgrade the user input from conversation history. Default is `False`.  
     - `stream` (bool): Flag to enable streaming of results. Default is `True`.  


### 🖥️ User Interface

- **Gradio UI**: FastRAG features a simple Gradio-based user interface for interacting with the chatbot.
- **Real-time Chat**: Users can upload a document and ask questions in real-time, with previous conversations stored and utilized for context-based improvements. [Providing the option to upload document is in progress]

---

### 🗃️ Storage and Database

- **QdrantDB**: The vector embeddings and chatbot conversation history are stored in QdrantDB. This allows the chatbot to utilize previous conversation context for improved responses.
---

### ⚡ Model Backend

- **Model**: [bartowski/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) 

---

### ⏱️ Latency Tracking

- **UI Display**: Latency of the chatbot's response is displayed in the Gradio interface.
- **Logging**: Detailed logs of latency and other events are saved for debugging and performance monitoring.

---


### 🧾 Document Parsing Options

FastRAG offers multiple options for segmenting documents into chunks:

1. **Raw Format**: This option allows experimenting with various chunk sizes, strides, and overlapping settings for raw text parsing.
2. **Markdown Format**: This method segments the document based on semantic information, creating more context-aware chunks.

---


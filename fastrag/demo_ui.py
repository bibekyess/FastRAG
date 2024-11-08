import gradio as gr
import requests
import os
import logging
import time
from time import perf_counter
from datetime import datetime
from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def log_to_file(question, response, latency):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = f"Timestamp: {timestamp}\n"
    log_entry += f"Question: {question}\n"
    log_entry += f"Response: {response}\n"
    log_entry += f"Latency: {latency:.4f} seconds\n"
    log_entry += "-" * 50 + "\n"
    
    # append mode
    with open("./logs/chatbot_log.txt", "a") as file:
        file.write(log_entry)
        
        
def call_parse_api(file_path, index_id, splitting_type):
    url = os.getenv("PARSER_API_URL", "http://localhost:8080") + "/parse"
    headers = {
        "accept": "application/json",
    }
    params = {
        "index_id": index_id,
        "splitting_type": splitting_type,
    }
    
    try:
        with open(file_path, 'rb') as file:
            filename = os.path.basename(file_path)
            logger.info(filename)
            files = {
                'file': (filename, file, 'application/pdf')
            }

            response = requests.post(url, params=params, headers=headers, files=files)
            response.raise_for_status()
            
    except HTTPError as http_err:
        logger.warning(f"HTTP error occurred: {http_err} (Status Code: {response.status_code})")
    except ConnectionError as conn_err:
        logger.warning(f"Connection error occurred: {conn_err}")
    except Timeout as timeout_err:
        logger.warning(f"Timeout error occurred: {timeout_err}")
    except RequestException as req_err:
        logger.warning(f"Request failed: {req_err}")
    except Exception as general_err:
        logger.warning(f"An unexpected error occurred: {general_err}")
    
        
    
def call_chat_api(user_input, index_id):
    url = os.getenv("PARSER_API_URL", "http://localhost:8080") +"/chat"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    data = {
        "user_input": user_input,
        "index_id": index_id,
        "llm_text": "local",
        "dense_top_k": 4,
        "upgrade_user_input": True,
        "stream": True
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, stream=True)
        response.raise_for_status()

        return response.iter_content(chunk_size=None, decode_unicode=True)
    
    except HTTPError as http_err:
        logger.warning(f"HTTP error occurred: {http_err} (Status Code: {response.status_code})")
    except ConnectionError as conn_err:
        logger.warning(f"Connection error occurred: {conn_err}")
    except Timeout as timeout_err:
        logger.warning(f"Timeout error occurred: {timeout_err}")
    except RequestException as req_err:
        logger.warning(f"Request failed: {req_err}")
    except Exception as general_err:
        logger.warning(f"An unexpected error occurred: {general_err}")
    
    return iter([])     


def chat(chatbot_history, splitting_approach):
    start_time = perf_counter()
    user_input = chatbot_history[-1][0] # idx-0 --> User input
    streamer = call_chat_api(user_input, splitting_approach)
    for chunk in streamer:
        chatbot_history[-1][1] += chunk
        yield chatbot_history, "## Calculating Latency ..."
        
    query = chatbot_history[-1][0]
    response_text = chatbot_history[-1][1]
    logger.info(f"query: {query}:  {response_text}")
    add_conversation_history(query=query, response_text=response_text)
    end_time = perf_counter()
    elapsed_time = end_time-start_time
    logger.info(f"Chat API executed in {elapsed_time:.4f} seconds")
    
    # Log the results to a .txt file
    log_to_file(query, response_text, elapsed_time)
    
    yield chatbot_history, f"## Latency of Last Response: {elapsed_time:.4f} seconds"


def get_conversation_history(collection_name: str="qna_collection", limit: int=10, max_retries: int=5, retry_delay: int=2):
    retries = 0
    while retries < max_retries:
        try:
            url = os.getenv("CONVERSATION_HISTORY_URL", "http://localhost:8080/conversation-history")
            payload = {
                "collection_name": collection_name,
                "limit": limit
            }
            response = requests.get(url, params=payload)
            response.raise_for_status()
            
            logger.info(f"JSON Response from Conversation History: {response.json()}")
            conversation_history = response.json()["response"]

            chat_history = []
            for record in conversation_history:
                chat_history += [[record['user'], record['system']]]
            return chat_history
        
        except HTTPError as http_err:
            logger.warning(f"HTTP error occurred: {http_err} (Status Code: {response.status_code})")
        except ConnectionError as conn_err:
            logger.warning(f"Connection error occurred: {conn_err}")
        except Timeout as timeout_err:
            logger.warning(f"Timeout error occurred: {timeout_err}")
        except RequestException as req_err:
            logger.warning(f"Request failed: {req_err}")
        except Exception as general_err:
            logger.warning(f"An unexpected error occurred: {general_err}")
            
        logger.warning(f"Error in accessing conversation history from database")
        retries += 1
        if retries < max_retries:
            logger.warning(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2
        else:
            logger.error(f"Failed to get conversation history after {max_retries} attempts.")
            return []


def add_conversation_history(query, response_text, collection_name:str="qna_collection"):
    try:
        url = os.getenv("CONVERSATION_HISTORY_URL", "http://localhost:8080/conversation-history")
        payload = {
            "collection_name": collection_name,
            "query": query,
            "response_text": response_text
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        logger.info(response.json()["message"])
        
    except HTTPError as http_err:
        logger.warning(f"HTTP error occurred: {http_err} (Status Code: {response.status_code})")
    except ConnectionError as conn_err:
        logger.warning(f"Connection error occurred: {conn_err}")
    except Timeout as timeout_err:
        logger.warning(f"Timeout error occurred: {timeout_err}")
    except RequestException as req_err:
        logger.warning(f"Request failed: {req_err}")
    except Exception as general_err:
        logger.warning(f"An unexpected error occurred: {general_err}")
        
    logger.warning(f"Failed to add to conversation history")
    return []    
    
    
def chatbot_history_collection(input_query, chat_history):
    
    if input_query is None or len(input_query) == 0:
        input_query=""

    return "", chat_history + [[input_query, '']]

def load_chatbot():
    history = get_conversation_history()
    return gr.update(value=history, visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
    
def dummy_return():
    return "## Generation Stopped by User"
    
def parse_document(file_path, splitting_approach):
    if splitting_approach=="raw-512-chunksize":
        splitting_type="raw"
    elif splitting_approach=="markdown-layoutaware":
        splitting_type="md"
    else:
        assert False, f"{splitting_approach} is not supported. Only 'raw-512-chunksize', 'markdown-layoutaware'"
    
    logger.info(file_path)
    call_parse_api(file_path, splitting_approach, splitting_type)
    return f"Parse succesfully created for {splitting_approach}"
        
    
def run():
    with gr.Blocks() as demo:
        with gr.Tab("Document Parsing"):
            gr.Markdown(
                """
                <h1 style="text-align: center; color: #3b3b3b;">📝 Document Parser</h1>
                """,
                elem_id="header"
            )
            
            doc_upload = gr.File(label="Upload Document")
            splitting_approach = gr.Dropdown(
                label="Choose splitting approach",
                choices = ["raw-512-chunksize", "markdown-layoutaware"],
                value="raw-512-chunksize",
                visible=True
            )
            
            parse_button = gr.Button("Parse Document")
            parse_output = gr.Textbox(label="Parsed Content")
            
            parse_button.click(parse_document, inputs=[doc_upload, splitting_approach], outputs=parse_output)
            
    
        with gr.Tab("Q&A"):
            gr.Markdown(
                """
                <h1 style="text-align: center; color: #3b3b3b;">💬 FastRAG Chatbot</h1>
                <h3 style="text-align: center; color: #666;">Upload any PDF and ask anything</h3>
                """,
                elem_id="header"
            )
            loading_indicator = gr.Markdown("## Loading Parser API...", visible=True, elem_id="loading-indicator")
            splitting_approach = gr.Dropdown(
                label="Choose splitting approach",
                choices = ["raw-512-chunksize", "markdown-layoutaware"],
                value="raw-512-chunksize",
                visible=False
            )
                
            chatbot = gr.Chatbot(elem_id="chatbot-display", visible=False)
            input_text = gr.Textbox(
                            placeholder="Type your message...",
                            show_label=False,
                            lines=1,
                            elem_id="user-input",
                            visible=False
                        )
                
            with gr.Row():
                clear_submit_btn = gr.ClearButton(visible=False)
                input_submit_btn= gr.Button("Submit", visible=False)
                stop_btn = gr.Button("Stop", visible=False)

            latency_display = gr.Markdown("", elem_id="latency-display")

            demo.load(
                load_chatbot,
                outputs=[chatbot, input_text, clear_submit_btn, input_submit_btn, stop_btn, loading_indicator, splitting_approach]
            )
            
            clear_submit_btn.add(
                components=[chatbot, input_text, latency_display]
            )
            
            submit_event = input_text.submit(
                fn = chatbot_history_collection,
                inputs=[input_text, chatbot],
                outputs=[input_text, chatbot],
            ).then(
                fn = chat,
                inputs = [chatbot, splitting_approach],
                outputs=[chatbot, latency_display]
            )
            
            click_event = input_submit_btn.click(
                fn = chatbot_history_collection,
                inputs=[input_text, chatbot],
                outputs=[input_text, chatbot],
            ).then(
                fn = chat,
                inputs = [chatbot, splitting_approach],
                outputs=[chatbot, latency_display]
            )
            
            stop_btn.click(fn=dummy_return, outputs=[latency_display], cancels=[click_event, submit_event])

        

    # Add custom CSS styling for a professional look
    demo.css = """
    #chat-container {
        max-width: 600px;
        margin: 0 auto;
    }

    #chatbot-display {
        border: 1px solid #dedede;
        border-radius: 8px;
        background-color: #f7f8fa;
        padding: 20px;
        color: #333333;
        font-family: Arial, sans-serif;
    }

    #user-input {
        border: 1px solid #aaaaaa;
        padding: 10px;
        border-radius: 8px;
        width: 100%;
    }

    #send-button {
        background-color: #0055a5;
        color: #ffffff;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
    }

    #send-button:hover {
        background-color: #004080;
    }
    
    
    #loading-indicator {
        font-weight: bold;
        color: #3498db; /* Soft blue for loading text */
        text-align: center;
        margin-top: 8em;
        margin-bottom: 1em;
    }
    
    #latency-display {
        border: 1px solid #dedede;
        border-radius: 8px;
        background-color: #f7f8fa;
        padding: 20px;
        color: #333333;
        font-family: Arial, sans-serif;
    }
    """

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__=="__main__":
    run()
    
    
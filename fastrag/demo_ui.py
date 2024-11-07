import gradio as gr
import requests
import os
import logging
import time
from time import perf_counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def call_chat_api(user_input):
    url = os.getenv("PARSER_API_URL", "http://localhost:8080/chat")
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    data = {
        "user_input": user_input,
        "index_id": "files",
        "llm_text": "local",
        "dense_top_k": 4,
        "stream": True
    }

    try:
        response = requests.post(url, headers=headers, json=data, stream=True)
        if response.status_code == 200:
                return response.iter_content(chunk_size=None, decode_unicode=True)
        else:
            print(f"Error: {response.status_code}")
            return iter([])
    except requests.exceptions.RequestException as e:
        # Handle any request-related errors
        print(f"Request failed: {e}")
        return iter([])    


def chat(chatbot_history):
    start_time = perf_counter()
    user_input = chatbot_history[-1][0] # idx-0 --> User input
    streamer = call_chat_api(user_input)
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
    
    return chatbot_history, f"## Latency of Last Response: {elapsed_time:.4f} seconds"


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
            logger.info(f"JSON Response from Conversation History: {response.json()}")
            conversation_history = response.json()["response"]

            chat_history = []
            for record in conversation_history:
                chat_history += [[record['user'], record['system']]]
            return chat_history
        except Exception as err:
            logger.warning(f"Error in accessing conversation history from database: {err}")
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
        logger.info(response.json()["message"])
        
    except Exception as err:
        logger.warning(f"Failed to add to conversation history: {err}")
        return []    
    
    
def chatbot_history_collection(input_query, chat_history):
    
    if input_query is None or len(input_query) == 0:
        input_query=""

    return "", chat_history + [[input_query, '']]

def load_chatbot():
    history = get_conversation_history()
    return gr.update(value=history, visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
    

def run():
    with gr.Blocks() as demo:
        # Header with a professional title and subtitle
        gr.Markdown(
            """
            <h1 style="text-align: center; color: #3b3b3b;">💬 FastRAG Chatbot</h1>
            <h3 style="text-align: center; color: #666;">Upload any PDF and ask anything</h3>
            """,
            elem_id="header"
        )
        loading_indicator = gr.Markdown("## Loading Parser API...", visible=True, elem_id="loading-indicator")

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
            outputs=[chatbot, input_text, clear_submit_btn, input_submit_btn, stop_btn, loading_indicator]
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
            inputs = [chatbot],
            outputs=[chatbot, latency_display]
        )
        
        click_event = input_submit_btn.click(
            fn = chatbot_history_collection,
            inputs=[input_text, chatbot],
            outputs=[input_text, chatbot],
        ).then(
            fn = chat,
            inputs = [chatbot],
            outputs=[chatbot, latency_display]
        )
        
        stop_btn.click(fn=lambda: "## Generation Stopped by User", inputs=None, outputs=latency_display, cancels=[click_event, submit_event])

        

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
    
    
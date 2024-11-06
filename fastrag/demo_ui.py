import gradio as gr
import requests
import os
import logging

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
    print("history: ", chatbot_history)
        
    user_input = chatbot_history[-1][0] # idx-0 --> User input
    streamer = call_chat_api(user_input)
    for chunk in streamer:
        chatbot_history[-1][1] += chunk
        yield chatbot_history
    return chatbot_history


def chatbot_history_collection(input_query, chat_history):
    try:
        url = "http://localhost:8080/conversation-history"
        payload = {
            "collection_name": "qna_collection",
            "limit": 10
        }
        response = requests.post(url, json=payload)
        conversation_history = response.json()["response"]

        for record in conversation_history:
            chat_history += [[record['user'], record['assistant']]]
    except Exception as err:
        logger.warning(f"Error in accessing conversation history from database: {err}")
        
    if input_query is None or len(input_query) == 0:
        input_query=""

    return "", chat_history + [[input_query, '']]


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


        chatbot = gr.Chatbot(elem_id="chatbot-display")
        input_text = gr.Textbox(
                        placeholder="Type your message...",
                        show_label=False,
                        lines=1,
                        elem_id="user-input"
                    )
        
        with gr.Row():
            clear_submit_btn = gr.ClearButton(visible=True)
            input_submit_btn= gr.Button("Submit", visible=True)
            stop_btn = gr.Button("Stop", visible=True)

        clear_submit_btn.add(
            components=[chatbot, input_text]
        )
        
        submit_event = input_text.submit(
            fn = chatbot_history_collection,
            inputs=[input_text, chatbot],
            outputs=[input_text, chatbot],
        ).then(
            fn = chat,
            inputs = [chatbot],
            outputs=[chatbot]
        )
        
        click_event = input_submit_btn.click(
            fn = chatbot_history_collection,
            inputs=[input_text, chatbot],
            outputs=[input_text, chatbot],
        ).then(
            fn = chat,
            inputs = [chatbot],
            outputs=[chatbot]
        )
        
        stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[click_event, submit_event])

        

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
    """

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__=="__main__":
    run()
    
    
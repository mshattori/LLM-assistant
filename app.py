#!/usr/bin/env python
import os
import logging
from argparse import ArgumentParser
from dotenv import load_dotenv
from yaml import safe_load

import gradio as gr
from chatbot_engine import chat, create_index_list

from langchain.memory import ChatMessageHistory

# Set up logging
logging.basicConfig(
    filename='app.log', 
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class App:
    def __init__(self, config):
        self.config = config
        self.indexes = create_index_list(config)
        logger.info('App initialized with config: %s', config)

    def _respond(self, message, chat_history):
        history = ChatMessageHistory()
        for [user_message, ai_message] in chat_history:
            history.add_user_message(user_message)
            history.add_ai_message(ai_message)

        logger.info('Input:\n%s', message)
        bot_message = chat(message, history, self.indexes)
        logger.info('Output:\n%s', bot_message)
        chat_history.append((message, bot_message))
        return '', chat_history

    def main(self):
        title = self.config.get('title', 'RAG Assistant')
        with gr.Blocks(title=title) as demo:
            chatbot = gr.Chatbot()
            msg = gr.Textbox()
            clear = gr.Button('Clear')

            msg.submit(self._respond, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)

        app_env = os.environ.get('APP_ENV', 'production')

        if app_env == 'production':
            username = os.environ['GRADIO_USERNAME']
            password = os.environ['GRADIO_PASSWORD']
            auth = (username, password)
        else:
            auth = None

        demo.launch(auth=auth)

if __name__ == '__main__':
    load_dotenv(override=True)

    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = safe_load(f)

    app = App(config)
    app.main()
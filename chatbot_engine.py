#!/usr/bin/env python
import os
import re
from typing import List

from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import VectorStoreInfo, VectorStoreToolkit
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders import (
    DirectoryLoader,
    GitLoader,
    PyPDFLoader
)
from langchain_community.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.tools import BaseTool

from llm import (
    create_llm,
    extend_llm_callbacks,
)
from embeddings import create_embeddings
from message import MessageExpander


class IndexHolder:
    def __init__(self, index: VectorStoreIndexWrapper, config: dict):
        self._index = index
        self._tool_name = config['tool_name']
        self._tool_description = config['tool_description']

    @property
    def index(self):
        return self._index

    @property
    def tool_name(self):
        return self._tool_name

    @property
    def tool_description(self):
        return self._tool_description

def _create_index(loaders: List[BaseLoader], persist_dir: str) -> VectorStoreIndexWrapper:
    embedding = create_embeddings(max_retries=5, chunk_size=16)
    if os.path.exists(persist_dir):
        print(f'Import persistent data from {persist_dir}')
        vectorstore = Chroma(
            embedding_function=embedding,
            persist_directory=persist_dir
        )
        return VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        return VectorstoreIndexCreator(
            vectorstore_cls=Chroma,
            vectorstore_kwargs={"persist_directory": persist_dir},
            embedding=embedding
        ).from_loaders(loaders)

def _get_loader(loader_type: str, loader_kwargs: dict) -> BaseLoader:
    LOADER_CLASSES = {
        'directory': DirectoryLoader,
        'git': GitLoader,
        'pdf': PyPDFLoader
    }
    if loader_type not in LOADER_CLASSES:
        raise ValueError(f'Loader type {loader_type} is not supported')
    loader_class = LOADER_CLASSES[loader_type]
    # Preprocess loader kwargs
    for key, value in loader_kwargs.items():
        if key.endswith('path'):
            loader_kwargs[key] = os.path.abspath(value)
        if key == 'file_filter':
            def filter(f):
                m = re.fullmatch(value, f)
                print(f) if m is not None else print(f, '...Skip')
                return m is not None
            loader_kwargs[key] = filter

    return loader_class(**loader_kwargs)

def create_index_list(config: dict) -> List[IndexHolder]:
    index_list = []

    for index_config in config['indexes']:
        loaders = []
        for loader_config in index_config['loaders']:
            loader = _get_loader(
                loader_type = loader_config['type'],
                loader_kwargs = loader_config['kwargs']
            )
            loaders.append(loader)
        index = _create_index(loaders, index_config['persist_directory'])
        print('Documents loaded')
        index_list.append(IndexHolder(index, index_config))

    return index_list

def _create_tools(indexes: List[IndexHolder], llm) -> List[BaseTool]:
    tools = []
    for index_holder in indexes:
        vectorstore_info = VectorStoreInfo(
            vectorstore=index_holder.index.vectorstore,
            name=index_holder.tool_name,
            description=index_holder.tool_description
        )
        toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)
        tools.extend(toolkit.get_tools())

    return tools

def chat(
    message: str, history: ChatMessageHistory, indexes: List[IndexHolder]
) -> str:
    llm = create_llm()

    tools = _create_tools(indexes, llm)

    memory = ConversationBufferMemory(
        chat_memory=history, memory_key="chat_history", return_messages=True
    )

    agent_chain = initialize_agent(
        tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        handle_parsing_errors=True
    )
    # Expand message commands
    messages = [MessageExpander().expand_message(message)]

    response = agent_chain.invoke({'input': messages}, config={'callbacks': extend_llm_callbacks()})
    return response['output']

def main():
    from dotenv import load_dotenv
    from argparse import ArgumentParser
    from yaml import safe_load

    load_dotenv(override=True)

    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config.yaml')
    parser.add_argument('--prompt-file', '-p', required=True,
                        help='Prompt file')
    
    args = parser.parse_args()
    with open(args.prompt_file) as f:
        prompt = f.read()
    with open(args.config, 'r') as f:
        config = safe_load(f) 
    indexes = create_index_list(config)
    response = chat(prompt, ChatMessageHistory(), indexes)
    print(response)

if __name__ == '__main__':
    main()
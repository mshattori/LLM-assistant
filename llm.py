#!/usr/bin/env python
import os
import re
import sys
from argparse import ArgumentParser

from dotenv import load_dotenv
import yaml

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


def create_llm(model_name: str = None, **kwargs) -> ChatOpenAI:
    """Create and return an LLM instance based on the specified model name.

    Args:
        model_name: The name of the model to use.
        **kwargs: Additional keyword arguments for the LLM constructor.

    Returns:
        An instance of ChatOpenAI or AzureChatOpenAI.
    """
    openai_api_type = os.environ.get('OPENAI_API_TYPE', None)

    if openai_api_type == 'azure':
        if not model_name:
            model_name = os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME', None)
        return AzureChatOpenAI(azure_deployment=model_name, **kwargs)

    if not model_name:
        model_name = os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini')

    return ChatOpenAI(model_name=model_name, **kwargs)


def extend_llm_callbacks(callbacks: list = []) -> list:
    """Extend the list of callbacks with LangFuse callback if applicable.

    Args:
        callbacks: A list of existing callbacks.

    Returns:
        A list of callbacks including the LangFuse callback if applicable.
    """
    if os.environ.get('LANGFUSE_PUBLIC_KEY'):
        from langfuse.callback import CallbackHandler
        callbacks.append(CallbackHandler())
    return callbacks


def expand_message(message: str) -> str:
    """Expand the message by handling special commands for file imports.

    Args:
        message: The message string that may contain file import commands.

    Returns:
        The expanded message with file contents included.
    """
    matches = re.findall(r'\[(.*)\]\((.+)\)', message)
    for name, path in matches:
        match = f'[{name}]({path})'
        name = name.strip()
        path = path.strip()

        if os.path.exists(path):
            with open(path, 'r') as f:
                content = f.read()
            if not name:
                name = os.path.basename(path)
            if name:
                content = f'### {name} ###\n```\n{content}\n```\n'
            message = message.replace(match, content)
        else:
            print(f'File not found: {path}', file=sys.stderr)

    return message


def load_prompt_file(prompt_file: str) -> dict:
    """Load the prompt file and return its contents.

    Args:
        prompt_file: The path to the prompt file in YAML format.

    Returns:
        A dictionary containing the parsed prompt data.
    """
    with open(prompt_file) as f:
        return yaml.safe_load(f)


def main():
    """Main function to execute the script."""
    load_dotenv()
    parser = ArgumentParser()
    parser.add_argument('--prompt-file', '-p', required=False,
                        help='Prompt file in YAML format')
    parser.add_argument('--output-file', '-o', required=False)
    parser.add_argument('--enable-import', '-i', action='store_true',
                        default=True, help='Enable file imports in messages')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--message', '-m', type=str, required=False)
    args = parser.parse_args()

    if not args.message and not args.prompt_file:
        parser.print_usage()
        sys.exit(1)

    messages = []
    if args.prompt_file:
        prompt = load_prompt_file(args.prompt_file)
        if 'system' in prompt:
            messages.append(SystemMessage(prompt['system']))

        if 'user' in prompt:
            if args.message:
                messages.append(HumanMessage(prompt['user']))
            else:
                args.message = prompt['user']

    messages.append(HumanMessage(args.message))

    if args.enable_import:
        for message in messages:
            message.content = expand_message(message.content)

    model_name = prompt.get('model') if args.prompt_file else None
    llm = create_llm(model_name=model_name)
    response = llm.invoke(messages, config={'callbacks': extend_llm_callbacks()})

    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(response.content)

    print(response.content)

    if args.verbose:
        from pprint import pprint
        detail = response.dict()
        detail.pop('content')  # Remove the content key
        pprint(detail)


if __name__ == '__main__':
    main()

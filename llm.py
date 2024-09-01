#!/usr/bin/env python
import os
import sys
from argparse import ArgumentParser

from dotenv import load_dotenv
import yaml

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from message import MessageExpander


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
    load_dotenv(override=True)
    parser = ArgumentParser()
    parser.add_argument('--prompt-file', '-p', required=False,
                        help='Prompt file')
    parser.add_argument('--system-file', '-s', required=False,
                        help='System file in YAML format')
    parser.add_argument('--output-file', '-o', required=False)
    parser.add_argument('--enable-import', '-i', action='store_true',
                        default=True, help='Enable file imports in messages')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--message', '-m', type=str, required=False)
    args = parser.parse_args()

    if not args.message and not args.prompt_file:
        parser.print_usage()
        sys.exit(1)

    if args.enable_import:
        expand_fn = MessageExpander().expand_message
    else:
        expand_fn = lambda s: HumanMessage(s)

    messages = []

    # Append system message if provided
    if args.system_file:
        with open(args.system_file) as f:
            system_config = yaml.safe_load(f)
        if 'system' in system_config:
            messages.append(SystemMessage(system_config['system']))

    # Append user message if provided
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read()
        if args.message:
            messages.append(expand_fn(prompt))
        else:
            args.message = prompt

    # Append user message from command line
    messages.append(expand_fn(args.message))

    model_name = system_config.get('model') if args.system_file else None
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

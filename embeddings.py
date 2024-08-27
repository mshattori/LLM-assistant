import os

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings


def create_embeddings(model_name=None, **kwargs):
    openai_api_type = os.environ.get('OPENAI_API_TYPE', None)
    if openai_api_type == 'azure':
        if not model_name:
            model_name = os.environ.get('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME', None)
        embeddings = AzureOpenAIEmbeddings(azure_deployment=model_name, **kwargs)
    else:
        if not model_name:
            model_name = os.environ.get('OPENAI_EMBEDDING_NAME', 'text-embedding-3-small')
        embeddings = OpenAIEmbeddings(model=model_name, **kwargs)
    return embeddings

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    embeddings = create_embeddings()

    print(embeddings.embed_query('Hello'))

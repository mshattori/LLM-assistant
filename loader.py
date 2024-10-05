import os
import re
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.document_loaders import WebBaseLoader


# dependencies: atlassian-python-api, lxml
class ConfluencePageLoader:
    def __init__(self):
        self.base_url = os.environ.get('CONFLUENCE_WIKI_URL')
        self.username = os.environ.get('ATTLASIAN_USER_EMAIL')
        self.api_key = os.environ.get('ATTLASIAN_API_TOKEN')

    def is_target_path(self, url):
        return url.startswith(self.base_url)

    def load(self, path):
        match = re.match(rf'{self.base_url}/(?:\S+/)+pages/(\d+)/.*', path)
        if not match:
            raise ValueError(f'Invalid Confluence URL: {path}')
        page_id = match.group(1)
        loader = ConfluenceLoader(
            url=self.base_url,
            username=self.username,
            api_key=self.api_key,
            page_ids=[page_id],
            keep_markdown_format=True
        )
        doc = loader.load()[0]
        return doc

# dependencies: langchain_community, beautifulsoup4
class WebPageLoader:
    def __init__(self):
        pass

    def is_target_path(self, path):
        from urllib.parse import urlparse
        parsed_url = urlparse(path)
        return parsed_url.scheme in ['http', 'https']

    def load(self, path):
        loader = WebBaseLoader(path)
        doc = loader.load()[0]
        if not doc.metadata.get('title'):
            doc.metadata['title'] = 'Untitled'
        return doc

if __name__ == '__main__':
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--output', '-o', required=False, help='Output file')
    parser.add_argument('source', help='The source URL or path')

    args = parser.parse_args()

    loaders = [ConfluencePageLoader(), WebPageLoader()]
    for loader in loaders:
        if loader.is_target_path(args.source):
            doc = loader.load(args.source)
            break

    if not doc:
        raise ValueError('No loader found for the given URL or path')

    def print_doc(doc, f):
        if doc.metadata.get('title'):
            f.write(f'# {doc.metadata["title"]}\n\n')
        f.write(doc.page_content)

    if args.output:
        with open(args.output, 'w') as f:
            print_doc(doc, f)

    print_doc(doc, sys.stdout)

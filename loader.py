#!/usr/bin/env python
import os
import re
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import YoutubeLoader as YtLoader
from langchain_core.documents import Document


class LoaderOptionParser():
    def __init__(self):
        self.options = {}

    def parse(self, options_str):
        for option in options_str.split(';'):
            key, value = option.split('=', 1)
            key, value = key.strip(), value.strip()
            if key == 'pages':
                value = self._parse_pages(value)
            self.options[key] = value
        return self.options

    @staticmethod
    def _parse_pages(pages: str) -> list:
        """Parse the pages option and return a list of zero-indexed page numbers."""
        page_numbers = set()
        for part in pages.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                page_numbers.update(range(start, end + 1))
            else:
                page_numbers.add(int(part))
        # Convert to zero-indexed page numbers
        page_numbers = [n - 1 for n in page_numbers]
        if any(n < 0 for n in page_numbers):
            raise ValueError(f'Invalid page numbers: {pages}')
        return list(sorted(page_numbers))

# dependencies: atlassian-python-api, lxml
class ConfluencePageLoader:
    def __init__(self):
        self.base_url = os.environ.get('CONFLUENCE_WIKI_URL')
        self.username = os.environ.get('ATTLASIAN_USER_EMAIL')
        self.api_key = os.environ.get('ATTLASIAN_API_TOKEN')

    def is_target_path(self, url):
        return url.startswith(self.base_url)

    def load(self, path, options={}):
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

    def load(self, path, options={}):
        loader = WebBaseLoader(path)
        doc = loader.load()[0]
        if not doc.metadata.get('title'):
            doc.metadata['title'] = 'Untitled'
        return doc

# dependencies: langchain_community, youtube-transcript-api, pytube
class YoutubeLoader:
    def __init__(self):
        pass

    def is_target_path(self, path):
        from urllib.parse import urlparse
        parsed_url = urlparse(path)
        return parsed_url.netloc == 'www.youtube.com'

    def load(self, path, options={}):
        loader = YtLoader.from_youtube_url(
            path,
            # ref. https://github.com/pytube/pytube/issues/1589
            # add_video_info=True,
            language=['ja', 'en']
        )
        doc = loader.load()[0]
        if not doc.metadata.get('title'):
            doc.metadata['title'] = 'Untitled'
        return doc

class PDFTextLoader:
    def __init__(self):
        pass

    def is_target_path(self, path):
        return path.endswith('.pdf') and os.path.isfile(path)

    def load(self, path, options={}):
        from pdfminer.high_level import extract_text
        from pdfminer.layout import LAParams

        page_numbers = options.get('pages', None)
        # LAParams: parameters for layout analysis
        # Ref. https://pdfminersix.readthedocs.io/en/latest/reference/composable.html
        laparams = LAParams()
        # Ref. https://pdfminersix.readthedocs.io/en/latest/reference/highlevel.html#extract-text
        text = extract_text(path, laparams=laparams, page_numbers=page_numbers)
        return Document(page_content=text)

if __name__ == '__main__':
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--output-file', '-o', required=False, help='Output file')
    parser.add_argument('--options', required=False, help='Loader options')
    parser.add_argument('source', help='The source URL or path')

    args = parser.parse_args()
    options = LoaderOptionParser().parse(args.options) if args.options else {}

    loaders = [ConfluencePageLoader(), YoutubeLoader(), WebPageLoader(), PDFTextLoader()]
    for loader in loaders:
        if loader.is_target_path(args.source):
            doc = loader.load(args.source, options)
            break

    if not doc:
        raise ValueError('No loader found for the given URL or path')

    def print_doc(doc, f):
        if doc.metadata.get('title'):
            f.write(f'# {doc.metadata["title"]}\n\n')
        f.write(doc.page_content)

    if args.output_file:
        with open(args.output_file, 'w') as f:
            print_doc(doc, f)

    print_doc(doc, sys.stdout)

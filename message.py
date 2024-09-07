import os
import re
import base64
from io import BytesIO

from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.document_loaders import WebBaseLoader

import pdf2image


def encode_image(image_path: str) -> str:
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class MessageExpander:
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif'}
    IMPORTER_PATTERN = r'\{(.*?)\}'

    def __init__(self):
        self.content_list = []
        self.loaders = [WebPageLoader(), ConfluencePageLoader()]

    def expand_message(self, message: str) -> HumanMessage:
        """Expand the message by expanding file imports.

        Args:
            message: The message string that may contain file imports.

        Returns:
            A HumanMessage object with the file imports expanded.
        """
        segments = self._separate_message(message)
        for segment in segments:
            if match := re.match(self.IMPORTER_PATTERN, segment):
                path, options = self._parse_placeholder(match.group(1))
                title = options.get('title')

                handled_by_loader = False
                for loader in self.loaders:
                    if loader.is_target_path(path):
                        content = loader.load(path)
                        self._append_text_content(content)
                        handled_by_loader = True
                        break
                if handled_by_loader:
                    continue
                elif os.path.exists(path):
                    if not title:
                        title = os.path.basename(path)
                    file_extension = os.path.splitext(path)[1].lower()
                    if file_extension in self.IMAGE_EXTENSIONS:
                        self._append_image_content(title, path)
                    elif file_extension == '.pdf':
                        self._append_pdf_content(path)
                    else:
                        self._append_file_content(title, path)
                else:
                    self._append_text_content(segment)
                continue
            self._append_text_content(segment)

        if len(self.content_list) == 1 and self.content_list[0].get('type') == 'text':
            return HumanMessage(content=self.content_list[0]['text'])
        return HumanMessage(content=self.content_list)

    @classmethod
    def _separate_message(cls, message: str):
        segments = []
        start = 0

        for match in re.finditer(cls.IMPORTER_PATTERN, message):
            # Add the text before the matched pattern as a segment
            if match.start() > start:
                segments.append(message[start:match.start()])

            # Add the matched text as a segment
            segments.append(match.group(0))

            start = match.end()

        # Add the remaining text after the last matched pattern as a segment
        if start < len(message):
            segments.append(message[start:])

        return segments

    @staticmethod
    def _parse_placeholder(placeholder_str: str) -> dict:
        if '|' not in placeholder_str:
            return placeholder_str, {}
        path, options_str = placeholder_str.split('|')
        options = {}
        for option in options_str.split(';'):
            key, value = option.split('=', 1)
            options[key.strip()] = value.strip()
        return path.strip(), options

    def _append_text_content(self, content):
        if self.content_list and self.content_list[-1].get('type') == 'text':
            self.content_list[-1]['text'] += '\n' + content
        else:
            self.content_list.append({'type': 'text', 'text': content})

    def _append_file_content(self, title, path):
        with open(path, 'r') as f:
            file_content = f.read()
        if title:
            file_content = f'### {title} ###\n```\n{file_content}\n```\n'
        self._append_text_content(file_content)

    def _append_image_content(self, title, path):
        if title:
            title_text = f'### {title} ###\n'
            self._append_text_content(title_text)
        base64_image = encode_image(path)
        image_type = os.path.splitext(path)[1].lstrip('.')
        self.content_list.append({
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/{image_type};base64,{base64_image}'
            }
        })

    def _append_pdf_content(self, pdf_path: str):
        for image in pdf2image.convert_from_path(pdf_path):
            buffered = BytesIO()
            image.save(buffered, format='jpeg')
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            self.content_list.append({
                'type': 'image_url',
                'image_url': {
                    'url': f'data:image/jpg;base64,{img_str}'
                }
            })

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
        title = doc.metadata['title']
        return f'### {title} ###\n' + doc.page_content

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
        title = doc.metadata.get('title', 'Untitled')
        return f'### {title} ###\n' + doc.page_content

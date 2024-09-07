import os
import re
import base64
from io import BytesIO

from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import ConfluenceLoader

import pdf2image


def encode_image(image_path: str) -> str:
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class MessageExpander:
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif'}
    IMPORTER_PATTERN = r'\[(.*)\]\((.+)\)'

    def __init__(self):
        self.content_list = []
        self.confluence_loader = ConfluencePageLoader()

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
                name, path = match.groups()
                name = name.strip()
                path = path.strip()
                if self.confluence_loader.is_confluence_url(path):
                    content = self.confluence_loader.load(path)
                    self._append_text_content(content)
                elif os.path.exists(path):
                    if not name:
                        name = os.path.basename(path)
                    file_extension = os.path.splitext(path)[1].lower()
                    if file_extension in self.IMAGE_EXTENSIONS:
                        self._append_image_content(name, path)
                    elif file_extension == '.pdf':
                        self._append_pdf_content(path)
                    else:
                        self._append_file_content(name, path)
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

    def _append_text_content(self, content):
        if self.content_list and self.content_list[-1].get('type') == 'text':
            self.content_list[-1]['text'] += '\n' + content
        else:
            self.content_list.append({'type': 'text', 'text': content})

    def _append_file_content(self, name, path):
        with open(path, 'r') as f:
            file_content = f.read()
        if name:
            file_content = f'### {name} ###\n```\n{file_content}\n```\n'
        self._append_text_content(file_content)

    def _append_image_content(self, name, path):
        if name:
            title = f'### {name} ###\n'
            self._append_text_content(title)
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

class ConfluencePageLoader:
    def __init__(self):
        self.url = os.environ.get('CONFLUENCE_WIKI_URL')
        self.username = os.environ.get('ATTLASIAN_USER_EMAIL')
        self.api_key = os.environ.get('ATTLASIAN_API_TOKEN')

    def is_confluence_url(self, url):
        return url.startswith(self.url)

    def load(self, url):
        match = re.match(rf'{self.url}/(?:\S+/)+pages/(\d+)/.*', url)
        if not match:
            raise ValueError(f'Invalid Confluence URL: {url}')
        page_id = match.group(1)
        loader = ConfluenceLoader(
            url=self.url,
            username=self.username,
            api_key=self.api_key,
            page_ids=[page_id],
            keep_markdown_format=True
        )
        doc = loader.load()[0]
        title = doc.metadata['title']
        return f'### {title} ###\n' + doc.page_content

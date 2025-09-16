import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_text_splitters import NLTKTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from common.log import logger


class TextExtractor:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.text_splitter = NLTKTextSplitter(chunk_size=256, chunk_overlap=0)

    def _store_chunk(self, documents):
        for i, doc in enumerate(documents):
            file_name = f"chunk_{i}"
            file_path = os.path.join(self.output_dir, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(doc.page_content)
            logger.info(f"Stored chunk {i} at {file_path}")

    def extract(self, file_path):
        suffix = os.path.splitext(file_path)[-1].lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            documents = self.text_splitter.split_documents(documents)
            self._store_chunk(documents)
        elif suffix == ".docx":
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            documents = self.text_splitter.split_documents(documents)
            self._store_chunk(documents)
        else:
            # 先不考虑其他文件类型，后续可以扩展
            raise ValueError(f"Unsupported file type: {suffix}")

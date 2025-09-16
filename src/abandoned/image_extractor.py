import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # PyMuPDF
from common.log import logger
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT


class ImageExtractor:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _handle_pdf(self, file_path):
        doc = fitz.open(file_path)
        image_paths = []

        for page_index in range(len(doc)):
            page = doc[page_index]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_path = os.path.join(
                    self.output_dir,
                    f"{os.path.splitext(os.path.basename(file_path))[0]}_p{page_index+1}_i{img_index+1}.{image_ext}",
                )
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                image_paths.append(image_path)
        logger.info(f"Extracted {len(image_paths)} images from {file_path}")

    def _handle_docx(self, file_path):
        doc = Document(file_path)
        rels = doc.part._rels
        image_paths = []

        for rel in rels:
            rel = rels[rel]
            if rel.reltype == RT.IMAGE:
                image_data = rel.target_part.blob
                image_ext = os.path.splitext(rel.target_part.partname)[
                    1
                ]  # 如 .png, .jpeg
                image_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_{rel.rId}{image_ext}"
                image_path = os.path.join(self.output_dir, image_filename)
                with open(image_path, "wb") as f:
                    f.write(image_data)
                image_paths.append(image_path)
        logger.info(f"Extracted {len(image_paths)} images from {file_path}")

    def extract(self, file_path):
        suffix = os.path.splitext(file_path)[-1].lower()
        if suffix == ".pdf":
            self._handle_pdf(file_path)
        elif suffix == ".docx":
            self._handle_docx(file_path)
        else:
            # 先不考虑其他文件类型，后续可以扩展
            raise ValueError(f"Unsupported file type: {suffix}")

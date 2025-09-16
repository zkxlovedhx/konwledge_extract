
from dataclasses import dataclass
import os
from pathlib import Path
from langchain_core.messages import HumanMessage

from src.utils.img_uploader import img2url

@dataclass
class TextItem:#不变对象
    type: str
    text: str
    page_idx:int
    text_level: int = 0 
    
    def __str__(self):
        return self.text

@dataclass
class ImageItem:#不变对象
    type: str
    img_path: str
    img_caption: list[str]
    img_footnote: list[str]
    page_idx:int
    text_level: int = 0  
    
    def __str__(self):
        return f'[image]{self.img_path}[/image]'

Item=TextItem|ImageItem

def convert_to_item_list(data, root_path:Path) -> list[Item]:
    items = []
    for item in data:
        if item.get('type') == 'text':
            filtered_item = {k: v for k, v in item.items() if k != 'bbox'}
            items.append(TextItem(**filtered_item))  # 使用过滤后的字典创建实例
        elif item.get('type') == 'image' and item['img_path']:
            items.append(ImageItem(
                type=item['type'], 
                img_path=os.path.normpath(root_path / item['img_path']),
                img_caption=item.get('img_caption', []),
                img_footnote=item.get('img_footnote', []),
                page_idx=item.get('page_idx', 0),
                text_level=item.get('text_level', 0)
            ))
    return items
class Chunk:
    
    def __init__(self, items:list[Item]):
        self.items= items
    
    def __str__(self):
        return "\n".join(map(str, self.items))

    def __len__(self):
        return sum(len(str(item)) for item in self.items)
        
    def to_message(self)->HumanMessage:
        """
        将 Chunk 转换为langchain消息格式
        """
        content=[]
        for item in self.items:
            if isinstance(item, TextItem):
                content.append({"type": "text", "text": str(item)})  # 使用 str(item) 获取文本内容
            elif isinstance(item, ImageItem):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": img2url(item.img_path),
                    }})
        return HumanMessage(content)

class Spliter:
    """
    """
    @classmethod
    def split(cls,items:list[Item],max_chunk_size:int=1000)->list[Chunk]:
        """按照窗口大小，使用滑动窗口合并chunk"""
        def _merge(chunks:list[Chunk])->list[Chunk]:
            """
            合并相邻的chunk
            """
            merged_chunks:list[Chunk] = []
            current_size=0
            current_chunk:list[Item] = []
            for chunk in chunks:
                if current_size + len(chunk) <= max_chunk_size:
                    current_chunk.extend(chunk.items)
                    current_size += len(chunk)
                else:
                    if current_chunk:
                        merged_chunks.append(Chunk(current_chunk))
                    current_chunk = chunk.items
                    current_size = len(chunk)
            if current_chunk:
                merged_chunks.append(Chunk(current_chunk))
            return merged_chunks

        chunks=cls._pre_split(items)
        chunks=_merge(chunks)
        return chunks

    def _pre_split(items:list[Item])->list[Chunk]:
        """
        按照TextItem的text_level 等级阈值 进行分割
        """
        threshold = 1
        chunks:list[Chunk] = []
        current_chunk: list[Item] = []
        last_text_level = None
        
        for item in items:
            if item.text_level < threshold or item.text_level==last_text_level:
                current_chunk.append(item)
                last_text_level = item.text_level
            else:
                if current_chunk:
                    chunks.append(Chunk(current_chunk))
                current_chunk = [item]
                last_text_level = item.text_level
        if current_chunk:
            chunks.append(Chunk(current_chunk))
        
        return chunks


import configparser
import json
import os
from pathlib import Path
import sys

from langchain_openai import ChatOpenAI
from py2neo import Graph
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.graph_builder import GraphBuilder
from src.core.knowledge_extractor import KnowledgeExtractor,ExtractResult
from src.core.schema_generator import Schema
from src.core.chunk import Spliter, convert_to_item_list
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
config_path = os.path.join(project_root, "config", "config.ini")
config = configparser.ConfigParser()
config.read(config_path)

model = ChatOpenAI(
    model=config.get("LLM", "model"),
    base_url=config.get("LLM", "base_url"),
    api_key=config.get("LLM", "api_key"),
    temperature=0,
)
graph = Graph("neo4j://127.0.0.1:7687", auth=("neo4j", "12345678"),name='aesthetics')

import argparse

async def handle(json_paths: list[Path], prompt_path: Path, schema_path: Path, save_dir: Path, clear: bool):
    save_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for json_path in json_paths:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        items=convert_to_item_list(data, json_path.parent)
        new_chunks=Spliter.split(items)
        chunks.extend(new_chunks)

    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt=f.read()
    schema = Schema.model_validate(json.load(open(schema_path, 'r', encoding='utf-8')))
    extractor = KnowledgeExtractor(prompt,model,schema)
    assert len(chunks) > 0, "No chunks to process. Please check the input JSON files."
    assert model is not None, "Model is not initialized."
    assert len(prompt) > 0, "Prompt is empty. Please check the prompt file."
    assert schema is not None, "Schema is not initialized. Please check the schema file."
    print("Starting knowledge extraction...")
    # exit()
    extract_result: ExtractResult = await extractor.execute(chunks)

    # 抽取数据源保存
    with open(save_dir / 'src_paths.json', 'w', encoding='utf-8') as f:
        json.dump([str(path) for path in json_paths], f, ensure_ascii=False, indent=4)
    with open(save_dir / 'prompt.txt', 'w', encoding='utf-8') as f:
        f.write(prompt)
    with open(save_dir / 'schema.json', 'w', encoding='utf-8') as f:
        json.dump(schema.model_dump(), f, ensure_ascii=False, indent=4)
    with open(save_dir / 'extract_result.json', 'w', encoding='utf-8') as f:
        json.dump(extract_result.model_dump(), f, ensure_ascii=False, indent=4)
    # 可选清空图数据库
    if clear:
        print("Clearing existing Neo4j graph...")
        graph.delete_all()
    graph_builder = GraphBuilder()
    knowledge_graph = graph_builder.build(extract_result)
    knowledge_graph.upload_neo4j(graph)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge extraction with required clear parameter (true|false) to control Neo4j data clearing.")
    parser.add_argument('--clear', action='store_true', help='Clear existing Neo4j data before upload')
    args = parser.parse_args()
    # 根据当前时间生成带时间戳的保存路径
    from datetime import datetime
    knowledge_extract_result_dir = config.get("Output", "knowledge_extract_result_dir")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(knowledge_extract_result_dir)/timestamp
    
    root_path = Path("D:/project/automobile_style_design/data/handled_document/")
    json_paths: list[Path] = []
    for file in root_path.rglob('*_content_list.json'):
        json_paths.append(file)
    prompt_path = Path(__file__).parent / '..' / 'src' / 'prompt' / 'knowledge_extractor_with_schema_prompt'
    schema_path = Path(__file__).parent / '..' / 'src' / 'schema' / '汽车造型设计美学概念schema.json'

    import asyncio
    asyncio.run(handle(json_paths, prompt_path, schema_path,save_dir, args.clear))
    print("Knowledge extraction and graph upload completed.")
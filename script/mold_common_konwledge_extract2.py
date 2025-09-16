import configparser
import json
import os
from pathlib import Path
import sys
from py2neo import Graph
from langchain_openai import ChatOpenAI
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import random
from src.core.graph_builder import GraphBuilder, KnowledgeGraph
from src.core.schema_generator import Schema
from src.core.chunk import Spliter, convert_to_item_list
from src.core.knowledge_extractor import KnowledgeExtractor, ExtractResult, Entity, Relation
from src.common.log import logger

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
config_path = os.path.join(project_root, "config", "config.ini")
graph = Graph("neo4j+s://685505c5.databases.neo4j.io",auth=("neo4j", "vEOl6QJyud536OpD-ySqR6OOs8mdjLdWj4XhYsSsC8o"))

# 读取 config.ini
config = configparser.ConfigParser()
config.read(config_path)

# 初始化 LLM
model = ChatOpenAI(
    model=config.get("LLM", "model"),
    base_url=config.get("LLM", "base_url"),
    api_key=config.get("LLM", "api_key"),
    temperature=0,
)

# 初始化 embedding 模型
embed_model = SentenceTransformer(config.get("Embedding", "model_path"), device='cpu')
embedding_dim = embed_model.get_sentence_embedding_dimension()
# FAISS 索引和节点映射
faiss_index = None
id_to_node = {}
faiss_dir = os.path.join(project_root, "output", "faiss_index")
faiss_index_file = os.path.join(faiss_dir, "nodes.index")
mapping_file = os.path.join(faiss_dir, "id_to_node.pkl")

# 加载实体匹配Prompt
def load_matching_prompt(prompt_path: str) -> str:
    """从文件加载实体匹配的Prompt模板"""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

# 配置实体匹配Prompt路径
matching_prompt_path = os.path.join(project_root, 'src', 'prompt', 'prompt')  
matching_prompt_template = load_matching_prompt(matching_prompt_path)


def save_faiss():
    os.makedirs(faiss_dir, exist_ok=True)
    faiss.write_index(faiss_index, faiss_index_file)
    with open(mapping_file, "wb") as f:
        pickle.dump(id_to_node, f)
    print(f"FAISS index saved to {faiss_index_file}, total nodes: {len(id_to_node)}")


def load_faiss():
    global faiss_index, id_to_node
    if os.path.exists(faiss_index_file) and os.path.exists(mapping_file):
        try:
            faiss_index = faiss.read_index(faiss_index_file)
            with open(mapping_file, "rb") as f:
                id_to_node = pickle.load(f)

            # 确保所有实体都有必要的属性
            for node_id, entity in id_to_node.items():
                if not hasattr(entity, 'aliases'):
                    entity.aliases = []
                    logger.info(f"升级旧实体 {node_id}，添加aliases属性")
                if not hasattr(entity, 'properties'):
                    entity.properties = {}
                    logger.info(f"升级旧实体 {node_id}，添加properties属性")
                    
            print(f"FAISS index loaded from {faiss_index_file}, {len(id_to_node)} nodes.")
        except Exception as e:
            print(f"加载FAISS索引失败，创建新的索引: {str(e)}")
            initialize_faiss()
    else:
        initialize_faiss()


def initialize_faiss():
    """初始化新的FAISS索引"""
    global faiss_index, id_to_node
    faiss_index = faiss.IndexFlatIP(embedding_dim)
    id_to_node = {}
    print("Initialized new FAISS index.")


def node_to_text(node: dict) -> str:
    # 兼容不同字段名
    name = node.get("name") or node.get("entityName")

    if not name and "properties" in node:
        for key in node["properties"]:
            if "名称" in key or "Name" in key:
                name = node["properties"][key]
                break

    if not name:
        name = "未知节点"
    
    # 拼接别名信息
    aliases = node.get("aliases", [])
    alias_text = f"（别名：{','.join(aliases)}）" if aliases else ""
    
    props = " ".join([f"{k}:{v}" for k, v in node.get("properties", {}).items()])
    return f"{name}{alias_text} {props}"


def add_node_to_faiss(node: dict):
    node.setdefault('aliases', [])
    node.setdefault('properties', {})
    text = node_to_text(node)
    emb = embed_model.encode(text, device="cpu", normalize_embeddings=True).astype(np.float32)
    new_id = len(id_to_node)
    
    # 创建新的Entity对象
    new_entity = Entity(
        type=node.get('type', ''),
        entityName=node.get('entityName') or node.get('name', '未知名称'),
        properties=node['properties'].copy(),  # 使用副本避免引用问题
        aliases=node['aliases'].copy()  # 使用副本避免引用问题
    )
    
    id_to_node[new_id] = new_entity
    faiss_index.add(np.array([emb]))
    
    logger.info(f"新增节点 ID {new_id}: {new_entity.entityName}")
    return new_id


def check_and_merge_node(new_node: dict, merge_decisions_log: list):
    # 确保新节点有必要的键
    new_node.setdefault('type', '')
    new_node.setdefault('entityName', '')
    new_node.setdefault('name', '')
    new_node.setdefault('properties', {})
    new_node.setdefault('aliases', [])
    
    # 提取新节点关键信息
    new_node_type = new_node['type']
    new_node_name = new_node['entityName'] or new_node['name'] or "未知名称"
    
    # 提取核心属性
    core_prop_keys = ["ID", "编号", "代码", "Unique_ID"]
    new_node_core_props = {
        k: v for k, v in new_node['properties'].items()
        if any(keyword in k for keyword in core_prop_keys)
    }
    new_node_all_props = new_node['properties']

    # 生成向量并检索
    text = node_to_text(new_node)
    new_emb = embed_model.encode(text, normalize_embeddings=True).astype(np.float32)

    # 检索相似节点
    k = config.getint("FAISS", "top_k")
    candidates = []
    if faiss_index.ntotal > 0:
        D, I = faiss_index.search(np.array([new_emb]), k)
        candidates = [(i, id_to_node[i]) for i in I[0] if i in id_to_node]

    # 如果没有候选节点，直接新增
    if not candidates:
        new_id = add_node_to_faiss(new_node)
        from datetime import datetime
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'decision': 'new (no candidates)',
            'new_node_name': new_node_name,
            'new_node_type': new_node_type,
            'property_changes': []
        }
        merge_decisions_log.append(log_entry)
        return "new", new_id

    # 格式化候选节点信息
    candidates_formatted = []
    for candidate_id, candidate in candidates:
        # 确保候选节点属性正确访问
        candidate_core_props = {
            k: v for k, v in candidate.properties.items()
            if any(keyword in k for keyword in core_prop_keys)
        }
        candidates_formatted.append(
            f"候选节点ID: {candidate_id}\n"
            f"类型: {candidate.type}\n"
            f"名称: {candidate.entityName}\n"
            f"别名: {candidate.aliases}\n"
            f"核心属性: {candidate_core_props}\n"
            f"全部属性: {candidate.properties}"
        )

    # 构建提示词
    prompt = matching_prompt_template.format(
        new_node_type=new_node_type,
        new_node_name=new_node_name,
        new_node_core_props=new_node_core_props,
        new_node_all_props=new_node_all_props,
        candidates_formatted="\n\n".join(candidates_formatted)
    )

    # 调用LLM判断
    decision = model.invoke(prompt).content.strip()
    
    # 记录决策信息到日志列表
    from datetime import datetime
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'decision': decision,
        'new_node_name': new_node_name,
        'new_node_type': new_node_type,
        'property_changes': []
    }
    
    logger.info(f"LLM融合决策: {decision} | 新节点: {new_node_name}")

    if decision.startswith("merge"):
        try:
            target_id = int(decision.split(":")[1].strip())
            if target_id in id_to_node:
                target_node = id_to_node[target_id]
                new_entity_name = new_node['entityName'] or new_node['name']

                # 1. 为存在的Node添加别名
                if new_entity_name and new_entity_name != target_node.entityName:
                    if new_entity_name not in target_node.aliases:
                        target_node.aliases.append(new_entity_name)
                        logger.info(f"节点 {target_id} 新增别名: {new_entity_name}")
                        log_entry['property_changes'].append(f"新增别名: {new_entity_name}")

                # 2. 属性扩充与融合
                new_properties = new_node['properties']
                for key, value in new_properties.items():
                    if key not in target_node.properties:
                        target_node.properties[key] = value
                        logger.info(f"节点 {target_id} 新增属性: {key}={value}")
                        log_entry['property_changes'].append(f"新增属性: {key}={value}")
                    else:
                        existing_value = target_node.properties[key]
                        if existing_value != value:
                            if not isinstance(existing_value, list):
                                target_node.properties[key] = [existing_value]
                            if value not in target_node.properties[key]:
                                target_node.properties[key].append(value)
                                logger.info(f"节点 {target_id} 融合属性 {key}: 新增值 {value}")
                                log_entry['property_changes'].append(f"融合属性 {key}: 新增值 {value}")

                # 更新FAISS向量
                merged_text = node_to_text({
                    "type": target_node.type,
                    "entityName": target_node.entityName,
                    "properties": target_node.properties,
                    "aliases": target_node.aliases
                })
                merged_emb = embed_model.encode(merged_text, normalize_embeddings=True).astype(np.float32)
                
                # 更新FAISS索引
                faiss_index.remove_ids(np.array([target_id], dtype=np.int64))
                faiss_index.add(np.array([merged_emb]))

                logger.info(f"节点融合：{new_entity_name} 合并到 ID {target_id}")
                merge_decisions_log.append(log_entry)
                return "merge", target_id
        except Exception as e:
            logger.warning(f"融合处理失败: {str(e)}")
            log_entry['error'] = str(e)
            merge_decisions_log.append(log_entry)
        # 失败时默认新增
        return "new", add_node_to_faiss(new_node)
    else:
        merge_decisions_log.append(log_entry)
        return "new", add_node_to_faiss(new_node)


import argparse
import asyncio


async def handle(json_paths: list[Path], prompt_path: Path, schema_path: Path, save_dir: Path, clear: bool):
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载/初始化 FAISS - 现在会保留之前的内容
    load_faiss()

    # 数据准备
    chunks = []
    for json_path in json_paths:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        items = convert_to_item_list(data, json_path.parent)
        new_chunks = Spliter.split(items)
        chunks.extend(new_chunks)

    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read()
    schema = Schema.model_validate(json.load(open(schema_path, 'r', encoding='utf-8')))

    extractor = KnowledgeExtractor(prompt, model, schema)
    assert len(chunks) > 0, "No chunks to process. Please check the input JSON files."
    print(f"Starting knowledge extraction with {len(chunks)} chunks...")

    extract_result: ExtractResult = await extractor.execute(chunks)

    # 保存中间结果
    with open(save_dir / 'src_paths.json', 'w', encoding='utf-8') as f:
        json.dump([str(path) for path in json_paths], f, ensure_ascii=False, indent=4)
    with open(save_dir / 'prompt.txt', 'w', encoding='utf-8') as f:
        f.write(prompt)
    with open(save_dir / 'schema.json', 'w', encoding='utf-8') as f:
        json.dump(schema.model_dump(), f, ensure_ascii=False, indent=4)
    with open(save_dir / 'extract_result.json', 'w', encoding='utf-8') as f:
        json.dump(extract_result.model_dump(), f, ensure_ascii=False, indent=4)

    # 创建融合决策日志文件
    merge_decisions_log = []

    # 可选清空图数据库 - 注意：这只会清空Neo4j，不会清空FAISS
    if clear:
        print("Clearing existing Neo4j graph...")
        graph.delete_all()

    merged_edges = []
    for edge in extract_result.edges:
        # 融合头实体
        action_head, head_id = check_and_merge_node(edge.head.model_dump(), merge_decisions_log)
        edge.head = id_to_node[head_id]
        # 融合尾实体
        action_tail, tail_id = check_and_merge_node(edge.tail.model_dump(), merge_decisions_log)
        edge.tail = id_to_node[tail_id]
        merged_edges.append(edge)

    extract_result.edges = merged_edges

    # 保存更新后的 FAISS - 现在会保留所有历史节点
    save_faiss()

    # 保存融合决策日志到文件
    with open(save_dir / 'merge_decisions.log', 'w', encoding='utf-8') as f:
        for entry in merge_decisions_log:
            f.write(f"{entry['timestamp']} | {entry['decision']} | 新节点: {entry['new_node_name']}\n")
            if 'property_changes' in entry and entry['property_changes']:
                for change in entry['property_changes']:
                    f.write(f"  -> {change}\n")
            if 'error' in entry:
                f.write(f"  -> 错误: {entry['error']}\n")
            f.write("\n")

    # 构建并上传知识图谱到Neo4j
    graph_builder = GraphBuilder()
    knowledge_graph = graph_builder.build(extract_result)
    knowledge_graph.upload_neo4j(graph)
    
    print(f"处理完成！当前FAISS索引中共有 {len(id_to_node)} 个节点")


def run():
    parser = argparse.ArgumentParser(description="Knowledge extraction with clear parameter to control Neo4j data clearing.")
    parser.add_argument('--clear', action='store_true', default=True, help='Clear existing Neo4j data before upload')
    args = parser.parse_args()

    from datetime import datetime
    knowledge_extract_result_dir = config.get("Output", "knowledge_extract_result_dir")
    knowledge_extract_result_dir = os.path.join(project_root, knowledge_extract_result_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(knowledge_extract_result_dir) / timestamp

    # 扫描所有 *_content_list.json
    output_docs_path = Path(os.path.join(project_root, "output", "docs", "最新模具标准应用手册"))
    json_paths: list[Path] = []
    for file in output_docs_path.rglob('*_content_list.json'):
        json_paths.append(file)

    prompt_path = os.path.join(project_root, 'src', 'prompt', 'mold_common_knowledge_prompt')
    schema_path = os.path.join(project_root, 'src', 'schema', 'schema_def.json')

    asyncio.run(handle(json_paths, prompt_path, schema_path, save_dir, args.clear))
    print("Knowledge extraction and graph upload completed.")


if __name__ == "__main__":
    run()
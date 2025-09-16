import os
import sys

from langchain_openai import ChatOpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.log import logger
import configparser
from src.common.schema_loader import SchemaLoader
from src.core.knowledge_extractor import KnowledgeExtractor

# 读取配置代码
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
config_path = os.path.join(project_root, "config", "config.ini")
config = configparser.ConfigParser()
config.read(config_path)


def collect_content_json_paths(data_dir):
    result = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith("_content_list.json"):
                file_path = os.path.join(root, file)
                result.append(file_path)
    return result


def run():
    model = ChatOpenAI(
        model=config.get("LLM", "model"),
        base_url=config.get("LLM", "base_url"),
        api_key=config.get("LLM", "api_key"),
        temperature=0,
    )
    # 预处理后的文档存放目录以及 Schema 文件路径
    handled_document_dir = config.get("DocumentHandler", "handled_document_dir")
    data_dir = os.path.join(project_root, handled_document_dir)
    schema_file_path = os.path.join(
        project_root, config.get("Schema", "schema_file_path")
    )
    # 抽取结果文件
    result_file_path = config.get("Output", "knowledge_extract_result_file")
    try:
        logger.info("Script2-knowledge-extract")
        data_file_path_list = collect_content_json_paths(data_dir)
        extractor = KnowledgeExtractor(data_file_path_list, schema_file_path, model)
        extractor.execute(result_file_path)
        logger.info("Knowledge extraction completed successfully")
    except Exception as e:
        logger.error("An error occurred while running the script", exc_info=True)
        raise e
    else:
        logger.info("Script2-knowledge-extract executed successfully")


"""
脚本 2-利用大模型进行知识抽取
"""
if __name__ == "__main__":
    run()

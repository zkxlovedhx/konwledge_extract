import os
import sys

from langchain_openai import ChatOpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.log import logger
from llm.template import PromptContainer
import json
from src.core.tool import load_data_to_text
from langchain_core.output_parsers.json import JsonOutputParser


class KnowledgeExtractor:

    def __init__(
        self, data_file_path_list: list[str], schema_file_path: str, model: ChatOpenAI
    ):
        """
        初始化数据加载器。

        :param data_file_path_list: 数据文件路径列表
        :param schema_file_path: Schema 文件路径
        """
        self.data_file_path_list = data_file_path_list

        # 加载 schema 到 LLM 模板
        self.prompt_container = PromptContainer()
        self.prompt_container.set_system_message_from_schema(schema_file_path)
        self.model = model
        parser = JsonOutputParser()
        self.chain = self.model | parser

    def execute(self, result_file_path: str):
        entities = []
        relations = []
        for data_path in self.data_file_path_list:
            text = load_data_to_text(data_path, self.model)
            # TODO 后续需要先做文本分片, 再输入给大模型

            # 使用 LLM 进行知识抽取
            logger.info(f"Processing data file: {data_path}")
            prompt = self.prompt_container.format(text)
            result = self.chain.invoke(prompt)
            logger.info(f"LLM response: {result}, data file path: {data_path}")
            entities.extend(result.get("entities", []))
            relations.extend(result.get("relations", []))

        # 将结果保存到文件
        merged_output = {"entities": entities, "relations": relations}
        with open(result_file_path, "w", encoding="utf-8") as f:
            json.dump(merged_output, f, ensure_ascii=False, indent=4)

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.messages.system import SystemMessage
from common.log import logger
import json


def build_ontology_text(schema: dict) -> str:
    """
    构建本体文本，用于填充到 LLM 提示词中
    """
    entity_descriptions = []
    for entity in schema.get("schema_definitions", []):
        attrs = "\n".join(
            [
                f"    - 属性名称：{attr['name']}（类型：{attr['type']}，描述：{attr['description']}）"
                for attr in entity.get("attributes", [])
            ]
        )
        entity_block = (
            f"- 实体类型：{entity['type']}\n"
            f"  描述：{entity['description']}\n"
            f"  属性列表：\n{attrs}"
        )
        entity_descriptions.append(entity_block)

    relation_descriptions = []
    for relation in schema.get("relation_types", []):
        relation_block = (
            f"- 关系类型：{relation['type']}\n"
            f"  描述：{relation['description']}\n"
            f"  头实体类型：{relation['head']}\n"
            f"  尾实体类型：{relation['tail']}"
        )
        relation_descriptions.append(relation_block)

    ontology_text = (
        "【实体定义】\n"
        + "\n\n".join(entity_descriptions)
        + "\n\n【关系定义】\n"
        + "\n\n".join(relation_descriptions)
    )
    return ontology_text


class PromptContainer:
    def __init__(self):
        self.system_message = ""

    def set_system_message_from_schema(self, schema_file_path: str):
        """
        读取一个 Schema 文件路径, 将知识图谱的本体定义填充到 LLM 系统提示词当中
        """
        logger.info(f"Loading schema file from: {schema_file_path}")
        with open(schema_file_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        ontology_text = build_ontology_text(schema)
        # 构造系统提示词
        self.system_message = f"""你是汽车造型设计领域的知识图谱构建专家，具备丰富的本体建模与信息抽取经验。当前，你的任务是依据预定义的知识图谱本体 schema，从输入文本中精准抽取出结构化的知识内容。

        请严格按照以下三个阶段完成知识抽取：

        1. 命名实体识别（NER）：
        - 识别文本中出现的**实体实例**及其对应的**属性值**；
        - 每个实体必须标注其类型，属性值应尽可能完整、精确；
        - 若属性在文本中未明确提及，则可省略该字段。

        2. 实体消融（Entity Linking / Canonicalization）：
        - 若文本中提及的实体在语义上指代相同对象（如“比亚迪”、“BYD”），应统一为一个标准名称；
        - 属性冲突时，优先保留更具体、更明确的描述。

        3. 关系抽取（Relation Extraction）：
        - 判断文本中是否存在两个实体之间的显式或隐式关系；
        - 仅抽取与 schema 中定义的关系类型匹配的关系；
        - 保证关系三元组的头实体、尾实体及其类型与本体一致。

        以下是你需要遵循的知识图谱本体定义：

        {ontology_text}

        ---

        【输出要求】

        请以严格的 JSON 格式返回抽取结果，结构如下：

        {{
        "entities": [
            {{
            "type": "<实体类型，必须与本体匹配>",
            "name": "<实体名称，可作为唯一标识>",
            "attributes": {{
                "<属性名称>": "<属性值，类型需与本体一致>"
            }}
            }},
            ...
        ],
        "relations": [
            {{
            "type": "<关系类型，必须与本体匹配>",
            "head": "<头实体名称>",
            "tail": "<尾实体名称>"
            }},
            ...
        ]
        }}

        注意事项：
        - 所有抽取内容必须严格依据本体定义进行，不得引入未定义的类型、属性或关系；
        - 所有属性值需为字符串、整数或日期等基本类型，不使用嵌套结构；
        - 请勿输出任何解释说明、多余文本或提示词，仅返回 JSON；
        - 若未能从文本中识别出任何实体或关系，返回：
        {{
            "entities": [],
            "relations": []
        }}

        你的目标是为后续的知识图谱构建提供高质量、结构化、规范化的知识片段。请保持语言精确，结构清晰，抽取结果专业严谨。
        """

    def format(self, text_to_extract: str):
        template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.system_message),
                ("human", "请从以下给定的文本中进行知识抽取：\n{text}"),
            ]
        )
        return template.format_messages(text=text_to_extract)

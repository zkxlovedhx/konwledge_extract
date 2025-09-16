import asyncio, json
from tqdm.asyncio import tqdm_asyncio
from langchain_openai import ChatOpenAI
from src.core.chunk import Chunk
from langchain_core.messages import SystemMessage, BaseMessage
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from tqdm import tqdm

from src.core.entity_discriminator import EntityDiscriminator
from src.common.log import logger


class RelationType(BaseModel):
    type: str = Field(..., description="The type of the relation")
    description: str = Field(..., description="A description of the relation")
    properties: dict[str, str] = Field(
        default_factory=dict, description="A dictionary of properties for the relation"
    )
    end_node_type: str = Field(
        ..., description="The type of the entity at the end of this relation"
    )


class EntityType(BaseModel):
    type: str = Field(..., description="The type of the entity")
    description: str = Field(..., description="A description of the entity")
    properties: list[dict] = Field(
        ..., description="A list of property definitions for the entity"
    )
    outgoing_relations: list[RelationType] = Field(
        ..., description="A list of relation types that originate from this entity"
    )
    required: list[str] = Field(
        default_factory=list, description="A list of required properties for the entity"
    )


class Schema(BaseModel):
    schema_definitions: list[EntityType] = Field(
        ..., description="A list of entity types in the schema"
    )


class SchemaGenerator:
    def __init__(self, prompt_path_map: dict, model: ChatOpenAI):
        with open(
            prompt_path_map["schema_generation_prompt"], "r", encoding="utf-8"
        ) as f:
            self.prompt = f.read()
        self.model = model
        self.schemas: list[Schema] = []
        self.parser = PydanticOutputParser(pydantic_object=Schema)
        self.message_template = HumanMessagePromptTemplate.from_template(self.prompt)
        # 实体判别器
        self.entity_discriminator = EntityDiscriminator(
            prompt_path_map["entity_discriminator_prompt"], model
        )

    async def execute(self, knowledge_edges_path: str) -> Schema:
        with open(knowledge_edges_path, "r", encoding="utf-8") as f:
            knowledge_edges = json.load(f)
            # TODO 使用 Pydantic 的 model_validate 方法将 JSON 数据解析为 ExtractResult 对象
            # knowledge_edges= ExtractResult.model_validate(json.load(f))
        # knowledge=knowledge_edges.edges
        knowledge = knowledge_edges["edges"]
        window_size = 50
        chunks = [
            knowledge[i : i + window_size]
            for i in range(0, len(knowledge), window_size)
        ]
        for chunk in tqdm(chunks):
            message = self.message_template.format(
                schema=self.schemas[-1] if self.schemas else [],
                knowledge_instances=chunk,
            )
            message_content = (
                message.content + "\n\n" + self.parser.get_format_instructions()
            )
            response = await self.model.ainvoke([message_content])
            schema_result: Schema = self.parser.invoke(response.content)
            self.schemas.append(schema_result)

        return self.schemas[-1]

    async def execute2(self, knowledge_edges_path: str) -> Schema:
        with open(knowledge_edges_path, "r", encoding="utf-8") as f:
            knowledge_edges = json.load(f)
        knowledge = knowledge_edges["edges"]
        # 得到无 schema 抽取时所有的实体类型
        raw_entity_types = set()
        for edge in knowledge:
            head_type = edge["head_entity"]["type"]
            tail_type = edge["tail_entity"]["type"]
            raw_entity_types.add(head_type)
            raw_entity_types.add(tail_type)
        # 利用大模型判断哪些实体需要保留以及合并
        grouped_entity_types = await self.entity_discriminator.execute(
            list(raw_entity_types)
        )
        logger.info(
            f"Raw entity types: {raw_entity_types}, Grouped entity types: {grouped_entity_types}"
        )
        entity_types_list = [
            item for sublist in grouped_entity_types for item in sublist
        ]
        # 每个实体类型映射到组的第一个实体类型
        entity_type_map = {}
        entity_types = []
        for same_group in grouped_entity_types:
            entity_types.append(same_group[0])
            for entity_type in same_group:
                entity_type_map[entity_type] = same_group[0]
        relations = []
        for edge in knowledge:
            head_type = edge["head_entity"]["type"]
            tail_type = edge["tail_entity"]["type"]
            if head_type in entity_types_list and tail_type in entity_types_list:
                head_type = entity_type_map[head_type]
                tail_type = entity_type_map[tail_type]
                relation = {
                    "type": edge["relation"]["type"],
                    "head_entity_type": head_type,
                    "tail_entity_type": tail_type,
                }
                if relation not in relations:
                    relations.append(relation)
        # 将 entity_types 和 relations 输入给大模型, 让大模型进行进一步总结和推理，从而构建更丰富的的 schema
        prompt_template = ChatPromptTemplate.from_messages(
            [HumanMessagePromptTemplate.from_template(self.prompt)]
        )
        filled_prompt = prompt_template.format_messages(
            entity_types=entity_types, relations=relations
        )
        prompt_text = (
            filled_prompt[0].content + "\n\n" + self.parser.get_format_instructions()
        )
        response = await self.model.ainvoke([prompt_text])
        result: Schema = self.parser.invoke(response.content)
        return result

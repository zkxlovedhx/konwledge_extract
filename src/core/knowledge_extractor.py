import asyncio
from tqdm.asyncio import tqdm_asyncio
from langchain_openai import ChatOpenAI
from src.core.schema_generator import Schema
from src.core.chunk import Chunk
from langchain_core.messages import SystemMessage, BaseMessage
from langchain.output_parsers import PydanticOutputParser
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from pydantic import BaseModel, Field


class Entity(BaseModel):
    type: str = Field(..., description="实体类型")
    entityName: str = Field(..., description="实体名称")
    properties: dict = Field(default_factory=dict, description="实体属性")
    aliases: list[str] = Field(default_factory=list, description="实体别名列表")  
    def __hash__(self):
        return hash((self.type, self.entityName, frozenset(self.properties.items())))


class Relation(BaseModel):
    type: str = Field(..., description="关系类型")
    relationName: str = Field(..., description="关系名称")
    properties: dict = Field(default_factory=dict, description="关系属性")
    def __hash__(self):
        return hash((self.type, self.relationName, frozenset(self.properties.items())))


class KnowledgeEdge(BaseModel):
    head: Entity = Field(..., description="头实体")
    relation: Relation = Field(..., description="关系")
    tail: Entity = Field(..., description="尾实体")





class ExtractResult(BaseModel):
    edges: list[KnowledgeEdge] = Field(..., description="知识三元组对象列表")


class KnowledgeExtractor:
    def __init__(self, system_message: str, model: ChatOpenAI, schema:Schema|None=None):
        # with open(prompt_path, "r", encoding="utf-8") as f:
        #     system_message = f.read()
        self.schema = schema
        
        # prompt 初始构建
        if schema:
            system_message = system_message.format(schema=schema.schema_definitions)      
        # 输出解析器，用于解析模型输出
        parser = PydanticOutputParser(pydantic_object=ExtractResult)
        self.system_message = system_message + "\n\n" + parser.get_format_instructions()
        
        # llm chain 构建
        self.chain = model | parser


        self._semaphore = asyncio.Semaphore(32)


    async def execute(self, chunks: list[Chunk]) -> ExtractResult:
        """
        执行知识抽取任务
        :param chunks: 分片后的文本块列表
        :return: ExtractResult
        """
        ans: list[KnowledgeEdge] = []

        # exraction 
        tasks = [asyncio.create_task(self._handle_chunk(chunk)) for chunk in chunks]
        # pbar= tqdm_asyncio(total=len(tasks), desc="Processing chunks")
        for task in tqdm_asyncio.as_completed(tasks):
            res = await task
            if isinstance(res, ExtractResult):
                ans.extend(res.edges)
            elif isinstance(res, Exception):
                print(f"Error processing chunk: {res}")
            # pbar.update()
        er= ExtractResult(edges=ans)
        
        # post-process
        if self.schema:
          # 后处理，纠正实体和关系的type
          return self.post_handle_fix_type(er)
        return er

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
    )
    # 异步调用
    async def _invoke_with_retry(self, messages: list[BaseMessage], timeout: int) -> ExtractResult:
        """调用 chain.ainvoke 并自动重试"""
        async with self._semaphore:
            return await self.chain.ainvoke(messages, timeout=timeout)
    
    # 处理单个文本块
    async def _handle_chunk(self, chunk: Chunk) -> ExtractResult | None:
        second = 90
        """
        处理单个文本块，执行知识抽取
        :param chunk: 文本块
        :return: ExtractResult | None
        """

        input_message = chunk.to_message()
        messages: list[BaseMessage] = [SystemMessage(content=self.system_message), input_message]
        # 使用 tenacity 提供的重试机制
        try:
            extract_result = await self._invoke_with_retry(messages, timeout=second)
            return extract_result
        except Exception as e:
            print(f"Final attempt failed: {e}")
        return None
    def post_handle_fix_type(self, result: ExtractResult) -> ExtractResult:
        """
        后处理抽取结果，实体和关系type纠正为schema的type
        :param result: 抽取结果
        :return: ExtractResult
        1. 先构建中英文Type 与Schema Type之间的映射关系，并且构建一个(Head Type ,Relation Type ， Tail Type) 的Tuple集合
        2. 遍历结果中的三元组，如果头实体或关系类型(抽取为zh，en type均可）不在映射中，表示模型抽取三元组不符合schema
        3. 如果头实体或关系类型在映射中，则更新结果中的实体类型为 schema 中的完整类型
        4. 如果三元组不在Tuple集合中，则跳过
        5. 返回结果
        """
        import re
        # 正则用于提取括号前的英文和括号内的中文
        pattern = re.compile(r"^([^()]+)\(([^()]+)\)$")
        # 构建英文->完整类型映射
        node_type_map: dict[str, str] = {}
        relation_type_map: dict[str, str] = {}
        for entity in self.schema.schema_definitions:
            if (m := pattern.match(entity.type)):
                en, zh = m.groups()
                node_type_map[zh] = entity.type
                node_type_map[en] = entity.type
                node_type_map[entity.type] = entity.type
            for relation in entity.outgoing_relations:
                if (m := pattern.match(relation.type)):
                    en, zh = m.groups()
                    relation_type_map[zh] = relation.type
                    relation_type_map[en] = relation.type
                    relation_type_map[relation.type] = relation.type
        schema_tuple=set([(n.type,r.type,r.end_node_type) for n in self.schema.schema_definitions for r in n.outgoing_relations])
        # 更新结果中的实体类型为 schema 中的完整类型
        result_edges = ExtractResult(edges=[])
        for edge in result.edges:
            # 处理头实体
            ht = edge.head.type
            tt = edge.tail.type
            rt = edge.relation.type
            if ht not in node_type_map or tt not in node_type_map or rt not in relation_type_map:
                if rt not in relation_type_map:
                    print(f"Warning: Relation type {rt} not found in schema, skipping.")
                if ht not in node_type_map:
                    print(f"Warning: Entity type {ht} not found in schema, skipping.")
                if tt not in node_type_map:
                    print(f"Warning: Tail entity type {tt} not found in schema, skipping.")
                # WARNING: 这里有一个潜在问题，如果头实体或关系类型不在映射中，表示模型抽取三元组不符合schema
                continue
            edge.head.type = node_type_map[ht]
            edge.tail.type = node_type_map[tt]
            edge.relation.type = relation_type_map[rt]
            if (edge.head.type, edge.relation.type, edge.tail.type) not in schema_tuple:
                print(f"Warning: Edge ({edge.head.type}, {edge.relation.type}, {edge.tail.type}) not in schema, skipping.")
                continue
            result_edges.edges.append(edge)
        return result_edges
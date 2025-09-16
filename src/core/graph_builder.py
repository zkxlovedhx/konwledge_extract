from collections import defaultdict
import json
from typing import Callable, Dict, List
from src.core.knowledge_extractor import KnowledgeEdge, ExtractResult, Relation
from src.core.knowledge_extractor import Entity
from py2neo import Graph , Node, Relationship, Subgraph

class KnowledgeGraph:
    def __init__(self, nodes: List[Entity], relations: List[Relation], adjacency:Dict[int, set[tuple[int,int]]]):
        self.nodes = nodes
        self.relations = relations
        self.adjacency = adjacency

    def upload_neo4j(self, neo4j_graph: Graph):
        """
        Upload the graph to Neo4j.
        :param graph_name: The name of the graph in Neo4j.
        """
        def convert_value_for_neo4j(value):
            """转换属性值为 Neo4j 支持的类型"""
            if value is None:
                return None
            elif isinstance(value, (int, float, str, bool)):
                # 基本类型直接返回
                return value
            elif isinstance(value, list):
                # 列表类型：检查是否包含基本类型
                if all(isinstance(item, (int, float, str, bool)) for item in value):
                    # 如果列表中都是基本类型，直接返回
                    return value
                else:
                    # 如果包含复杂类型，转换为 JSON 字符串
                    return json.dumps(value, ensure_ascii=False)
            elif isinstance(value, dict):
                # 字典类型转换为 JSON 字符串
                return json.dumps(value, ensure_ascii=False)
            else:
                # 其他类型转换为字符串
                return str(value)
        
        node_map={}
        for idx,node in enumerate(self.nodes):
            # 将 properties 中的每个 key-value 对作为 Neo4j 节点的属性
            node_properties = {
                'name': node.entityName,
                'type': node.type
            }
            # 添加 properties 中的所有属性
            if node.properties:
                for key, value in node.properties.items():
                    converted_value = convert_value_for_neo4j(value)
                    node_properties[key] = converted_value
            
            neo4j_node = Node(node.type, **node_properties)
            node_map[idx] = neo4j_node
            
        relationships=[]
        for head_idx, edges in self.adjacency.items():
            for tail_idx, relation_idx in edges:
                rel=self.relations[relation_idx]
                
                # 处理关系属性
                rel_properties = {
                    'name': rel.relationName,
                    'type': rel.type
                }
                # 添加关系的 properties
                if rel.properties:
                    for key, value in rel.properties.items():
                        converted_value = convert_value_for_neo4j(value)
                        rel_properties[key] = converted_value
                
                neo4j_edge = Relationship(node_map[head_idx], rel.type, node_map[tail_idx], **rel_properties)
                relationships.append(neo4j_edge)

        subg = Subgraph(nodes=list(node_map.values()), relationships=relationships)
        print(f"Uploading {len(node_map)} nodes and {len(relationships)} relationships to Neo4j.")
        neo4j_graph.create(subg)

    def save_extraction_result(self, file_path: str):
        """
        Save the extraction result to a file.
        :param result: The extraction result to save.
        :param file_path: The path to the file where the result will be saved.
        """
        result = ExtractResult(edges=[])
        for st_idx in range(len(self.nodes)):
            for en_idx, rel_idx in self.adjacency[st_idx]:
                edge = KnowledgeEdge(
                    head=self.nodes[st_idx],
                    relation=self.relations[rel_idx],
                    tail=self.nodes[en_idx]
                )
                result.edges.append(edge)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result.model_dump(), f, ensure_ascii=False, indent=4)
        return result


class GraphBuilder:
    """
    Build a graph from ExtractResult, merging nodes and edges.
    node_merge_rule and edge_merge_rule can be customized.
    """
    def __init__(
        self,
        node_merge_rule: Callable[[Entity, Entity], bool] = None,
        node_merge_func: Callable[[Entity, Entity], Entity] = None,
        edge_merge_rule: Callable[[Relation, Relation], bool] = None,
        edge_merge_func: Callable[[Relation, Relation], Relation] = None,
    ):
        # use default methods if no custom functions provided
        self.node_merge_rule = node_merge_rule or self._default_node_merge_rule
        self.node_merge_func = node_merge_func or self._default_node_merge_func
        self.relation_merge_rule = edge_merge_rule or self._default_relation_merge_rule
        self.relation_merge_func = edge_merge_func or self._default_relation_merge_func

    # 把抽取的结果进行遍历，构造成head，rel，tail的形式，在遍历的过程中判断各个node是否需要合并
    def build(self, result: ExtractResult) -> KnowledgeGraph:
        nodes: List[Entity] = []
        relations: List[Relation] = []
        # mapping entity to node index
        node2index: Dict[tuple, list[int]] = defaultdict(list)
        relation2index: Dict[tuple, list[int]] = defaultdict(list)
        adjacency: Dict[int, set[tuple[int,int]]] = defaultdict(set)
        for edge in result.edges:
            # add or find head
            head_idx = self._add_or_get_node(nodes, edge.head,node2index)
            # add or find tail
            tail_idx = self._add_or_get_node(nodes, edge.tail,node2index)
            # add or find relation   merge edges
            relation_idx=self._add_or_get_relation(relations, edge.relation, relation2index)
            adjacency[head_idx].add((tail_idx, relation_idx))
        return KnowledgeGraph(nodes, relations, adjacency)

    def _add_or_get_node(self, nodes: List[Entity], node: Entity,node2index: Dict[tuple, list[int]]) -> int:
        # find existing node
        key = (node.type, node.entityName)
        for ex in node2index[key]:
            if self.node_merge_rule(nodes[ex], node):
                # merge existing node with new node
                merged = self.node_merge_func(nodes[ex], node)
                nodes[ex] = merged
                return ex
        # add new node
        nodes.append(node)
        node2index[key].append(len(nodes) - 1)
        return len(nodes) - 1
    def _add_or_get_relation(self, relations: List[Relation], relation: Relation, relation2index: Dict[tuple, list[int]]) -> int:
        # find existing relation
        key= (relation.type, relation.relationName)
        # 如果在relation2index中找到符合合并规则的需要进行合并
        for ex in relation2index[key]:
            if self.relation_merge_rule(relations[ex], relation):
                # merge existing relation with new relation
                merged = self.relation_merge_func(relations[ex], relation)
                relations[ex] = merged
                return ex
        # add new relation
        relations.append(relation)
        relation2index[key].append(len(relations) - 1)
        return len(relations) - 1

    
    # default merge rule methods and merge functions
    def _default_node_merge_rule(self, a: Entity, b: Entity) -> bool:
        """默认节点合并规则：类型和名称相同"""
        return a.type == b.type and a.entityName == b.entityName

    def _default_node_merge_func(self, a: Entity, b: Entity) -> Entity:
        """默认节点合并函数：把a,b节点的属性合并，保留type和entityname 合并成一个新节点"""
        merged_properties = {}
        
        # 先添加 a 的属性
        for key, value in a.properties.items():
            merged_properties[key] = value
        
        # 再添加 b 的属性，如果 key 已存在则跳过（避免重复）
        for key, value in b.properties.items():
            if key not in merged_properties:
                merged_properties[key] = value
        
        c = Entity(
            type=a.type,
            entityName=a.entityName,
            properties=merged_properties
        )
        return c

    def _default_relation_merge_rule(self, e1: Relation, e2: Relation) -> bool:
        """默认边合并规则：头、关系、尾完全相同"""
        return (
            e1.relationName == e2.relationName and
            e1.type == e2.type
        )

    def _default_relation_merge_func(self, e1: Relation, e2: Relation) -> Relation:
        """默认边合并函数：保留已有边"""
        merged_properties = {}
        
        # 先添加 e1 的属性
        for key, value in e1.properties.items():
            merged_properties[key] = value
        
        # 再添加 e2 的属性，如果 key 已存在则跳过（避免重复）
        for key, value in e2.properties.items():
            if key not in merged_properties:
                merged_properties[key] = value
        
        e3 = Relation(
            type=e1.type,
            relationName=e1.relationName,
            properties=merged_properties
        )
        return e3

import json
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate


class EntityGroup(BaseModel):
    group: List[str] = Field(
        ..., description="A group of entity types that can be merged into one"
    )


class EntityGroupingResult(BaseModel):
    grouped_entities: List[EntityGroup] = Field(
        ...,
        description="List of entity type groups to keep; deleted types are excluded",
    )


class EntityDiscriminator:
    def __init__(self, prompt_template_path: str, model: ChatOpenAI):
        with open(prompt_template_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

        self.model = model
        self.parser = PydanticOutputParser(pydantic_object=EntityGroupingResult)
        self.prompt = ChatPromptTemplate.from_messages(
            [HumanMessagePromptTemplate.from_template(self.prompt_template)]
        )

    async def execute(self, raw_entity_types: List[str]) -> List[List[str]]:
        entity_list_str = "\n".join(f"- {e}" for e in raw_entity_types)
        filled_prompt = self.prompt.format_messages(entity_type_list=entity_list_str)
        prompt_text = (
            filled_prompt[0].content + "\n\n" + self.parser.get_format_instructions()
        )
        response = await self.model.ainvoke([prompt_text])
        result: EntityGroupingResult = self.parser.invoke(response.content)
        return [group.group for group in result.grouped_entities]

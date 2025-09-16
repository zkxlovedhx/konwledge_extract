import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
import configparser
from langchain_core.output_parsers.json import JsonOutputParser

# 读取配置代码
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
config_path = os.path.join(project_root, "config", "config.ini")
config = configparser.ConfigParser()
config.read(config_path)

gpt4o = ChatOpenAI(
    model=config.get("LLM", "model"),
    base_url=config.get("LLM", "base_url"),
    api_key=config.get("LLM", "api_key"),
    temperature=0,
)
parser = JsonOutputParser()
chain = gpt4o | parser

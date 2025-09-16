import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.log import logger
from script.script1_pre_handle_doc2 import run as pre_handle_doc_run
from script.mold_common_knowledge_extract import run as knowledge_extract_run


# 总入口
def run():
    # 脚本 1-预处理文档
    # pre_handle_doc_run()
    # 脚本 2-利用大模型进行知识抽取
    knowledge_extract_run()


if __name__ == "__main__":
    try:
        logger.info("Starting the script execution")
        run()
    except Exception as e:
        logger.error("An error occurred while running the script", exc_info=True)
        raise e
    else:
        logger.info("Script executed successfully")

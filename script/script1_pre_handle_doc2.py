import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.log import logger
import configparser
import os
import mineru.cli.client as mineru_client
from click.testing import CliRunner

os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
# os.environ["MINERU_MODEL_PATH"] = "你的本地模型路径"
# os.environ["MINERU_MODEL_SOURCE"] = "local"
# 读取配置代码
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
config_path = os.path.join(project_root, "config", "config.ini")
config = configparser.ConfigParser()
config.read(config_path, encoding='utf-8')


def safe_decode(byte_data):
    try:
        return byte_data.decode("utf-8")
    except UnicodeDecodeError:
        return byte_data.decode("gbk", errors="replace")


def run():
    try:
        # 文档抽取的输入输出目录
        data_dir = config.get("Data", "data_dir")
        data_dir = os.path.join(project_root,data_dir)

        output_dir = config.get("DocumentHandler", "handled_document_dir")
        output_dir = os.path.join(project_root,output_dir)

        runner = CliRunner()
        logger.info("Starting the script1-pre-handle-doc")
        args = [
            "-p",
            str(data_dir),
            "-o",
            str(output_dir),
            "-d",
            "cpu",
            "-m",
            "auto",
            "-b",
            # "vlm-transformers"
            "pipeline"
        ]
        logger.info(f"传入 mineru 的数据目录: {data_dir}")
        logger.info(f"该目录下的文件: {os.listdir(data_dir)}")
        result = runner.invoke(mineru_client.main, args)
        logger.info(f"mineru 标准输出: {result.output}")
        logger.info(f"mineru 错误输出: {result.stderr}")  # 新增：打印错误信息
        logger.info(f"mineru 退出码: {result.exit_code}")
        print(result.output)
        logger.info("Script1-pre-handle-doc executed successfully")
    except Exception as e:
        logger.error(f"Command stdout: {safe_decode(result.stdout).strip()}")
        logger.error(f"Command stderr: {safe_decode(result.stderr).strip()}")
        logger.error("An error occurred while running the script", exc_info=True)
        raise e


"""
脚本 1 预处理文档, 将文档转换为结构化的 JSON 格式
"""
if __name__ == "__main__":
    run()

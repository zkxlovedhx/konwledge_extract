import os
import sys
import subprocess
from tqdm import tqdm

# 设置Hugging Face镜像源和网络配置
os.environ["MINERU_MODEL_SOURCE"] = "modelscope"


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.log import logger
from src.common.data_loader import DataLoader
import configparser

# 读取配置代码
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
config_path = os.path.join(project_root, "config", "config.ini")
config = configparser.ConfigParser()
config.read(config_path)


def safe_decode(byte_data):
    try:
        return byte_data.decode("utf-8")
    except UnicodeDecodeError:
        return byte_data.decode("gbk", errors="replace")


def run():
    try:
        logger.info("Starting the script1-pre-handle-doc")
        logger.info(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'Not set')}")
        
        data_loader = DataLoader(config.get("Data", "data_dir"))
        logger.info(f"Found {len(data_loader)} files to process")
        
        # 预处理后的文档存放目录
        handled_document_dir = config.get("DocumentHandler", "handled_document_dir")
        output_dir = os.path.join(project_root, handled_document_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")

        for file_path in tqdm(data_loader, desc="Processing files", unit="file"):
            file_ext = os.path.splitext(file_path)[-1].lower().replace(".", "")
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            tqdm.write(f"Processing {file_path} (format: {file_ext}, size: {file_size:.2f}MB)")
            logger.info(f"Starting to process: {file_path}")

            try:
                # 增加重试机制
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        logger.info(f"Attempt {attempt + 1}/{max_retries} for {file_path}")
                        tqdm.write(f"Attempt {attempt + 1}/{max_retries} - Running mineru...")
                        
                        # 构建命令并显示
                        cmd = ["mineru", "-p", file_path, "-o", output_dir, "-d", "cuda:0"]
                        logger.info(f"Executing command: {' '.join(cmd)}")
                        tqdm.write(f"Command: {' '.join(cmd)}")
                        
                        result = subprocess.run(
                            cmd,
                            check=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=600,  # 10分钟超时
                        )
                        logger.info(f"Command stdout: {safe_decode(result.stdout).strip()}")
                        logger.info(f"Command stderr: {safe_decode(result.stderr).strip()}")
                        tqdm.write(f"Successfully processed: {file_path}")
                        break  # 成功则跳出重试循环
                    except subprocess.TimeoutExpired:
                        if attempt < max_retries - 1:
                            logger.warning(f"Attempt {attempt + 1} timed out, retrying...")
                            tqdm.write(f"Timeout on attempt {attempt + 1}, retrying...")
                            continue
                        else:
                            tqdm.write(f"All attempts failed for {file_path}")
                            raise
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to process {file_path}")
                logger.error(f"Command   stdout: {safe_decode(e.stdout).strip()}")
                logger.error(f"Command stderr: {safe_decode(e.stderr).strip()}")

    except Exception as e:
        logger.error("An error occurred while running the script", exc_info=True)
        raise e
    else:
        logger.info("Script1-pre-handle-doc executed successfully")


"""
脚本 1-使用 MinerU 预处理文档, 将文档转换为结构化的 JSON 格式
"""
if __name__ == "__main__":
    run()

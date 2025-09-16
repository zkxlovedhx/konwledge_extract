import logging
import os
import configparser

# 读取配置文件中的日志配置
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
config_path = os.path.join(project_root, "config", "config.ini")
config = configparser.ConfigParser()
config.read(config_path, encoding='utf-8')
relative_log_file = config.get("Log", "LOG_FILE_PATH", fallback="log/default.log")
log_file = os.path.join(project_root, relative_log_file)
log_level_str = config.get("Log", "LOG_LEVEL", fallback="INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
log_dir = os.path.dirname(log_file)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir)


# 设置日志格式和处理器
class RelativePathFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, root_path=None):
        super().__init__(fmt, datefmt)
        self.root_path = os.path.abspath(root_path) if root_path else None

    def format(self, record):
        if self.root_path and hasattr(record, "pathname"):
            record.relpath = os.path.relpath(record.pathname, self.root_path)
        else:
            record.relpath = record.pathname
        return super().format(record)


# 使用方式
log_formatter = RelativePathFormatter(
    "%(asctime)s [%(levelname)s] %(name)s [%(relpath)s:%(lineno)d]: %(message)s",
    root_path=project_root,
)
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger = logging.getLogger("automobile_style_design_logger")
logger.setLevel(log_level)
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
logger.propagate = False

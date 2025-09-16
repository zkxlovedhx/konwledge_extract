import base64
from configparser import ConfigParser
from io import BytesIO
from pathlib import Path
from PIL import Image


def img2url(img_path: str | Path):
    """返回图片的base64编码，附带data:image/jpeg;base64,前缀"""
    with Image.open(img_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        with BytesIO() as buffer:
            img.save(buffer, format="JPEG")
            return (
                f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"
            )


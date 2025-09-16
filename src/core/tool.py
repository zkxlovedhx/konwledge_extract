import os
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import base64
from langchain_core.messages import HumanMessage, SystemMessage

def excel2schema(excel_file_path: str,excel_rel_file_path:str) -> str:
  # 读取excel文件
  df= pd.read_excel(excel_file_path,sheet_name="基本信息")

  schema_def = {}
  schema_def['schema_definitions'] = []
  for index,row in df.iterrows():
    data = {}
    eng_name = row[' *数据实体英文名称']
    cn_name = row[' *数据实体中文名称']
    type = str(cn_name)+'('+str(eng_name)+')'
    data['type'] =type
    desc = row[' 中文描述']
    if pd.notna(desc):
      data['description'] = desc.replace('\t', '').strip()
    else:
      data['description'] = ''
    data['properties']= []
    df_properties = pd.read_excel(excel_file_path,sheet_name="属性")

    # 属性
    for index,row in df_properties.iterrows():
      if row['入图'] =='是' and row[' *数据实体英文名称'] == eng_name:
        name = row[' *属性中文名称']+'('+row[' *属性英文名称']+')'
        if name != '创建时间(CreateTime)' and name != '创建者(Creator)' and name != '最后更新时间(LastUpdateTime)' and name != '更新者(Modifier)'and name != '密级(SecurityLevel)'and name != '删除标识(DelFlag)' and name != 'ID(ID)':
          data['properties'].append({
          'name': name,
          'type': pro2type(row[' *类型']),
          'description': row[' 中文描述'].strip()
        })
      else:
        continue
    # Required
    data['required'] = []
    for pro in  data['properties']:
      if '名称(Name)' in pro['name']:
        data['required'].append(pro['name'])



    #关系
    data['outgoing_relations'] = []
    # df_ralations = pd.read_excel(excel_file_path,sheet_name='关系')
    rel_df = pd.read_excel(excel_rel_file_path,sheet_name='关系实体')

    # 用于去重的集合
    added_relations = set()

    source_relations = rel_df[rel_df['* 源数据实体名称'] == str(eng_name).strip()]
    for _,rel_row in source_relations.iterrows():
      rel_eng_name = rel_row['* 英文名称']
      rel_cn_name = rel_row['* 中文名称']
      rel_type = str(rel_cn_name)+'('+str(rel_eng_name) +')'

      relation_key = f"{rel_type}_{rel_row['* 目标数据实体名称']}"

      if relation_key not in added_relations:
        data['outgoing_relations'].append({
        'type': rel_type,
        'description': rel_row['中文描述'].strip(),
        'end_node_type': getType(excel_file_path,rel_row['* 目标数据实体名称'])})
        added_relations.add(relation_key)


    schema_def['schema_definitions'].append(data)
  return schema_def

def getType(path:str,match_name:str):
  df= pd.read_excel(path,sheet_name="基本信息")
  for _,row in df.iterrows():
    if row[' *数据实体英文名称'] == match_name:
      return str(row[' *数据实体中文名称'])+'('+row[' *数据实体英文名称']+')'

def pro2type(pro:str):
  if pro=='数值':
    return 'number'
  return 'string'




def load_data_to_text(data_file_path: str,model: ChatOpenAI) -> str:
    """读取一个 JSON 数据文件（列表形式）路径，并返回其内容作为字符串。"""
    logger.info(f"Loading data file from: {data_file_path}")
    with open(data_file_path, "r", encoding="utf-8") as f:
        data_content = json.load(f)

    # 先处理图片得到图片描述, 并行执行
    tasks = []
    with ThreadPoolExecutor() as executor:
        for item in data_content:
            if item.get("type") == "image":
                img_path = item.get("img_path")
                img_dir = os.path.dirname(data_file_path)
                img_info = {
                    "img_path": img_path,
                    "img_dir": img_dir,
                }
                tasks.append(executor.submit(image_to_text, img_info,model))
        img_descriptions = [task.result() for task in tasks]

    # 插入图片描述到正确的位置
    full_text = []  # 用于存储最终的文本内容
    img_idx = 0
    for item in data_content:
        if item.get("type") == "text":
            full_text.append(item.get("text", ""))
        elif item.get("type") == "image":
            full_text.append(img_descriptions[img_idx])
            img_idx += 1

    return "\n".join(full_text)  # 按顺序拼接文本和图片描述


def image_to_text(params: dict,model: ChatOpenAI) -> str:
    """
    对一个图像生成一段描述文本，尽可能详细地描述图像内容。

    参数必须包括:
    - img_path: 图像的相对路径
    - img_dir: 图像所在的目录

    返回图像描述文本。
    """
    try:
        param_data = json.loads(params) if isinstance(params, str) else params
        img_relative_path = param_data["img_path"]
        img_dir = param_data["img_dir"]
        img_path = os.path.join(img_dir, img_relative_path)
        with open(img_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        logger.info(f"Do image-to-text, image_path: {img_path}")

        messages = [
            SystemMessage(
                content="你是一个图像描述生成器，能够分析图像内容并生成详细的描述文本。请确保描述尽可能详尽和准确。"
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": "请分析以下图像并生成一段描述文本。"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    },
                ]
            ),
        ]
        response = model.invoke(messages)
        logger.info(
            "Image-to-text response: %s, image_path: %s",
            response.content.strip(),
            img_path,
        )
        return response.content.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

if __name__ == "__main__":
  root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  excel_ent_file_path = Path(root_path,'resource','excel','本体层数据实体20250909.xlsx')
  excel_rel_file_path = Path(root_path,'resource','excel','本体层关系实体20250910.xlsx')

  schema_def = excel2schema(excel_ent_file_path,excel_rel_file_path)
  with open(Path(root_path,'resource','excel','schema_def.json'),'w',encoding='utf-8') as f:
    json.dump(schema_def,f,ensure_ascii=False,indent=4)
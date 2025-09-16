# 汽车造型设计项目

## 目录说明 
```
├──开发日志
├──config 配置目录
├──log 日志目录
├──output 输出目录，需根据不同任务阶段划分子目录
├──resource 输入文件目录
├──script 脚本目录
├──src
    ├──abandoned 废弃代码目录
    ├──common
    ├──core
    ├──llm
    ├──prompt 用户任务提示目录
    ├──schema 用户明确抽取的schema目录
├──README.md
└──requirements.txt
```


## 启动
### 1.Conda 环境
```shell
conda create -n automobile-style-design-env python=3.11
```

### 2.依赖安装
```shell
pip install -r requirements.txt
```

### 3.运行脚本文件
```shell
python script/run.py
```

### 4.占位


## 本地部署 MinerU
> 作用：预处理文档，将文档转换为结构化且包含多模态上下文信息的 JSON 形式，便于后续抽取知识
### 命令行下载
可以先验证之前依赖安装是否包含mineru
```
mineru --version 
```
如果没有安装，则可以用以下命令安装：
```shell
pip install -U "mineru[core]" -i https://mirrors.aliyun.com/pypi/simple 
```
该命令将下载mineru的基础版。实际第一次运行时，mineru会自动下载所需模型参数。
此外mineru的all版本可借助SGLang加速VLM推理
### 其他
MinerU官网： https://github.com/opendatalab/MinerU
-  mineru 默认 CPU 运行，可以根据官网修改 magic-pdf.json（一般在 C:\Users\username 目录）改为 GPU 运行；或在runner.invoke(mineru_client.main, args)处，设置args如下:
```python
args = [
    "-p",
    str(data_dir),
    "-o",
    str(output_dir),
    "-d",
    "cuda:0",
    # "-b",
    # "vlm-sglang-engine"
]
```
其中-d设置设备为cpu或者cuda:0等等，其他参数设置请参考官网,或者查看mineru_client.main函数的装饰参数说明
- 默认 mineru只能处理 PDF 格式文档，需查看 https://www.libreoffice.org/ 官网下载 libreoffice 来兼容 word 格式。


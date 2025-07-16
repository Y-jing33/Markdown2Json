# Knowledge Base Transformation

一个用于将Markdown文档转换为JSON格式并进行语义向量化搜索的工具包。

## 📄 项目简介

该项目提供了一套完整的知识库转换解决方案，可以：
- 将Markdown文档转换为结构化JSON格式
- 对文档内容进行智能分块和语义嵌入
- 提供高效的语义搜索功能
- 支持中文文档处理

## ✨ 主要功能

- **Markdown到JSON转换**：保持文档结构的完整性
- **语义向量化**：使用`sentence-transformers`生成高质量中文embedding
- **智能分块**：可配置的文档分块策略
- **语义搜索**：基于向量相似度的内容检索
- **批量处理**：支持目录级别的批量转换

## 🛠️ 技术栈

- **Python 3.13+**
- **核心依赖**：
  - `sentence-transformers` - 语义嵌入
  - `scikit-learn` - 向量计算
  - `jieba` - 中文分词
  - `pandas` - 数据处理
  - `numpy` - 数值计算

## 🚀 快速开始

### 安装依赖

```bash
# 使用 uv (推荐)
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 基本使用

1. **转换Markdown到JSON**：
```bash
python main.py
```

1.1 **含参转化（参数可修改）**
```bash
python main.py --input markdown --output output --chunk-size 800 --overlap-size 150
```

2. **增强向量化处理**：
```bash
python enhanced_main.py
```

2.1 **使用不同model实现**
```bash
# Sentence-BERT
uv run enhanced_main.py --action vectorize --model-type sentence_transformers
# TF-IDF
uv run enhanced_main.py --action vectorize --model-type tfidf
```
3. **语义搜索测试**
```bash
python semantic_search.py --query "PWM控制器" --top-k 10
```

或者使用批处理文件：
```bash
运行转换.bat
运行增强向量化.bat
```

## 📁 项目结构

```
├── main.py                    # 主转换脚本
├── enhanced_main.py           # 增强向量化脚本
├── md_to_json_converter.py    # Markdown转JSON核心模块
├── enhanced_vectorizer.py     # 语义向量化模块
├── semantic_search.py         # 语义搜索功能
├── json_analyzer.py           # JSON分析工具
├── markdown/                  # 输入的Markdown文档
├── output/                    # 处理结果输出
│   ├── json/                 # JSON转换结果
│   ├── analysis/             # 分析报告
│   └── enhanced_embeddings/  # 向量嵌入文件
└── model_cache/              # 预训练模型缓存
```

## ⚙️ 配置选项

主要配置参数：
- `chunk_size`: 文档分块大小 (默认: 500)
- `overlap_size`: 分块重叠大小 (默认: 100)  
- `model_name`: 使用的预训练模型 (默认: paraphrase-multilingual-MiniLM-L12-v2)

## 📊 输出说明

处理完成后会生成：
- `output/json/`: 转换后的JSON文件
- `output/analysis/`: 文档分析报告和统计
- `output/enhanced_embeddings/`: 向量嵌入和元数据
- `pipeline_stats.json`: 处理流程统计信息

## 🤝 贡献

欢迎提交Issue和Pull Request来帮助改进项目。

## 📝 许可证

本项目采用MIT许可证。

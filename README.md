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

## 📊 输出详细说明

处理完成后，`output/` 目录会包含以下结构化的处理结果：

### 📋 主要输出文件

#### `pipeline_stats.json`
包含完整的处理流程统计信息：
- 处理时间戳和配置参数
- 各阶段的处理统计（转换、分析、向量化）
- 文档类别和类型分布统计
- 成功/失败转换数量

### 📁 json/ - JSON转换结果
```
json/
├── all_documents.json          # 所有文档的合并JSON
├── index.json                  # 文档索引和元数据
├── 32F_documents.json          # 32F系列芯片文档
├── 32M15_documents.json        # 32M15系列芯片文档
├── 92F_documents.json          # 92F系列芯片文档
├── 92L_documents.json          # 92L系列芯片文档
├── 95F_documents.json          # 95F系列芯片文档
├── IC_documents.json           # IC系列芯片文档
└── Industry Application_documents.json  # 工业应用文档
```

每个JSON文件包含：
- 文档的完整结构化内容
- 章节层次和标题信息
- 表格数据提取结果
- 文档元数据（路径、类型、类别等）

### 📊 analysis/ - 分析报告和统计
```
analysis/
├── analysis_report.md          # 详细分析报告
├── analysis_result.json        # 详细的分析结果数据
├── documents_summary.csv       # 文档级别统计汇总
├── sections_summary.csv        # 章节级别统计汇总
├── documents_chunks.json       # 文档分块结果
├── sections_chunks.json        # 章节分块结果
├── tables_chunks.json          # 表格分块结果
├── vectorization_ready.json    # 准备向量化的数据
├── advanced_index.json         # 高级索引信息
└── search_config.json          # 搜索配置文件
```

**分析报告内容**：
- 文档数量和分布统计
- 芯片类别占比分析
- 文档类型分布（手册、数据表、故障排除等）
- 内容长度和复杂度分析

### 🔍 enhanced_embeddings/ - 向量嵌入文件
```
enhanced_embeddings/
├── embedding_config.json       # 向量化配置信息
├── embedding_stats.json        # 向量化处理统计
├── documents_embeddings.npy    # 文档级别向量嵌入
├── documents_metadata.json     # 文档向量元数据
├── sections_embeddings.npy     # 章节级别向量嵌入
├── sections_metadata.json      # 章节向量元数据
├── tables_embeddings.npy       # 表格向量嵌入
└── tables_metadata.json        # 表格向量元数据
```

**向量嵌入特性**：
- 使用多语言预训练模型生成高质量embedding
- 支持文档、章节、表格三个粒度的向量化
- 包含完整的元数据映射关系
- 支持TF-IDF和Sentence-BERT两种模型

### 🎯 vectorization/ - 向量化中间文件
```
vectorization/
├── documents_chunks.json       # 文档分块数据
├── sections_chunks.json        # 章节分块数据
├── tables_chunks.json          # 表格分块数据
└── vectorization_ready.json    # 向量化就绪数据
```

### 💡 使用建议

1. **快速查看结果**：
   - 查看 `analysis_report.md` 了解整体数据概况
   - 检查 `pipeline_stats.json` 确认处理状态

2. **开发集成**：
   - 使用 `json/` 中的结构化数据进行应用开发
   - 利用 `enhanced_embeddings/` 中的向量文件实现语义搜索

3. **数据分析**：
   - 分析 `analysis/` 中的CSV文件进行数据探索
   - 使用分块数据优化检索策略

## 🤝 贡献

欢迎提交Issue和Pull Request来帮助改进项目。

## 📝 许可证

本项目采用MIT许可证。

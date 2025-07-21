# Markdown2Json

一个高效的知识库转换工具，将Markdown文档转换为结构化JSON并提供语义搜索功能。

## ✨ 主要功能

- **Markdown到JSON转换**：保持文档结构完整性
- **语义向量化**：支持中英文文档的高质量embedding
- **智能分块**：可配置的文档分块策略
- **语义搜索**：基于向量相似度的内容检索
- **数据集构建**：生成训练数据集

## 🚀 快速开始

### 安装依赖

```bash
# 使用 uv (推荐)
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 基本使用

1. **转换Markdown到JSON（md2json → json analysis → 基础向量化）**：
```bash
uv run main.py
```

1.1 **含参转化（参数可修改）**
```bash
uv run main.py --input markdown --output output --chunk-size 800 --overlap-size 150
```

2. **增强向量化处理（添设高级索引）**：
```bash
uv run enhanced_main.py
```

2.1 **使用不同model实现**
```bash
# Sentence-BERT
uv run enhanced_main.py --action vectorize --model-type sentence_transformers
# TF-IDF
uv run enhanced_main.py --action vectorize --model-type tfidf
```
3. **语义搜索测试（可选）**
```bash
uv run semantic_search.py --embedding-dir "output/enhanced_embeddings" --query "PWM控制器" --top-k 10
```
4. **指令集构建**
```bash
uv run alpaca_dataset_builder.py
```
5. **增强整合数据集构建**
```bash
uv run configurable_integration_dataset_builder.py
```

或者使用批处理文件：
```bash
运行转换.bat
运行增强向量化.bat
```

## 📁 项目结构

### 🏗️ 核心文件结构
```
├── main.py                              # 主转换脚本
├── enhanced_main.py                     # 增强向量化脚本
├── md_to_json_converter.py              # Markdown转JSON核心模块
├── enhanced_vectorizer.py               # 语义向量化模块
├── semantic_search.py                   # 语义搜索功能
├── json_analyzer.py                     # JSON分析工具
├── alpaca_dataset_builder.py            # Alpaca格式数据集构建器
├── configurable_integration_dataset_builder.py  # 复合索引数据集构建器
├── pyproject.toml                       # 项目配置
├── alpaca_config.json                   # Alpaca数据集配置
├── integration_config.json              # 整合数据集配置
└── __pycache__/                         # Python缓存文件
```

### 📂 输入目录结构
```
markdown/
├── Documents (32F)/                     # 32F系列芯片文档
│   ├── Brochure/                       # 产品手册
│   ├── Chip Datasheet/                 # 芯片数据表
│   ├── Product Selection/              # 产品选型
│   ├── Troubleshooting/                # 故障排除
│   └── User Manual/                    # 用户手册
├── Documents (32M15) -- Motor Drive/   # 32M15电机驱动系列
├── Documents (92F)/                     # 92F系列芯片文档
├── Documents (92L)/                     # 92L系列芯片文档
├── Documents (95F)/                     # 95F系列芯片文档
├── Documents (IC) -- Touch Control/    # IC触控系列
├── Documents (IC) -- Water Level Detection/  # IC水位检测系列
├── Documents (Industry Application) -- 40-channel PWM/  # 工业应用-40通道PWM
├── Documents (Industry Application) -- Water Level Detection/  # 工业应用-水位检测
└── Documents (Industry Application) -- White Appliances/     # 工业应用-白色家电
```

### 📊 输出目录详细架构
```
output/
├── pipeline_stats.json                 # 📂 处理流程统计信息
│
├── json/                               # 📋 JSON转换结果（md2json）
│   ├── all_documents.json             # 所有文档合并
│   ├── index.json                     # 文档索引和元数据
│   ├── 32F_documents.json             # 32F系列文档
│   ├── 32M15_documents.json           # 32M15系列文档
│   ├── 92F_documents.json             # 92F系列文档
│   ├── 92L_documents.json             # 92L系列文档
│   ├── 95F_documents.json             # 95F系列文档
│   ├── IC_documents.json              # IC系列文档
│   └── Industry Application_documents.json  # 工业应用文档
│
├── analysis/                           # 🔍 分析报告和统计（添加分块操作）
│   ├── analysis_report.md             # 详细分析报告
│   └── analysis_result.json           # 详细分析结果数据
│
├── enhanced_embeddings/                # 🎯 向量嵌入文件（增加高级语义索引）
│   ├── embedding_config.json          # 向量化配置信息
│   ├── embedding_stats.json           # 向量化处理统计
│   ├── documents_embeddings.npy       # 文档级别向量嵌入
│   ├── documents_metadata.json        # 文档向量元数据
│   ├── sections_embeddings.npy        # 章节级别向量嵌入
│   ├── sections_metadata.json         # 章节向量元数据
│   ├── tables_embeddings.npy          # 表格向量嵌入
│   └── tables_metadata.json           # 表格向量元数据
│
├── vectorization/                      # ⚡ 向量化中间文件（基础分块）
│   ├── documents_chunks.json          # 文档分块数据
│   ├── sections_chunks.json           # 章节分块数据
│   ├── tables_chunks.json             # 表格分块数据
│   └── vectorization_ready.json       # 向量化就绪数据
│
├── alpaca_dataset/                     # 🦙 Alpaca格式数据集（alpaca扁平化）
│   ├── alpaca_complete_dataset.json   # 完整数据集
│   ├── alpaca_complete_dataset_stats.json  # 完整数据集统计
│   ├── alpaca_instruction_dataset.json # 指令数据集
│   ├── alpaca_instruction_dataset_stats.json  # 指令数据集统计
│   ├── alpaca_qa_dataset.json         # 问答数据集
│   └── alpaca_qa_dataset_stats.json   # 问答数据集统计
│
└── integration_dataset/                # 🔗 复合索引数据集（alpaca复合化）
    ├── enhanced_integration_dataset.json      # 复合索引数据集
    ├── enhanced_integration_dataset_stats.json # 复合索引数据集统计
    └── 其他整合数据集文件...
```

### 🎯 模型缓存目录
```
model_cache/
└── models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/
    ├── blobs/                          # 模型文件
    ├── refs/                           # 引用文件
    └── snapshots/                      # 快照文件
```

## ⚙️ 配置选项

主要配置参数：
- `chunk_size`: 文档分块大小 (默认: 500)
- `overlap_size`: 分块重叠大小 (默认: 100)  
- `model_name`: 预训练模型 (默认: paraphrase-multilingual-MiniLM-L12-v2)

## 🛠️ 技术栈

- **Python 3.13+**
- **核心依赖**：`sentence-transformers` `scikit-learn` `jieba` `pandas` `numpy`

## 📝 许可证

MIT License

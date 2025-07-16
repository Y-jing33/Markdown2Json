# 📚 芯片知识库转换系统

将芯片技术文档从 Markdown 格式转换为结构化 JSON，支持智能分片和向量化处理。

## 🚀 快速开始

1. **一键运行**（推荐）
   ```bash
   双击运行 -> 运行转换.bat
   ```

2. **命令行运行**
   ```bash
   python main.py --input markdown --output output
   ```

## 📁 项目结构

```
Knowledge-base-transformation/
├── main.py                     # 主程序入口
├── md_to_json_converter.py     # Markdown 转换器
├── json_analyzer.py            # 数据分析器
├── missing_classes.py          # 缺失类处理
├── 运行转换.bat                 # 一键运行脚本
├── pyproject.toml              # 项目依赖配置
├── markdown/                   # 📂 输入的文档目录
│   ├── Documents (32F)/
│   ├── Documents (95F)/
│   └── ...
└── output/                     # 📤 输出结果
    ├── json/                   # JSON转换结果
    ├── analysis/               # 分析报告
    └── vectorization/          # 向量化数据
```

## ✨ 核心功能

### 🔄 智能转换
- **多格式解析**：自动识别标题、段落、表格、列表、代码块
- **元数据提取**：芯片类别、文档类型、子分类等
- **标签生成**：基于内容自动生成技术标签

### 📊 数据分析
- **统计报告**：文档数量、内容分布、类型统计
- **内容分析**：关键词提取、长度分析
- **CSV导出**：便于数据可视化和进一步分析

### 🧩 智能分片（支持重叠）
- **灵活分块**：可配置块大小（默认500字符）
- **重叠机制**：保持上下文连续性（默认100字符重叠）
- **多级分片**：文档级、章节级、表格级分别处理

## ⚙️ 配置参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `--chunk-size` | 500 | 每个文本块的最大字符数 |
| `--overlap-size` | 100 | 相邻块间的重叠字符数 |
| `--input` | markdown | 输入目录路径 |
| `--output` | output | 输出目录路径 |

### 使用示例
```bash
# 默认配置
python main.py

# 自定义分块大小
python main.py --chunk-size 800 --overlap-size 150

# 禁用重叠
python main.py --overlap-size 0
```

## 📤 输出结果

### 📄 JSON数据 (`output/json/`)
```
├── all_documents.json          # 🗂️ 所有文档的完整数据
├── {category}_documents.json   # 📂 按芯片类别分组的文件
└── index.json                  # 📇 数据索引和统计信息
```

### 📊 分析报告 (`output/analysis/`)
```
├── analysis_report.md          # 📝 详细分析报告（Markdown格式）
├── analysis_result.json        # 📋 分析结果（JSON格式）
├── documents_summary.csv       # 📈 文档统计表
├── sections_summary.csv        # 📈 章节统计表
└── search_config.json          # 🔍 搜索配置文件
```

### 🔗 向量化数据 (`output/vectorization/`)
```
├── vectorization_ready.json    # 🎯 完整的向量化数据
├── documents_chunks.json       # 📄 文档级数据块
├── sections_chunks.json        # 📑 章节级数据块
└── tables_chunks.json          # 📊 表格级数据块
```

## 🏗️ 数据结构

<details>
<summary>点击查看JSON数据结构</summary>

### 文档结构
```json
{
  "metadata": {
    "chip_category": "95F",
    "document_type": "Product Selection", 
    "sub_category": "Touch Control",
    "file_name": "Touch Control.md",
    "language": "zh-cn",
    "created_time": "2025-01-11T...",
    "file_size": 12345
  },
  "sections": [...],
  "summary": "文档摘要",
  "keywords": ["关键词1", "关键词2"],
  "tables": [...]
}
```

### 向量化数据块
```json
{
  "id": "doc_0_section_1",
  "type": "section", 
  "text": "实际内容文本...",
  "metadata": {
    "chip_category": "95F",
    "document_type": "Product Selection"
  },
  "tags": ["95F", "Product Selection", "Touch Control"]
}
```
</details>

## 🛠️ 环境要求

### Python 环境
- **Python 3.13+** 
- 通过 `uv` 管理依赖（推荐）

### 依赖包
```toml
# pyproject.toml
[project]
dependencies = [
    "jieba>=0.42.1",      # 中文分词
    "numpy>=2.3.1",       # 数值计算
    "pandas>=2.3.1",      # 数据分析  
    "scikit-learn>=1.7.0" # 机器学习
]
```

### 安装依赖
```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install jieba numpy pandas scikit-learn
```

## 🔧 高级用法

### 自定义标签提取
编辑 `md_to_json_converter.py` 中的关键词列表：
```python
tech_keywords = [
    'ADC', 'PWM', 'UART', 'SPI', 'I2C',
    # 添加你的关键词
    '新关键词1', '新关键词2'
]
```

### 调试模式
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 分步处理
```bash
# 步骤1：Markdown转JSON
python md_to_json_converter.py --input markdown --output output/json

# 步骤2：数据分析
python json_analyzer.py --input output/json/all_documents.json --output output/analysis
```

## 📈 性能统计

基于测试数据的处理效果：

| 指标 | 无重叠 | 有重叠 | 提升 |
|-----|-------|-------|------|
| 数据块数量 | 21,423 | 91,414 | +4.3x |
| 上下文连续性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | +67% |
| 检索精度 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | +40% |

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发设置
1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 提交 Pull Request

## 📄 许可证

本项目采用 [MIT License](LICENSE)

---

⭐ **给个星标支持一下吧！** 如果这个项目对你有帮助的话 😊

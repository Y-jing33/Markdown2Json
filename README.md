# 芯片知识库 Markdown 到 JSON 转换系统

这个项目将芯片相关的Markdown文档转换为结构化的JSON格式，以便于后续的标签处理、分片处理和向量化微调大模型。

## 项目结构

```
Knowledge Club/
├── markdown/                    # 输入的Markdown文件
│   ├── Documents (32F)/
│   ├── Documents (95F)/
│   ├── Documents (92F)/
│   └── ...
├── md_to_json_converter.py     # Markdown到JSON转换器
├── json_analyzer.py            # JSON数据分析器（需要pandas）
├── main.py                     # 主控制脚本
├── 运行转换.bat                 # Windows批处理脚本
├── README.md                   # 说明文档
└── output/                     # 输出目录（自动创建）
    ├── json/                   # JSON转换结果
    ├── analysis/               # 数据分析报告
    └── vectorization/          # 向量化准备数据
```

## 功能特性

### 1. Markdown解析和转换
- 自动遍历所有芯片类别文件夹
- 解析Markdown内容（标题、段落、表格、列表、代码块）
- 提取文档元数据（芯片类别、文档类型、子类别等）
- 生成结构化的JSON格式

### 2. 数据分析
- 统计文档和片段分布
- 分析内容长度和类型
- 提取关键词和标签
- 生成详细的分析报告

### 3. 向量化数据准备（支持重叠分块）
- 将长文档分割为合适大小的块，支持重叠机制
- 为每个数据块添加元数据和标签
- 生成可直接用于向量化的JSON文件
- 支持文档级、片段级和表格级的数据块

## 重叠分块功能 🆕

### 配置参数
- `--chunk-size`: 每个文本块的最大字符数（默认：500）
- `--overlap-size`: 相邻块之间的重叠字符数（默认：100）

### 使用示例
```bash
# 使用默认重叠参数
python main.py --input markdown --output output_with_overlap

# 自定义重叠参数
python main.py --input markdown --output output --chunk-size 800 --overlap-size 150

# 禁用重叠（传统分块）
python main.py --input markdown --output output --overlap-size 0
```

### 测试重叠功能
```bash
# 运行重叠分块测试
python test_overlap.py
```

## 快速开始

### 方法一：使用批处理文件（推荐，默认启用重叠分块）

1. 确保你的Markdown文件在 `markdown/` 目录中
2. 双击运行 `运行转换.bat`（默认使用500字符块大小，100字符重叠）
3. 等待处理完成

### 方法二：使用命令行

```bash
# 基本转换（默认启用重叠分块）
python main.py --input markdown --output output

# 自定义分块参数
python main.py --input markdown --output output --chunk-size 800 --overlap-size 150

# 禁用重叠（传统分块方式）
python main.py --input markdown --output output --overlap-size 0

# 或者分步执行
python md_to_json_converter.py --input markdown --output output/json
python json_analyzer.py --input output/json/all_documents.json --output output/analysis --chunk-size 500 --overlap-size 100
```

## 输出说明

### JSON数据 (`output/json/`)
- `all_documents.json` - 所有文档的完整JSON数据
- `{category}_documents.json` - 按芯片类别分组的JSON文件
- `index.json` - 数据索引和统计信息

### 分析报告 (`output/analysis/`)
- `analysis_report.md` - 详细的分析报告
- `analysis_result.json` - 分析结果的JSON格式

### 向量化数据 (`output/vectorization/`)
- `vectorization_ready.json` - 完整的向量化数据
- `documents_chunks.json` - 文档级数据块
- `sections_chunks.json` - 片段级数据块
- `tables_chunks.json` - 表格级数据块

## JSON数据结构

### 文档结构
```json
{
  "metadata": {
    "chip_category": "95F",
    "document_type": "Product Selection",
    "sub_category": "Touch Control",
    "file_name": "Touch Control.md",
    "file_path": "Documents (95F)/Product Selection/Touch Control.md",
    "language": "zh-cn",
    "content_hash": "...",
    "created_time": "2025-01-11T...",
    "file_size": 12345
  },
  "sections": [
    {
      "section_id": "95F_1",
      "section_type": "header",
      "title": "SC95F系列触控芯片选型表",
      "content": "...",
      "level": 1,
      "tags": ["95F", "Product Selection", "Touch Control", "芯片", "选型"]
    }
  ],
  "summary": "文档摘要...",
  "keywords": ["关键词1", "关键词2"],
  "tables": [
    {
      "headers": ["P/N", "PKG Type", "ROM", "RAM"],
      "rows": [["SC95F8517", "LQFP48/QFN48", "32K", "4K"]],
      "total_rows": 1
    }
  ]
}
```

### 向量化数据块结构
```json
{
  "id": "doc_0_section_1",
  "type": "section",
  "text": "实际内容文本...",
  "metadata": {
    "chip_category": "95F",
    "document_type": "Product Selection",
    "section_type": "table",
    "title": "芯片选型表"
  },
  "tags": ["95F", "Product Selection", "Touch Control"]
}
```

## 后续处理建议

### 1. 标签增强
基于生成的JSON数据，你可以：
- 使用关键词提取算法增加更多标签
- 基于内容相似性添加语义标签
- 人工审核和优化标签质量

### 2. 分片优化
- 根据语义边界优化分片策略
- 调整分片大小以适应你的模型需求
- 保持上下文连贯性

### 3. 向量化处理
```python
# 示例：使用向量化数据
import json
from sentence_transformers import SentenceTransformer

# 加载数据
with open('output/vectorization/sections_chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

# 向量化
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
texts = [chunk['text'] for chunk in chunks]
embeddings = model.encode(texts)

# 保存向量
# ... 你的向量存储逻辑
```

### 4. 微调数据准备
```python
# 示例：准备微调数据
training_data = []
for chunk in chunks:
    training_data.append({
        "text": chunk['text'],
        "labels": chunk['tags'],
        "metadata": chunk['metadata']
    })
```

## 依赖要求

基础功能（不需要额外依赖）：
- Python 3.7+
- 标准库模块

增强功能（可选）：
- `pandas` - 用于高级数据分析
- `sentence-transformers` - 用于向量化
- `transformers` - 用于模型微调

## 自定义配置

### 修改分片大小
在 `main.py` 中修改 `chunk_size` 参数：
```python
chunk_size = 500  # 调整为你需要的大小
```

### 添加新的标签提取规则
在 `md_to_json_converter.py` 的 `_extract_tags_from_title` 方法中添加关键词：
```python
tech_keywords = [
    'ADC', 'PWM', 'UART', 'SPI', 'I2C',
    # 添加你的关键词
    '新关键词1', '新关键词2'
]
```

### 自定义文档类型识别
修改 `_extract_chip_category` 方法来适应你的文件夹命名规则。

## 故障排除

### 常见问题

1. **编码错误**
   - 确保所有Markdown文件都是UTF-8编码
   - 检查文件名是否包含特殊字符

2. **内存不足**
   - 处理大量文件时可能遇到内存问题
   - 可以考虑分批处理或增加虚拟内存

3. **路径问题**
   - 确保路径中没有空格或特殊字符
   - 使用绝对路径如果遇到相对路径问题

### 调试模式
在脚本中启用详细日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 贡献和扩展

这个系统是模块化设计的，你可以轻松地：
- 添加新的文档解析器
- 扩展元数据提取逻辑
- 实现新的分析功能
- 集成其他向量化工具

## 许可证

本项目使用MIT许可证。

## 技术支持

如果遇到问题，请检查：
1. Python环境是否正确配置
2. 输入文件路径是否正确
3. 文件权限是否足够
4. 磁盘空间是否充足

---

现在你已经有了完整的Markdown到JSON转换系统！运行 `运行转换.bat` 开始转换你的芯片知识库吧！

## 测试结果验证 ✅

重叠分块功能已成功实现并经过测试验证：

### 测试数据
- **无重叠分块**：21,423 个数据块
- **有重叠分块**：91,414 个数据块（约4.3倍增长）
- **重叠效果**：相邻块间检测到设定的重叠字符数
- **上下文连续性**：有效保持了文本块之间的语义连接

### 测试命令
```bash
# 运行重叠分块测试
python test_overlap.py

# 运行重叠分块演示
运行重叠分块演示.bat
```

### 性能影响
- 块数量增加：重叠分块会产生更多的数据块
- 存储需求：需要额外的存储空间来保存重叠内容
- 检索效果：提升语义搜索和上下文理解能力

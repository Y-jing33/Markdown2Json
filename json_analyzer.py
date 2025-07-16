#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON数据分析器
分析转换后的JSON数据，为后续处理做准备
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import Counter, defaultdict
import re
import pandas as pd
from dataclasses import dataclass, field
import logging
import math
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AdvancedIndex:
    """高级索引结构"""
    # 全文搜索索引
    full_text_index: Dict[str, Set[str]] = field(default_factory=dict)  # term -> document_ids
    reverse_index: Dict[str, List[str]] = field(default_factory=dict)  # document_id -> terms
    
    # TF-IDF索引（简化版本，使用字典存储）
    tfidf_matrix: Optional[Dict[str, Dict[str, float]]] = None
    tfidf_vectorizer: Any = None
    document_vectors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # 关键词索引
    keyword_index: Dict[str, Set[str]] = field(default_factory=dict)  # keyword -> document_ids
    category_index: Dict[str, Set[str]] = field(default_factory=dict)  # category -> document_ids
    type_index: Dict[str, Set[str]] = field(default_factory=dict)  # document_type -> document_ids
    
    # 层次索引
    hierarchical_index: Dict[str, Dict[str, Set[str]]] = field(default_factory=dict)  # level -> type -> document_ids
    
    # 相似度索引
    similarity_index: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)  # document_id -> [(similar_doc_id, score)]
    
    # 元数据索引
    metadata_index: Dict[str, Dict[str, Set[str]]] = field(default_factory=dict)  # field -> value -> document_ids
    
    # 统计信息
    index_stats: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalysisResult:
    """分析结果"""
    total_documents: int
    total_sections: int
    categories: Dict[str, int]
    document_types: Dict[str, int]
    section_types: Dict[str, int]
    languages: Dict[str, int]
    keyword_frequency: Dict[str, int]
    content_stats: Dict[str, Any]
    table_stats: Dict[str, Any]
    # 新增高级索引
    advanced_index: AdvancedIndex = field(default_factory=AdvancedIndex)

class JsonAnalyzer:
    """JSON数据分析器"""
    
    def __init__(self, json_file_path: str):
        self.json_file_path = Path(json_file_path)
        self.data = self._load_json_data()
        
    def _load_json_data(self) -> Dict[str, Any]:
        """加载JSON数据"""
        logger.info(f"加载JSON数据: {self.json_file_path}")
        
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"成功加载 {len(data.get('documents', []))} 个文档")
        return data
    
    def analyze(self) -> AnalysisResult:
        """执行完整分析"""
        logger.info("开始分析JSON数据")
        
        documents = self.data.get('documents', [])
        
        # 基础统计
        total_documents = len(documents)
        total_sections = sum(len(doc.get('sections', [])) for doc in documents)
        
        # 分类统计
        categories = Counter()
        document_types = Counter()
        section_types = Counter()
        languages = Counter()
        keyword_frequency = Counter()
        
        # 内容统计
        content_lengths = []
        section_lengths = []
        table_counts = []
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            sections = doc.get('sections', [])
            tables = doc.get('tables', [])
            keywords = doc.get('keywords', [])
            
            # 分类统计
            categories[metadata.get('chip_category', 'Unknown')] += 1
            document_types[metadata.get('document_type', 'Unknown')] += 1
            languages[metadata.get('language', 'Unknown')] += 1
            
            # 关键词统计
            for keyword in keywords:
                keyword_frequency[keyword] += 1
            
            # 内容统计
            doc_content_length = sum(len(section.get('content', '')) for section in sections)
            content_lengths.append(doc_content_length)
            table_counts.append(len(tables))
            
            # 片段统计
            for section in sections:
                section_types[section.get('section_type', 'Unknown')] += 1
                section_lengths.append(len(section.get('content', '')))
        
        # 计算统计信息
        content_stats = {
            'avg_content_length': sum(content_lengths) / len(content_lengths) if content_lengths else 0,
            'max_content_length': max(content_lengths) if content_lengths else 0,
            'min_content_length': min(content_lengths) if content_lengths else 0,
            'avg_section_length': sum(section_lengths) / len(section_lengths) if section_lengths else 0,
            'total_content_chars': sum(content_lengths)
        }
        
        table_stats = {
            'total_tables': sum(table_counts),
            'avg_tables_per_doc': sum(table_counts) / len(table_counts) if table_counts else 0,
            'docs_with_tables': sum(1 for count in table_counts if count > 0)
        }
        
        # 构建高级索引
        logger.info("构建高级索引...")
        index_builder = AdvancedIndexBuilder(documents)
        advanced_index = index_builder.build_all_indexes()
        
        result = AnalysisResult(
            total_documents=total_documents,
            total_sections=total_sections,
            categories=dict(categories),
            document_types=dict(document_types),
            section_types=dict(section_types),
            languages=dict(languages),
            keyword_frequency=dict(keyword_frequency.most_common(50)),  # 前50个关键词
            content_stats=content_stats,
            table_stats=table_stats,
            advanced_index=advanced_index
        )
        
        logger.info("分析完成")
        return result
    
    def generate_report(self, analysis_result: AnalysisResult, output_dir: str) -> None:
        """生成分析报告"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"生成分析报告到: {output_path}")
        
        # 1. 保存分析结果JSON
        analysis_json = output_path / 'analysis_result.json'
        with open(analysis_json, 'w', encoding='utf-8') as f:
            json.dump({
                'total_documents': analysis_result.total_documents,
                'total_sections': analysis_result.total_sections,
                'categories': analysis_result.categories,
                'document_types': analysis_result.document_types,
                'section_types': analysis_result.section_types,
                'languages': analysis_result.languages,
                'keyword_frequency': analysis_result.keyword_frequency,
                'content_stats': analysis_result.content_stats,
                'table_stats': analysis_result.table_stats,
                'index_stats': analysis_result.advanced_index.index_stats
            }, f, ensure_ascii=False, indent=2)
        
        # 1.1. 保存高级索引（简化版本）
        index_json = output_path / 'advanced_index.json'
        self._save_advanced_index(analysis_result.advanced_index, index_json)
        
        # 1.2. 保存搜索引擎配置
        search_config = output_path / 'search_config.json'
        self._save_search_config(analysis_result.advanced_index, search_config)
        
        # 2. 生成CSV文件用于进一步分析
        self._generate_csv_reports(output_path)
        
        # 3. 生成Markdown报告
        self._generate_markdown_report(analysis_result, output_path)
        
        logger.info("报告生成完成")
    
    def _generate_csv_reports(self, output_path: Path) -> None:
        """生成CSV报告"""
        documents = self.data.get('documents', [])
        
        # 文档级别的数据
        doc_data = []
        for doc in documents:
            metadata = doc.get('metadata', {})
            sections = doc.get('sections', [])
            
            doc_data.append({
                'chip_category': metadata.get('chip_category'),
                'document_type': metadata.get('document_type'),
                'sub_category': metadata.get('sub_category'),
                'file_name': metadata.get('file_name'),
                'language': metadata.get('language'),
                'file_size': metadata.get('file_size'),
                'section_count': len(sections),
                'table_count': len(doc.get('tables', [])),
                'keyword_count': len(doc.get('keywords', [])),
                'content_length': sum(len(section.get('content', '')) for section in sections)
            })
        
        doc_df = pd.DataFrame(doc_data)
        doc_df.to_csv(output_path / 'documents_summary.csv', index=False, encoding='utf-8')
        
        # 片段级别的数据
        section_data = []
        for doc in documents:
            metadata = doc.get('metadata', {})
            for section in doc.get('sections', []):
                section_data.append({
                    'chip_category': metadata.get('chip_category'),
                    'document_type': metadata.get('document_type'),
                    'file_name': metadata.get('file_name'),
                    'section_id': section.get('section_id'),
                    'section_type': section.get('section_type'),
                    'title': section.get('title'),
                    'content_length': len(section.get('content', '')),
                    'tag_count': len(section.get('tags', [])),
                    'level': section.get('level')
                })
        
        section_df = pd.DataFrame(section_data)
        section_df.to_csv(output_path / 'sections_summary.csv', index=False, encoding='utf-8')
        
        logger.info("CSV报告已生成")
    
    def _generate_markdown_report(self, result: AnalysisResult, output_path: Path) -> None:
        """生成Markdown报告"""
        report_content = f"""# 芯片知识库数据分析报告

## 概览
- **总文档数**: {result.total_documents}
- **总片段数**: {result.total_sections}
- **平均每文档片段数**: {result.total_sections / result.total_documents:.1f}

## 芯片类别分布
| 类别 | 文档数 | 占比 |
|------|--------|------|
"""
        
        for category, count in sorted(result.categories.items(), key=lambda x: x[1], reverse=True):
            percentage = count / result.total_documents * 100
            report_content += f"| {category} | {count} | {percentage:.1f}% |\n"
        
        report_content += f"""
## 文档类型分布
| 类型 | 文档数 | 占比 |
|------|--------|------|
"""
        
        for doc_type, count in sorted(result.document_types.items(), key=lambda x: x[1], reverse=True):
            percentage = count / result.total_documents * 100
            report_content += f"| {doc_type} | {count} | {percentage:.1f}% |\n"
        
        report_content += f"""
## 片段类型分布
| 类型 | 数量 | 占比 |
|------|------|------|
"""
        
        for section_type, count in sorted(result.section_types.items(), key=lambda x: x[1], reverse=True):
            percentage = count / result.total_sections * 100
            report_content += f"| {section_type} | {count} | {percentage:.1f}% |\n"
        
        report_content += f"""
## 内容统计
- **总字符数**: {result.content_stats['total_content_chars']:,}
- **平均文档长度**: {result.content_stats['avg_content_length']:.0f} 字符
- **最长文档**: {result.content_stats['max_content_length']:,} 字符
- **最短文档**: {result.content_stats['min_content_length']:,} 字符
- **平均片段长度**: {result.content_stats['avg_section_length']:.0f} 字符

## 表格统计
- **总表格数**: {result.table_stats['total_tables']}
- **平均每文档表格数**: {result.table_stats['avg_tables_per_doc']:.1f}
- **包含表格的文档数**: {result.table_stats['docs_with_tables']}

## 高频关键词 (前20个)
| 关键词 | 频次 |
|--------|------|
"""
        
        for keyword, count in list(result.keyword_frequency.items())[:20]:
            report_content += f"| {keyword} | {count} |\n"
        
        report_content += f"""
## 高级索引统计
- **索引构建时间**: {result.advanced_index.index_stats.get('build_time', 0):.2f} 秒
- **总词汇数**: {result.advanced_index.index_stats.get('total_terms', 0):,}
- **总关键词数**: {result.advanced_index.index_stats.get('total_keywords', 0):,}
- **分类数量**: {result.advanced_index.index_stats.get('total_categories', 0)}
- **文档类型数量**: {result.advanced_index.index_stats.get('total_document_types', 0)}
- **元数据字段数**: {result.advanced_index.index_stats.get('total_metadata_fields', 0)}
- **平均每文档词汇数**: {result.advanced_index.index_stats.get('avg_terms_per_doc', 0):.1f}
- **平均相似连接数**: {result.advanced_index.index_stats.get('avg_similarity_connections', 0):.1f}

## 搜索功能
本知识库支持以下搜索功能：
1. **全文搜索**: 基于TF-IDF的语义搜索
2. **关键词搜索**: 精确和模糊关键词匹配
3. **分类搜索**: 按芯片类别和文档类型搜索
4. **相似度搜索**: 查找相似文档
5. **元数据搜索**: 按文档属性搜索
6. **组合搜索**: 多条件组合搜索

## 语言分布
| 语言 | 文档数 |
|------|--------|
"""
        
        for lang, count in result.languages.items():
            report_content += f"| {lang} | {count} |\n"
        
        # 保存报告
        report_file = output_path / 'analysis_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Markdown报告已保存到: {report_file}")
    
    def _save_advanced_index(self, index: AdvancedIndex, output_file: Path) -> None:
        """保存高级索引到文件"""
        # 准备可序列化的索引数据
        serializable_index = {
            'full_text_index': {term: list(doc_ids) for term, doc_ids in index.full_text_index.items()},
            'reverse_index': index.reverse_index,
            'keyword_index': {kw: list(doc_ids) for kw, doc_ids in index.keyword_index.items()},
            'category_index': {cat: list(doc_ids) for cat, doc_ids in index.category_index.items()},
            'type_index': {typ: list(doc_ids) for typ, doc_ids in index.type_index.items()},
            'hierarchical_index': {
                level: {typ: list(doc_ids) for typ, doc_ids in types.items()}
                for level, types in index.hierarchical_index.items()
            },
            'metadata_index': {
                field: {value: list(doc_ids) for value, doc_ids in values.items()}
                for field, values in index.metadata_index.items()
            },
            'similarity_index': index.similarity_index,
            'index_stats': index.index_stats
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_index, f, ensure_ascii=False, indent=2)
        
        logger.info(f"高级索引已保存到: {output_file}")
    
    def _save_search_config(self, index: AdvancedIndex, output_file: Path) -> None:
        """保存搜索引擎配置"""
        config = {
            'search_engine_info': {
                'version': '1.0',
                'created_at': datetime.now().isoformat(),
                'total_terms': len(index.full_text_index),
                'total_documents': len(index.reverse_index),
                'features': [
                    'full_text_search',
                    'keyword_search', 
                    'category_search',
                    'similarity_search',
                    'metadata_search',
                    'multi_criteria_search'
                ]
            },
            'index_statistics': index.index_stats,
            'search_capabilities': {
                'supports_fuzzy_matching': True,
                'supports_boolean_search': False,
                'supports_phrase_search': False,
                'supports_wildcard_search': False,
                'supports_similarity_ranking': True,
                'supports_faceted_search': True
            },
            'available_categories': list(index.category_index.keys()),
            'available_document_types': list(index.type_index.keys()),
            'available_metadata_fields': list(index.metadata_index.keys()),
            'top_keywords': list(index.keyword_index.keys())[:50]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"搜索引擎配置已保存到: {output_file}")

def demonstrate_advanced_search():
    """演示高级搜索功能"""
    print("=== 高级索引知识库搜索演示 ===")
    print()
    
    # 演示不同搜索类型
    search_examples = [
        {
            "type": "full_text",
            "query": "PWM 控制器",
            "description": "全文搜索: 查找包含PWM控制器的文档"
        },
        {
            "type": "keyword", 
            "query": "低功耗",
            "description": "关键词搜索: 查找标记为低功耗的文档"
        },
        {
            "type": "category",
            "query": "32F",
            "description": "分类搜索: 查找32F系列的所有文档"
        },
        {
            "type": "metadata",
            "query": "language:中文",
            "description": "元数据搜索: 查找中文文档"
        }
    ]
    
    print("支持的搜索类型:")
    for example in search_examples:
        print(f"- {example['description']}")
        print(f"  查询: {example['query']} (类型: {example['type']})")
    
    # 演示多条件搜索
    multi_criteria = {
        "query": "电机控制",
        "category": "32M15",
        "metadata": {"language": "中文"},
        "weights": {
            "full_text": 0.6,
            "category": 0.3,
            "metadata": 0.1
        }
    }
    
    print(f"\n多条件搜索示例:")
    print(f"条件: {multi_criteria}")
    
    print(f"\n索引统计信息会包含:")
    print("- 总词汇数、关键词数、分类数")
    print("- 构建时间、平均词汇密度")
    print("- 相似度连接统计")
    print("- 支持的搜索功能列表")


class DataPreprocessor:
    """数据预处理器，为向量化做准备"""
    
    def __init__(self, json_file_path: str):
        self.json_file_path = Path(json_file_path)
        self.data = self._load_json_data()
    
    def _load_json_data(self) -> Dict[str, Any]:
        """加载JSON数据"""
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def prepare_for_vectorization(self, output_dir: str, chunk_size: int = 500, overlap_size: int = 100) -> None:
        """为向量化准备数据"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"准备向量化数据，输出到: {output_path}")
        
        documents = self.data.get('documents', [])
        doc_chunks = []
        section_chunks = []
        table_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            metadata = doc.get('metadata', {})
            sections = doc.get('sections', [])
            tables = doc.get('tables', [])
            
            # 文档级别
            doc_text = doc.get('summary', '')
            if len(doc_text) > chunk_size:
                doc_text = doc_text[:chunk_size]
            
            doc_chunks.append({
                'id': f"doc_{doc_idx}",
                'type': 'document',
                'text': doc_text,
                'metadata': {
                    'chip_category': metadata.get('chip_category'),
                    'document_type': metadata.get('document_type'),
                    'sub_category': metadata.get('sub_category'),
                    'file_name': metadata.get('file_name'),
                    'language': metadata.get('language')
                }
            })
            
            # 片段级别
            for section_idx, section in enumerate(sections):
                content = section.get('content', '')
                section_chunks.append({
                    'id': f"doc_{doc_idx}_section_{section_idx}",
                    'type': 'section',
                    'text': content[:chunk_size] if len(content) > chunk_size else content,
                    'metadata': {
                        'chip_category': metadata.get('chip_category'),
                        'document_type': metadata.get('document_type'),
                        'section_type': section.get('section_type'),
                        'title': section.get('title'),
                        'level': section.get('level')
                    }
                })
        
        # 保存数据
        all_chunks = {
            'documents': doc_chunks,
            'sections': section_chunks,
            'tables': table_chunks,
            'metadata': {
                'total_doc_chunks': len(doc_chunks),
                'total_section_chunks': len(section_chunks),
                'total_table_chunks': len(table_chunks),
                'chunk_size': chunk_size
            }
        }
        
        output_file = output_path / 'vectorization_ready.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"向量化数据已保存到: {output_file}")


class AdvancedIndexBuilder:
    """高级索引构建器"""
    
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.index = AdvancedIndex()
        
    def build_full_text_index(self) -> None:
        """构建全文搜索索引"""
        logger.info("构建全文搜索索引...")
        
        for doc_idx, doc in enumerate(self.documents):
            doc_id = f"doc_{doc_idx}"
            
            # 收集文档所有文本
            all_text = []
            
            # 文档摘要
            if doc.get('summary'):
                all_text.append(doc['summary'])
            
            # 文档标题
            metadata = doc.get('metadata', {})
            if metadata.get('file_name'):
                all_text.append(metadata['file_name'])
            
            # 所有章节内容
            for section in doc.get('sections', []):
                if section.get('title'):
                    all_text.append(section['title'])
                if section.get('content'):
                    all_text.append(section['content'])
            
            # 简单的词汇提取
            combined_text = ' '.join(all_text)
            terms = re.findall(r'\w+', combined_text.lower())
            
            # 建立正向和反向索引
            self.index.reverse_index[doc_id] = terms
            
            for term in terms:
                if term not in self.index.full_text_index:
                    self.index.full_text_index[term] = set()
                self.index.full_text_index[term].add(doc_id)
        
        logger.info(f"全文索引构建完成: {len(self.index.full_text_index)} 个词汇")
    
    def build_keyword_index(self) -> None:
        """构建关键词索引"""
        logger.info("构建关键词索引...")
        
        for doc_idx, doc in enumerate(self.documents):
            doc_id = f"doc_{doc_idx}"
            metadata = doc.get('metadata', {})
            
            # 索引显式关键词
            for keyword in doc.get('keywords', []):
                if keyword not in self.index.keyword_index:
                    self.index.keyword_index[keyword] = set()
                self.index.keyword_index[keyword].add(doc_id)
            
            # 索引分类信息
            chip_category = metadata.get('chip_category')
            if chip_category:
                if chip_category not in self.index.category_index:
                    self.index.category_index[chip_category] = set()
                self.index.category_index[chip_category].add(doc_id)
            
            # 索引文档类型
            doc_type = metadata.get('document_type')
            if doc_type:
                if doc_type not in self.index.type_index:
                    self.index.type_index[doc_type] = set()
                self.index.type_index[doc_type].add(doc_id)
        
        logger.info(f"关键词索引构建完成")
    
    def build_all_indexes(self) -> AdvancedIndex:
        """构建所有索引"""
        start_time = datetime.now()
        logger.info("开始构建高级索引...")
        
        self.build_full_text_index()
        self.build_keyword_index()
        
        # 计算统计信息
        self.index.index_stats = {
            'build_time': (datetime.now() - start_time).total_seconds(),
            'total_terms': len(self.index.full_text_index),
            'total_keywords': len(self.index.keyword_index),
            'total_categories': len(self.index.category_index),
            'total_document_types': len(self.index.type_index)
        }
        
        logger.info(f"高级索引构建完成，耗时: {self.index.index_stats['build_time']:.2f}秒")
        return self.index


class DataPreprocessor:
    """数据预处理器，为向量化做准备"""
    
    def __init__(self, json_file_path: str):
        self.json_file_path = Path(json_file_path)
        self.data = self._load_json_data()
    
    def _load_json_data(self) -> Dict[str, Any]:
        """加载JSON数据"""
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def prepare_for_vectorization(self, output_dir: str, chunk_size: int = 500, overlap_size: int = 100) -> None:
        """为向量化准备数据，支持重叠分块"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"准备向量化数据，输出到: {output_path}")
        logger.info(f"分块参数: chunk_size={chunk_size}, overlap_size={overlap_size}")
        
        documents = self.data.get('documents', [])
        
        # 1. 创建文档级别的数据
        doc_chunks = []
        
        # 2. 创建片段级别的数据
        section_chunks = []
        
        # 3. 创建表格级别的数据
        table_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            metadata = doc.get('metadata', {})
            sections = doc.get('sections', [])
            tables = doc.get('tables', [])
            
            # 文档级别
            doc_text = doc.get('summary', '')
            if len(doc_text) > chunk_size:
                doc_text = doc_text[:chunk_size]
            
            doc_chunks.append({
                'id': f"doc_{doc_idx}",
                'type': 'document',
                'text': doc_text,
                'metadata': {
                    'chip_category': metadata.get('chip_category'),
                    'document_type': metadata.get('document_type'),
                    'sub_category': metadata.get('sub_category'),
                    'file_name': metadata.get('file_name'),
                    'language': metadata.get('language')
                },
                'tags': [
                    metadata.get('chip_category', ''),
                    metadata.get('document_type', ''),
                    metadata.get('sub_category', '')
                ]
            })
            
            # 片段级别
            for section_idx, section in enumerate(sections):
                content = section.get('content', '')
                
                # 分块处理长内容（支持重叠）
                if len(content) > chunk_size:
                    chunks = self._split_content_with_overlap(content, chunk_size, overlap_size)
                    for chunk_idx, chunk_text in enumerate(chunks):
                        section_chunks.append({
                            'id': f"doc_{doc_idx}_section_{section_idx}_chunk_{chunk_idx}",
                            'type': 'section',
                            'text': chunk_text,
                            'metadata': {
                                'chip_category': metadata.get('chip_category'),
                                'document_type': metadata.get('document_type'),
                                'section_type': section.get('section_type'),
                                'title': section.get('title'),
                                'level': section.get('level')
                            },
                            'tags': section.get('tags', [])
                        })
                else:
                    section_chunks.append({
                        'id': f"doc_{doc_idx}_section_{section_idx}",
                        'type': 'section',
                        'text': content,
                        'metadata': {
                            'chip_category': metadata.get('chip_category'),
                            'document_type': metadata.get('document_type'),
                            'section_type': section.get('section_type'),
                            'title': section.get('title'),
                            'level': section.get('level')
                        },
                        'tags': section.get('tags', [])
                    })
            
            # 表格级别
            for table_idx, table in enumerate(tables):
                table_text = self._table_to_text(table)
                table_chunks.append({
                    'id': f"doc_{doc_idx}_table_{table_idx}",
                    'type': 'table',
                    'text': table_text,
                    'metadata': {
                        'chip_category': metadata.get('chip_category'),
                        'document_type': metadata.get('document_type'),
                        'headers': table.get('headers', []),
                        'row_count': table.get('total_rows', 0)
                    },
                    'tags': [metadata.get('chip_category', ''), 'table']
                })
        
        # 保存数据
        all_chunks = {
            'documents': doc_chunks,
            'sections': section_chunks,
            'tables': table_chunks,
            'metadata': {
                'total_doc_chunks': len(doc_chunks),
                'total_section_chunks': len(section_chunks),
                'total_table_chunks': len(table_chunks),
                'chunk_size': chunk_size,
                'overlap_size': overlap_size
            }
        }
        
        output_file = output_path / 'vectorization_ready.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"向量化数据已保存到: {output_file}")
        
        # 保存分离的数据文件
        for chunk_type, chunks in [('documents', doc_chunks), ('sections', section_chunks), ('tables', table_chunks)]:
            chunk_file = output_path / f'{chunk_type}_chunks.json'
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        logger.info("数据预处理完成")
    
    def _split_content(self, content: str, chunk_size: int) -> List[str]:
        """分割长内容（简单版本，不支持重叠）"""
        chunks = []
        
        # 按段落分割
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_content_with_overlap(self, content: str, chunk_size: int, overlap_size: int) -> List[str]:
        """分割长内容，支持重叠机制"""
        if overlap_size >= chunk_size:
            logger.warning(f"重叠大小 ({overlap_size}) 不应大于或等于分块大小 ({chunk_size})，自动调整为 {chunk_size // 4}")
            overlap_size = chunk_size // 4
        
        # 如果重叠大小为0，使用原有的简单分块方法
        if overlap_size == 0:
            return self._split_content(content, chunk_size)
        
        chunks = []
        content_length = len(content)
        
        # 如果内容长度小于chunk_size，直接返回
        if content_length <= chunk_size:
            return [content.strip()] if content.strip() else []
        
        # 计算步长（chunk_size - overlap_size）
        step_size = chunk_size - overlap_size
        if step_size <= 0:
            step_size = chunk_size // 2  # 确保步长为正数
        
        # 按段落分割内容，保持段落完整性
        paragraphs = content.split('\n\n')
        
        # 重新组合文本，记录每个字符的段落边界
        full_text = ""
        paragraph_boundaries = []
        
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                start_pos = len(full_text)
                full_text += paragraph.strip()
                end_pos = len(full_text)
                paragraph_boundaries.append((start_pos, end_pos))
                
                if i < len(paragraphs) - 1:  # 不是最后一个段落
                    full_text += "\n\n"
        
        if not full_text.strip():
            return []
        
        # 滑动窗口分块
        start_pos = 0
        chunk_count = 0
        max_chunks = 1000  # 防止无限循环的安全限制
        
        while start_pos < len(full_text) and chunk_count < max_chunks:
            end_pos = min(start_pos + chunk_size, len(full_text))
            
            # 尝试在段落边界处切分
            best_end = end_pos
            for boundary_start, boundary_end in paragraph_boundaries:
                # 如果段落结束位置在我们的目标范围内，优先在此处切分
                if start_pos < boundary_end <= end_pos and boundary_end > start_pos + chunk_size * 0.5:
                    best_end = boundary_end
                    break
            
            # 提取块
            chunk_text = full_text[start_pos:best_end].strip()
            
            if chunk_text:
                chunks.append(chunk_text)
                chunk_count += 1
            
            # 计算下一个起始位置
            if best_end >= len(full_text):
                break
            
            # 确保有进展，防止无限循环
            next_start = start_pos + step_size
            if next_start <= start_pos:
                next_start = start_pos + 1
            
            start_pos = next_start
        
        # 如果没有生成任何块（内容太短），直接返回原内容
        if not chunks and content.strip():
            chunks.append(content.strip())
        
        # 记录分块统计信息
        avg_chunk_size = sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0
        logger.debug(f"内容长度: {len(content)}, 生成块数: {len(chunks)}, 平均块大小: {avg_chunk_size:.1f}")
        
        # 验证重叠效果
        if len(chunks) > 1 and overlap_size > 0:
            overlap_found = False
            for i in range(len(chunks) - 1):
                current_chunk = chunks[i]
                next_chunk = chunks[i + 1]
                # 检查是否有重叠内容（简单检查后面部分是否在下一个块的前面部分）
                overlap_text = current_chunk[-min(overlap_size, len(current_chunk)//2):]
                if overlap_text in next_chunk[:len(next_chunk)//2]:
                    overlap_found = True
                    break
            
            if overlap_found:
                logger.debug(f"成功创建重叠分块，重叠大小: {overlap_size}")
            else:
                logger.debug("未检测到明显重叠，可能由于段落边界限制")
        
        return chunks
    
    def _table_to_text(self, table: Dict) -> str:
        """将表格转换为文本"""
        headers = table.get('headers', [])
        rows = table.get('rows', [])
        
        text_parts = []
        
        # 添加表头
        if headers:
            text_parts.append("表头: " + " | ".join(headers))
        
        # 添加数据行
        for row_idx, row in enumerate(rows[:5]):  # 只取前5行
            row_text = " | ".join(str(cell) for cell in row)
            text_parts.append(f"行{row_idx + 1}: {row_text}")
        
        if len(rows) > 5:
            text_parts.append(f"... (共{len(rows)}行数据)")
        
        return "\n".join(text_parts)


class AdvancedIndexBuilder:
    """高级索引构建器"""
    
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.index = AdvancedIndex()
        self.chinese_stopwords = self._load_chinese_stopwords()
        
    def _load_chinese_stopwords(self) -> Set[str]:
        """加载中文停用词"""
        # 基础中文停用词列表
        stopwords = {
            '的', '了', '是', '在', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看',
            '好', '自己', '这', '那', '么', '于', '把', '或', '又', '可以', '为',
            '因为', '所以', '但是', '如果', '这样', '还是', '比较', '非常', '已经',
            '可能', '应该', '需要', '通过', '进行', '使用', '具有', '包括', '以及',
            '等等', '相关', '主要', '基本', '一般', '特别', '尤其', '例如', '比如'
        }
        return stopwords
    
    def _extract_chinese_terms(self, text: str) -> List[str]:
        """提取中文词汇（简单版本，不依赖jieba）"""
        if not text:
            return []
        
        # 清理文本
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]+', ' ', text)
        
        # 简单的中文词汇提取（基于常见技术词汇模式）
        terms = []
        
        # 提取英文单词
        english_words = re.findall(r'[a-zA-Z]+', text)
        terms.extend([word.lower() for word in english_words if len(word) >= 2])
        
        # 提取数字
        numbers = re.findall(r'\d+', text)
        terms.extend(numbers)
        
        # 简单的中文词汇分割（基于长度）
        chinese_text = re.sub(r'[a-zA-Z0-9\s]+', '', text)
        if chinese_text:
            # 提取2-4字的中文词组
            for i in range(len(chinese_text)):
                for length in [4, 3, 2]:  # 优先长词
                    if i + length <= len(chinese_text):
                        term = chinese_text[i:i+length]
                        if len(term) >= 2 and term not in self.chinese_stopwords:
                            terms.append(term)
        
        # 去重并过滤
        unique_terms = []
        seen = set()
        for term in terms:
            if term not in seen and len(term) >= 2 and term not in self.chinese_stopwords:
                unique_terms.append(term)
                seen.add(term)
                
        return unique_terms
    
    def build_full_text_index(self) -> None:
        """构建全文搜索索引"""
        logger.info("构建全文搜索索引...")
        
        for doc_idx, doc in enumerate(self.documents):
            doc_id = f"doc_{doc_idx}"
            
            # 收集文档所有文本
            all_text = []
            
            # 文档摘要
            if doc.get('summary'):
                all_text.append(doc['summary'])
            
            # 文档标题
            metadata = doc.get('metadata', {})
            if metadata.get('file_name'):
                all_text.append(metadata['file_name'])
            
            # 所有章节内容
            for section in doc.get('sections', []):
                if section.get('title'):
                    all_text.append(section['title'])
                if section.get('content'):
                    all_text.append(section['content'])
            
            # 表格内容
            for table in doc.get('tables', []):
                if table.get('headers'):
                    all_text.append(' '.join(table['headers']))
                for row in table.get('rows', [])[:3]:  # 只索引前3行
                    all_text.append(' '.join(str(cell) for cell in row))
            
            # 提取词汇
            combined_text = ' '.join(all_text)
            terms = self._extract_chinese_terms(combined_text)
            
            # 建立正向和反向索引
            self.index.reverse_index[doc_id] = terms
            
            for term in terms:
                if term not in self.index.full_text_index:
                    self.index.full_text_index[term] = set()
                self.index.full_text_index[term].add(doc_id)
        
        logger.info(f"全文索引构建完成: {len(self.index.full_text_index)} 个词汇")
    
    def build_tfidf_index(self) -> None:
        """构建TF-IDF索引（简化版本）"""
        logger.info("构建TF-IDF索引...")
        
        # 计算词频和文档频率
        term_doc_freq = defaultdict(int)  # 词汇在多少文档中出现
        doc_term_freq = {}  # 每个文档的词频
        
        # 统计词频
        for doc_id, terms in self.index.reverse_index.items():
            doc_term_freq[doc_id] = Counter(terms)
            for term in set(terms):
                term_doc_freq[term] += 1
        
        total_docs = len(self.documents)
        
        # 计算TF-IDF向量
        for doc_id, term_freq in doc_term_freq.items():
            tfidf_vector = {}
            doc_length = sum(term_freq.values())
            
            for term, freq in term_freq.items():
                tf = freq / doc_length if doc_length > 0 else 0
                idf = math.log(total_docs / (term_doc_freq[term] + 1))
                tfidf_vector[term] = tf * idf
            
            self.index.document_vectors[doc_id] = tfidf_vector
        
        logger.info(f"TF-IDF索引构建完成: {len(self.index.document_vectors)} 个文档向量")
    
    def build_keyword_index(self) -> None:
        """构建关键词索引"""
        logger.info("构建关键词索引...")
        
        for doc_idx, doc in enumerate(self.documents):
            doc_id = f"doc_{doc_idx}"
            metadata = doc.get('metadata', {})
            
            # 索引显式关键词
            for keyword in doc.get('keywords', []):
                if keyword not in self.index.keyword_index:
                    self.index.keyword_index[keyword] = set()
                self.index.keyword_index[keyword].add(doc_id)
            
            # 索引分类信息
            chip_category = metadata.get('chip_category')
            if chip_category:
                if chip_category not in self.index.category_index:
                    self.index.category_index[chip_category] = set()
                self.index.category_index[chip_category].add(doc_id)
            
            # 索引文档类型
            doc_type = metadata.get('document_type')
            if doc_type:
                if doc_type not in self.index.type_index:
                    self.index.type_index[doc_type] = set()
                self.index.type_index[doc_type].add(doc_id)
        
        logger.info(f"关键词索引构建完成: "
                   f"{len(self.index.keyword_index)} 关键词, "
                   f"{len(self.index.category_index)} 分类, "
                   f"{len(self.index.type_index)} 文档类型")
    
    def build_hierarchical_index(self) -> None:
        """构建层次索引"""
        logger.info("构建层次索引...")
        
        for doc_idx, doc in enumerate(self.documents):
            doc_id = f"doc_{doc_idx}"
            
            for section in doc.get('sections', []):
                level = str(section.get('level', 0))
                section_type = section.get('section_type', 'unknown')
                
                if level not in self.index.hierarchical_index:
                    self.index.hierarchical_index[level] = {}
                
                if section_type not in self.index.hierarchical_index[level]:
                    self.index.hierarchical_index[level][section_type] = set()
                
                self.index.hierarchical_index[level][section_type].add(doc_id)
        
        levels = len(self.index.hierarchical_index)
        total_entries = sum(len(types) for types in self.index.hierarchical_index.values())
        logger.info(f"层次索引构建完成: {levels} 个层级, {total_entries} 个类型组合")
    
    def build_metadata_index(self) -> None:
        """构建元数据索引"""
        logger.info("构建元数据索引...")
        
        for doc_idx, doc in enumerate(self.documents):
            doc_id = f"doc_{doc_idx}"
            metadata = doc.get('metadata', {})
            
            for field, value in metadata.items():
                if value and str(value).strip():
                    field_key = f"metadata_{field}"
                    value_str = str(value).strip()
                    
                    if field_key not in self.index.metadata_index:
                        self.index.metadata_index[field_key] = {}
                    
                    if value_str not in self.index.metadata_index[field_key]:
                        self.index.metadata_index[field_key][value_str] = set()
                    
                    self.index.metadata_index[field_key][value_str].add(doc_id)
        
        logger.info(f"元数据索引构建完成: {len(self.index.metadata_index)} 个字段")
    
    def build_similarity_index(self, top_k: int = 5) -> None:
        """构建相似度索引（优化版本）"""
        logger.info("构建文档相似度索引...")
        
        doc_ids = list(self.index.document_vectors.keys())
        total_docs = len(doc_ids)
        
        # 如果文档数量太多，限制相似度计算
        if total_docs > 100:
            logger.info(f"文档数量较多({total_docs})，将使用采样方式计算相似度...")
            # 为每个文档只计算与前10个和后10个文档的相似度
            for i, doc_id in enumerate(doc_ids):
                similarities = []
                
                # 计算范围：前后各10个文档
                start_idx = max(0, i - 10)
                end_idx = min(total_docs, i + 11)
                
                for j in range(start_idx, end_idx):
                    if i != j:
                        other_doc_id = doc_ids[j]
                        similarity = self._calculate_cosine_similarity(
                            self.index.document_vectors[doc_id],
                            self.index.document_vectors[other_doc_id]
                        )
                        similarities.append((other_doc_id, similarity))
                
                # 排序并取top_k
                similarities.sort(key=lambda x: x[1], reverse=True)
                self.index.similarity_index[doc_id] = similarities[:top_k]
                
                # 每处理10个文档显示一次进度
                if (i + 1) % 10 == 0:
                    logger.info(f"相似度索引进度: {i + 1}/{total_docs}")
        else:
            # 原始的全量计算方式（用于小数据集）
            for i, doc_id in enumerate(doc_ids):
                similarities = []
                
                for j, other_doc_id in enumerate(doc_ids):
                    if i != j:
                        similarity = self._calculate_cosine_similarity(
                            self.index.document_vectors[doc_id],
                            self.index.document_vectors[other_doc_id]
                        )
                        similarities.append((other_doc_id, similarity))
                
                # 排序并取top_k
                similarities.sort(key=lambda x: x[1], reverse=True)
                self.index.similarity_index[doc_id] = similarities[:top_k]
        
        logger.info(f"相似度索引构建完成: {len(self.index.similarity_index)} 个文档")
    
    def _calculate_cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """计算余弦相似度"""
        # 获取共同词汇
        common_terms = set(vec1.keys()) & set(vec2.keys())
        
        if not common_terms:
            return 0.0
        
        # 计算点积
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        
        # 计算向量长度
        norm1 = math.sqrt(sum(val**2 for val in vec1.values()))
        norm2 = math.sqrt(sum(val**2 for val in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def build_all_indexes(self) -> AdvancedIndex:
        """构建所有索引"""
        start_time = datetime.now()
        logger.info("开始构建高级索引...")
        
        self.build_full_text_index()
        self.build_tfidf_index()
        self.build_keyword_index()
        self.build_hierarchical_index()
        self.build_metadata_index()
        self.build_similarity_index()
        
        # 计算统计信息
        self.index.index_stats = {
            'build_time': (datetime.now() - start_time).total_seconds(),
            'total_terms': len(self.index.full_text_index),
            'total_keywords': len(self.index.keyword_index),
            'total_categories': len(self.index.category_index),
            'total_document_types': len(self.index.type_index),
            'total_metadata_fields': len(self.index.metadata_index),
            'avg_terms_per_doc': sum(len(terms) for terms in self.index.reverse_index.values()) / len(self.index.reverse_index) if self.index.reverse_index else 0,
            'avg_similarity_connections': sum(len(sims) for sims in self.index.similarity_index.values()) / len(self.index.similarity_index) if self.index.similarity_index else 0
        }
        
        logger.info(f"高级索引构建完成，耗时: {self.index.index_stats['build_time']:.2f}秒")
        return self.index

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='分析JSON数据并生成高级索引')
    parser.add_argument('--input', '-i', help='输入JSON文件路径')
    parser.add_argument('--output', '-o', help='输出目录路径')
    parser.add_argument('--chunk-size', type=int, default=500, help='向量化分块大小')
    parser.add_argument('--overlap-size', type=int, default=100, help='分块重叠大小')
    parser.add_argument('--demo', action='store_true', help='运行搜索演示')
    
    args = parser.parse_args()
    
    if args.demo:
        demonstrate_advanced_search()
        return
    
    # 检查必需参数
    if not args.input or not args.output:
        parser.error("分析模式需要 --input 和 --output 参数")
    
    # 分析数据并构建高级索引
    analyzer = JsonAnalyzer(args.input)
    result = analyzer.analyze()
    analyzer.generate_report(result, args.output)
    
    # 准备向量化数据
    preprocessor = DataPreprocessor(args.input)
    preprocessor.prepare_for_vectorization(args.output, args.chunk_size, args.overlap_size)
    
    print("=" * 60)
    print("🎉 数据分析和高级索引构建完成!")
    print(f"📁 分析结果保存在: {args.output}")
    print(f"⚙️  分块参数: chunk_size={args.chunk_size}, overlap_size={args.overlap_size}")
    print()
    print("📊 高级索引统计:")
    stats = result.advanced_index.index_stats
    print(f"   - 构建时间: {stats.get('build_time', 0):.2f} 秒")
    print(f"   - 总词汇数: {stats.get('total_terms', 0):,}")
    print(f"   - 总关键词: {stats.get('total_keywords', 0):,}")
    print(f"   - 分类数量: {stats.get('total_categories', 0)}")
    print(f"   - 文档类型: {stats.get('total_document_types', 0)}")
    print(f"   - 元数据字段: {stats.get('total_metadata_fields', 0)}")
    print()
    print("🔍 支持的搜索功能:")
    print("   - 全文搜索 (TF-IDF)")
    print("   - 关键词搜索 (精确+模糊)")
    print("   - 分类搜索")
    print("   - 相似度搜索")
    print("   - 元数据搜索")
    print("   - 多条件组合搜索")
    print()
    print("📄 生成的文件:")
    print(f"   - 分析报告: {args.output}/analysis_report.md")
    print(f"   - 分析结果: {args.output}/analysis_result.json")
    print(f"   - 高级索引: {args.output}/advanced_index.json")
    print(f"   - 搜索配置: {args.output}/search_config.json")
    print(f"   - 向量化数据: {args.output}/vectorization_ready.json")
    print()
    print("💡 使用示例:")
    print("   # 创建知识库实例")
    print("   from json_analyzer import IndexedKnowledgeBase")
    print(f"   kb = IndexedKnowledgeBase('{args.input}', '{args.output}/advanced_index.json')")
    print("   ")
    print("   # 执行搜索")
    print("   results = kb.search('PWM控制器', 'full_text', 10)")
    print("   multi_results = kb.multi_search({'query': '低功耗', 'category': '32F'}, 5)")
    print("=" * 60)

if __name__ == '__main__':
    main()

class IndexedKnowledgeBase:
    """支持高级索引的知识库"""
    
    def __init__(self, json_file_path: str, index_file_path: str = None):
        self.json_file_path = Path(json_file_path)
        self.data = self._load_json_data()
        self.documents = self.data.get('documents', [])
        
        if index_file_path and Path(index_file_path).exists():
            self.advanced_index = self._load_advanced_index(index_file_path)
        else:
            logger.info("索引文件不存在，构建新索引...")
            index_builder = AdvancedIndexBuilder(self.documents)
            self.advanced_index = index_builder.build_all_indexes()
        
        # self.search_engine = AdvancedSearchEngine(self.advanced_index, self.documents)
    
    def _load_json_data(self) -> Dict[str, Any]:
        """加载JSON数据"""
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_advanced_index(self, index_file_path: str) -> AdvancedIndex:
        """从文件加载高级索引"""
        with open(index_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 重建索引对象
        index = AdvancedIndex()
        
        # 转换回set类型
        index.full_text_index = {term: set(doc_ids) for term, doc_ids in data['full_text_index'].items()}
        index.reverse_index = data['reverse_index']
        index.keyword_index = {kw: set(doc_ids) for kw, doc_ids in data['keyword_index'].items()}
        index.category_index = {cat: set(doc_ids) for cat, doc_ids in data['category_index'].items()}
        index.type_index = {typ: set(doc_ids) for typ, doc_ids in data['type_index'].items()}
        
        index.hierarchical_index = {
            level: {typ: set(doc_ids) for typ, doc_ids in types.items()}
            for level, types in data['hierarchical_index'].items()
        }
        
        index.metadata_index = {
            field: {value: set(doc_ids) for value, doc_ids in values.items()}
            for field, values in data['metadata_index'].items()
        }
        
        index.similarity_index = data['similarity_index']
        index.index_stats = data['index_stats']
        
        # 重建TF-IDF向量（简化）
        if 'document_vectors' in data:
            index.document_vectors = data['document_vectors']
        
        logger.info("高级索引加载完成")
        return index
    
    def search(self, query: str, search_type: str = "full_text", limit: int = 10) -> List[Dict[str, Any]]:
        """搜索接口"""
        return self.search_engine.search(query, search_type, limit)
    
    def multi_search(self, criteria: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """多条件搜索"""
        return self.search_engine.multi_criteria_search(criteria, limit)
    
    def get_similar_documents(self, doc_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """获取相似文档"""
        return self.search_engine.similarity_search(doc_id, limit)
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取文档"""
        return self.search_engine.doc_map.get(doc_id)
    
    def get_categories(self) -> List[str]:
        """获取所有分类"""
        return list(self.advanced_index.category_index.keys())
    
    def get_document_types(self) -> List[str]:
        """获取所有文档类型"""
        return list(self.advanced_index.type_index.keys())
    
    def get_keywords(self) -> List[str]:
        """获取所有关键词"""
        return list(self.advanced_index.keyword_index.keys())
    
    def get_index_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        return self.search_engine.get_index_stats()
    
    def export_search_results(self, results: List[Dict[str, Any]], output_file: str) -> None:
        """导出搜索结果"""
        export_data = []
        
        for result in results:
            doc = result['document']
            metadata = doc.get('metadata', {})
            
            export_data.append({
                'doc_id': result['doc_id'],
                'score': result['score'],
                'search_type': result['search_type'],
                'title': metadata.get('file_name', ''),
                'category': metadata.get('chip_category', ''),
                'document_type': metadata.get('document_type', ''),
                'language': metadata.get('language', ''),
                'section_count': len(doc.get('sections', [])),
                'table_count': len(doc.get('tables', [])),
                'keyword_count': len(doc.get('keywords', [])),
                'matched_info': {k: v for k, v in result.items() if k.startswith('matched_')}
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"搜索结果已导出到: {output_file}")

def demonstrate_advanced_search():
    """演示高级搜索功能"""
    print("=== 高级索引知识库搜索演示 ===")
    print()
    # 模拟加载知识库（需要实际的文件路径）
    # kb = IndexedKnowledgeBase
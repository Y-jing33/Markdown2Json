#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON数据分析器
分析转换后的JSON数据，为后续处理做准备
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import Counter, defaultdict
import re
import pandas as pd
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        
        result = AnalysisResult(
            total_documents=total_documents,
            total_sections=total_sections,
            categories=dict(categories),
            document_types=dict(document_types),
            section_types=dict(section_types),
            languages=dict(languages),
            keyword_frequency=dict(keyword_frequency.most_common(50)),  # 前50个关键词
            content_stats=content_stats,
            table_stats=table_stats
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
                'table_stats': analysis_result.table_stats
            }, f, ensure_ascii=False, indent=2)
        
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

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='分析JSON数据并生成报告')
    parser.add_argument('--input', '-i', required=True, help='输入JSON文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出目录路径')
    parser.add_argument('--chunk-size', type=int, default=500, help='向量化分块大小')
    parser.add_argument('--overlap-size', type=int, default=100, help='分块重叠大小')
    
    args = parser.parse_args()
    
    # 分析数据
    analyzer = JsonAnalyzer(args.input)
    result = analyzer.analyze()
    analyzer.generate_report(result, args.output)
    
    # 准备向量化数据
    preprocessor = DataPreprocessor(args.input)
    preprocessor.prepare_for_vectorization(args.output, args.chunk_size, args.overlap_size)
    
    print("数据分析和预处理完成!")
    print(f"分析结果保存在: {args.output}")
    print(f"分块参数: chunk_size={args.chunk_size}, overlap_size={args.overlap_size}")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主控制脚本
执行完整的Markdown到JSON转换和分析流程
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import json

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from md_to_json_converter import MarkdownToJsonConverter
# 注释掉pandas相关的导入，改为手动处理CSV
# from json_analyzer import JsonAnalyzer, DataPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProcessController:
    """处理控制器"""
    
    def __init__(self, input_dir: str, output_base_dir: str):
        self.input_dir = Path(input_dir)
        self.output_base_dir = Path(output_base_dir)
        
        # 创建输出目录结构
        self.json_output_dir = self.output_base_dir / 'json'
        self.analysis_output_dir = self.output_base_dir / 'analysis'
        self.vectorization_output_dir = self.output_base_dir / 'vectorization'
        
        # 确保目录存在
        for dir_path in [self.json_output_dir, self.analysis_output_dir, self.vectorization_output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def run_full_pipeline(self) -> dict:
        """运行完整的处理流程"""
        logger.info("开始执行完整的Markdown到JSON转换和分析流程")
        
        pipeline_stats = {
            'start_time': datetime.now().isoformat(),
            'stages': {}
        }
        
        try:
            # 阶段1: Markdown到JSON转换
            logger.info("=== 阶段1: Markdown到JSON转换 ===")
            conversion_stats = self._run_conversion()
            pipeline_stats['stages']['conversion'] = conversion_stats
            
            # 阶段2: 简化的数据分析（不依赖pandas）
            logger.info("=== 阶段2: 数据分析 ===")
            analysis_stats = self._run_simple_analysis()
            pipeline_stats['stages']['analysis'] = analysis_stats
            
            # 阶段3: 向量化数据准备
            logger.info("=== 阶段3: 向量化数据准备 ===")
            vectorization_stats = self._run_vectorization_prep()
            pipeline_stats['stages']['vectorization'] = vectorization_stats
            
            pipeline_stats['status'] = 'success'
            pipeline_stats['end_time'] = datetime.now().isoformat()
            
            logger.info("完整流程执行成功!")
            
        except Exception as e:
            logger.error(f"流程执行失败: {str(e)}")
            pipeline_stats['status'] = 'failed'
            pipeline_stats['error'] = str(e)
            pipeline_stats['end_time'] = datetime.now().isoformat()
        
        # 保存流程统计
        stats_file = self.output_base_dir / 'pipeline_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(pipeline_stats, f, ensure_ascii=False, indent=2)
        
        return pipeline_stats
    
    def _run_conversion(self) -> dict:
        """执行Markdown到JSON转换"""
        converter = MarkdownToJsonConverter(
            str(self.input_dir), 
            str(self.json_output_dir)
        )
        
        conversion_stats = converter.convert_all()
        
        logger.info(f"转换完成: {conversion_stats['successful_conversions']}/{conversion_stats['total_files']} 文件成功")
        
        return conversion_stats
    
    def _run_simple_analysis(self) -> dict:
        """执行简化的数据分析（不依赖pandas）"""
        json_file = self.json_output_dir / 'all_documents.json'
        
        if not json_file.exists():
            raise FileNotFoundError(f"找不到JSON文件: {json_file}")
        
        # 加载数据
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = data.get('documents', [])
        
        # 简化的分析
        analysis_stats = {
            'total_documents': len(documents),
            'total_sections': sum(len(doc.get('sections', [])) for doc in documents),
            'categories': {},
            'document_types': {},
            'section_types': {},
            'languages': {},
            'content_stats': {}
        }
        
        # 统计各种分布
        content_lengths = []
        section_lengths = []
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            sections = doc.get('sections', [])
            
            # 分类统计
            category = metadata.get('chip_category', 'Unknown')
            doc_type = metadata.get('document_type', 'Unknown')
            language = metadata.get('language', 'Unknown')
            
            analysis_stats['categories'][category] = analysis_stats['categories'].get(category, 0) + 1
            analysis_stats['document_types'][doc_type] = analysis_stats['document_types'].get(doc_type, 0) + 1
            analysis_stats['languages'][language] = analysis_stats['languages'].get(language, 0) + 1
            
            # 内容统计
            doc_content_length = sum(len(section.get('content', '')) for section in sections)
            content_lengths.append(doc_content_length)
            
            # 片段统计
            for section in sections:
                section_type = section.get('section_type', 'Unknown')
                analysis_stats['section_types'][section_type] = analysis_stats['section_types'].get(section_type, 0) + 1
                section_lengths.append(len(section.get('content', '')))
        
        # 内容统计
        if content_lengths:
            analysis_stats['content_stats'] = {
                'avg_content_length': sum(content_lengths) / len(content_lengths),
                'max_content_length': max(content_lengths),
                'min_content_length': min(content_lengths),
                'total_content_chars': sum(content_lengths)
            }
        
        if section_lengths:
            analysis_stats['content_stats']['avg_section_length'] = sum(section_lengths) / len(section_lengths)
        
        # 生成简化报告
        self._generate_simple_report(analysis_stats)
        
        # 保存分析结果
        analysis_file = self.analysis_output_dir / 'analysis_result.json'
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分析结果已保存到: {analysis_file}")
        
        return analysis_stats
    
    def _generate_simple_report(self, analysis_stats: dict) -> None:
        """生成简化的分析报告"""
        report_content = f"""# 芯片知识库数据分析报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 概览
- **总文档数**: {analysis_stats['total_documents']}
- **总片段数**: {analysis_stats['total_sections']}
- **平均每文档片段数**: {analysis_stats['total_sections'] / analysis_stats['total_documents']:.1f}

## 芯片类别分布
"""
        
        # 按数量排序
        sorted_categories = sorted(analysis_stats['categories'].items(), key=lambda x: x[1], reverse=True)
        for category, count in sorted_categories:
            percentage = count / analysis_stats['total_documents'] * 100
            report_content += f"- **{category}**: {count} 个文档 ({percentage:.1f}%)\n"
        
        report_content += f"""
## 文档类型分布
"""
        
        sorted_doc_types = sorted(analysis_stats['document_types'].items(), key=lambda x: x[1], reverse=True)
        for doc_type, count in sorted_doc_types:
            percentage = count / analysis_stats['total_documents'] * 100
            report_content += f"- **{doc_type}**: {count} 个文档 ({percentage:.1f}%)\n"
        
        report_content += f"""
## 片段类型分布
"""
        
        sorted_section_types = sorted(analysis_stats['section_types'].items(), key=lambda x: x[1], reverse=True)
        for section_type, count in sorted_section_types:
            percentage = count / analysis_stats['total_sections'] * 100
            report_content += f"- **{section_type}**: {count} 个片段 ({percentage:.1f}%)\n"
        
        if analysis_stats['content_stats']:
            content_stats = analysis_stats['content_stats']
            report_content += f"""
## 内容统计
- **总字符数**: {content_stats.get('total_content_chars', 0):,}
- **平均文档长度**: {content_stats.get('avg_content_length', 0):.0f} 字符
- **最长文档**: {content_stats.get('max_content_length', 0):,} 字符
- **最短文档**: {content_stats.get('min_content_length', 0):,} 字符
- **平均片段长度**: {content_stats.get('avg_section_length', 0):.0f} 字符
"""
        
        report_content += f"""
## 语言分布
"""
        
        for language, count in analysis_stats['languages'].items():
            report_content += f"- **{language}**: {count} 个文档\n"
        
        # 保存报告
        report_file = self.analysis_output_dir / 'analysis_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"分析报告已保存到: {report_file}")
    
    def _run_vectorization_prep(self) -> dict:
        """准备向量化数据"""
        json_file = self.json_output_dir / 'all_documents.json'
        
        if not json_file.exists():
            raise FileNotFoundError(f"找不到JSON文件: {json_file}")
        
        # 加载数据
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = data.get('documents', [])
        chunk_size = 500
        
        # 准备向量化数据
        doc_chunks = []
        section_chunks = []
        table_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            metadata = doc.get('metadata', {})
            sections = doc.get('sections', [])
            tables = doc.get('tables', [])
            
            # 文档级别的块
            doc_summary = doc.get('summary', '')
            if doc_summary:
                if len(doc_summary) > chunk_size:
                    doc_summary = doc_summary[:chunk_size]
                
                doc_chunks.append({
                    'id': f"doc_{doc_idx}",
                    'type': 'document',
                    'text': doc_summary,
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
            
            # 片段级别的块
            for section_idx, section in enumerate(sections):
                content = section.get('content', '').strip()
                if not content:
                    continue
                
                # 处理长内容
                if len(content) > chunk_size:
                    chunks = self._split_content(content, chunk_size)
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
            
            # 表格级别的块
            for table_idx, table in enumerate(tables):
                table_text = self._table_to_text(table)
                if table_text:
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
        
        # 合并所有块
        all_chunks = {
            'documents': doc_chunks,
            'sections': section_chunks,
            'tables': table_chunks,
            'metadata': {
                'total_doc_chunks': len(doc_chunks),
                'total_section_chunks': len(section_chunks),
                'total_table_chunks': len(table_chunks),
                'total_chunks': len(doc_chunks) + len(section_chunks) + len(table_chunks),
                'chunk_size': chunk_size
            }
        }
        
        # 保存向量化数据
        vectorization_file = self.vectorization_output_dir / 'vectorization_ready.json'
        with open(vectorization_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
        # 保存分离的块文件
        for chunk_type, chunks in [('documents', doc_chunks), ('sections', section_chunks), ('tables', table_chunks)]:
            chunk_file = self.vectorization_output_dir / f'{chunk_type}_chunks.json'
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        vectorization_stats = {
            'total_chunks': len(doc_chunks) + len(section_chunks) + len(table_chunks),
            'doc_chunks': len(doc_chunks),
            'section_chunks': len(section_chunks),
            'table_chunks': len(table_chunks),
            'chunk_size': chunk_size
        }
        
        logger.info(f"向量化数据已保存到: {vectorization_file}")
        logger.info(f"总共生成 {vectorization_stats['total_chunks']} 个数据块")
        
        return vectorization_stats
    
    def _split_content(self, content: str, chunk_size: int) -> list:
        """分割长内容"""
        chunks = []
        
        # 按段落分割
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # 如果还有超长的块，按字符数强制分割
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= chunk_size:
                final_chunks.append(chunk)
            else:
                # 强制分割
                for i in range(0, len(chunk), chunk_size):
                    final_chunks.append(chunk[i:i+chunk_size])
        
        return final_chunks
    
    def _table_to_text(self, table: dict) -> str:
        """将表格转换为文本"""
        headers = table.get('headers', [])
        rows = table.get('rows', [])
        
        if not headers and not rows:
            return ""
        
        text_parts = []
        
        # 添加表头
        if headers:
            text_parts.append("表头: " + " | ".join(headers))
        
        # 添加数据行（限制行数以避免过长）
        max_rows = 10
        for row_idx, row in enumerate(rows[:max_rows]):
            row_text = " | ".join(str(cell) for cell in row)
            text_parts.append(f"行{row_idx + 1}: {row_text}")
        
        if len(rows) > max_rows:
            text_parts.append(f"... (总共{len(rows)}行数据)")
        
        return "\n".join(text_parts)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='执行完整的Markdown到JSON转换和分析流程')
    parser.add_argument('--input', '-i', required=True, help='输入Markdown目录路径')
    parser.add_argument('--output', '-o', required=True, help='输出基础目录路径')
    
    args = parser.parse_args()
    
    # 检查输入目录
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入目录不存在: {input_path}")
        return 1
    
    # 运行处理流程
    controller = ProcessController(args.input, args.output)
    stats = controller.run_full_pipeline()
    
    if stats['status'] == 'success':
        print("[成功] 处理流程成功完成!")
        print(f"[统计] 统计信息:")
        
        if 'conversion' in stats['stages']:
            conv_stats = stats['stages']['conversion']
            print(f"  - 转换: {conv_stats['successful_conversions']}/{conv_stats['total_files']} 文件成功")
        
        if 'analysis' in stats['stages']:
            analysis_stats = stats['stages']['analysis']
            print(f"  - 分析: {analysis_stats['total_documents']} 个文档, {analysis_stats['total_sections']} 个片段")
        
        if 'vectorization' in stats['stages']:
            vec_stats = stats['stages']['vectorization']
            print(f"  - 向量化准备: {vec_stats['total_chunks']} 个数据块")
        
        print(f"[输出] 输出目录: {args.output}")
        print("[提示] 可以进行下一步的向量化和微调处理了!")
        
        return 0
    else:
        print("[失败] 处理流程失败!")
        print(f"错误: {stats.get('error', '未知错误')}")
        return 1

if __name__ == '__main__':
    exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown to JSON Converter
将芯片知识库的Markdown文件转换为结构化JSON格式
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import hashlib
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """文档元数据"""
    chip_category: str          # 芯片类别 (32F, 95F, etc.)
    document_type: str          # 文档类型 (Product Selection, Chip Datasheet, etc.)
    sub_category: Optional[str] # 子类别 (Touch Control, General, etc.)
    file_name: str              # 文件名
    file_path: str              # 文件路径
    file_size: int              # 文件大小
    created_time: str           # 创建时间
    content_hash: str           # 内容哈希
    language: str               # 语言 (zh-cn, en, etc.)

@dataclass
class ContentSection:
    """内容片段"""
    section_id: str             # 片段ID
    section_type: str           # 片段类型 (header, table, paragraph, list, code)
    title: Optional[str]        # 标题
    content: str                # 内容
    level: Optional[int]        # 标题级别 (1-6)
    parent_section: Optional[str] # 父片段ID
    tags: List[str]             # 标签
    
@dataclass
class Document:
    """文档对象"""
    metadata: DocumentMetadata
    sections: List[ContentSection]
    summary: str                # 文档摘要
    keywords: List[str]         # 关键词
    tables: List[Dict]          # 表格数据
    images: List[str]           # 图片路径

class MarkdownParser:
    """Markdown解析器"""
    
    def __init__(self):
        self.section_counter = 0
        
    def parse_markdown(self, content: str, metadata: DocumentMetadata) -> Document:
        """解析Markdown内容"""
        self.section_counter = 0
        sections = []
        tables = []
        keywords = set()
        
        # 按行分割内容
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # 处理标题
            if line.startswith('#'):
                # 保存之前的section
                if current_section and current_content:
                    current_section.content = '\n'.join(current_content).strip()
                    if current_section.content:
                        sections.append(current_section)
                
                # 创建新section
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                current_section = self._create_section(
                    section_type='header',
                    title=title,
                    level=level,
                    metadata=metadata
                )
                current_content = []
                keywords.add(title)
                
            # 处理表格
            elif '|' in line and line.count('|') >= 2:
                table_lines = [line]
                j = i + 1
                
                # 收集表格所有行
                while j < len(lines) and ('|' in lines[j] or lines[j].strip() == ''):
                    if '|' in lines[j]:
                        table_lines.append(lines[j])
                    j += 1
                
                if len(table_lines) >= 2:  # 至少有标题行和分隔行
                    table_data = self._parse_table(table_lines)
                    if table_data:
                        tables.append(table_data)
                        
                        # 创建表格section
                        table_section = self._create_section(
                            section_type='table',
                            title=f"表格 {len(tables)}",
                            metadata=metadata
                        )
                        table_section.content = '\n'.join(table_lines)
                        sections.append(table_section)
                        
                        # 提取表格中的关键词
                        for row in table_data.get('rows', []):
                            for cell in row:
                                if isinstance(cell, str) and len(cell) < 50:
                                    keywords.add(cell.strip())
                
                i = j - 1
                
            # 处理列表
            elif line.startswith(('-', '*', '+')):
                list_lines = [line]
                j = i + 1
                
                while j < len(lines) and (lines[j].startswith((' ', '\t', '-', '*', '+')) or lines[j].strip() == ''):
                    if lines[j].strip():
                        list_lines.append(lines[j])
                    j += 1
                
                list_section = self._create_section(
                    section_type='list',
                    title=None,
                    metadata=metadata
                )
                list_section.content = '\n'.join(list_lines)
                sections.append(list_section)
                
                i = j - 1
                
            # 处理代码块
            elif line.startswith('```'):
                code_lines = [line]
                j = i + 1
                
                while j < len(lines) and not lines[j].strip().startswith('```'):
                    code_lines.append(lines[j])
                    j += 1
                    
                if j < len(lines):  # 找到结束标记
                    code_lines.append(lines[j])
                    
                    code_section = self._create_section(
                        section_type='code',
                        title=None,
                        metadata=metadata
                    )
                    code_section.content = '\n'.join(code_lines)
                    sections.append(code_section)
                    
                i = j
                
            # 普通段落
            elif line:
                current_content.append(line)
                
            i += 1
        
        # 保存最后一个section
        if current_section and current_content:
            current_section.content = '\n'.join(current_content).strip()
            if current_section.content:
                sections.append(current_section)
        
        # 生成摘要
        summary = self._generate_summary(sections)
        
        return Document(
            metadata=metadata,
            sections=sections,
            summary=summary,
            keywords=list(keywords)[:20],  # 限制关键词数量
            tables=tables,
            images=[]  # 后续可以扩展图片处理
        )
    
    def _create_section(self, section_type: str, metadata: DocumentMetadata, 
                       title: Optional[str] = None, level: Optional[int] = None) -> ContentSection:
        """创建内容片段"""
        self.section_counter += 1
        
        # 生成标签
        tags = [metadata.chip_category, metadata.document_type]
        if metadata.sub_category:
            tags.append(metadata.sub_category)
        if title:
            tags.extend(self._extract_tags_from_title(title))
        
        return ContentSection(
            section_id=f"{metadata.chip_category}_{self.section_counter}",
            section_type=section_type,
            title=title,
            content="",
            level=level,
            parent_section=None,
            tags=tags
        )
    
    def _parse_table(self, table_lines: List[str]) -> Optional[Dict]:
        """解析表格"""
        if len(table_lines) < 2:
            return None
            
        # 解析表头
        header_line = table_lines[0].strip()
        if not header_line.startswith('|') or not header_line.endswith('|'):
            return None
            
        headers = [cell.strip() for cell in header_line[1:-1].split('|')]
        
        # 跳过分隔行
        data_lines = table_lines[2:] if len(table_lines) > 2 else []
        
        rows = []
        for line in data_lines:
            line = line.strip()
            if line.startswith('|') and line.endswith('|'):
                cells = [cell.strip() for cell in line[1:-1].split('|')]
                if len(cells) == len(headers):
                    rows.append(cells)
        
        return {
            'headers': headers,
            'rows': rows,
            'total_rows': len(rows)
        }
    
    def _extract_tags_from_title(self, title: str) -> List[str]:
        """从标题提取标签"""
        tags = []
        
        # 技术相关关键词
        tech_keywords = [
            'ADC', 'PWM', 'UART', 'SPI', 'I2C', 'GPIO', 'LCD', 'LED', 'Touch',
            'ROM', 'RAM', 'EEPROM', 'Flash', '电压', '频率', '封装', '引脚',
            '功耗', '温度', '精度', '分辨率', '通道', '定时器', '中断', '看门狗'
        ]
        
        for keyword in tech_keywords:
            if keyword.lower() in title.lower():
                tags.append(keyword)
        
        return tags
    
    def _generate_summary(self, sections: List[ContentSection]) -> str:
        """生成文档摘要"""
        # 取前几个段落作为摘要
        content_parts = []
        
        for section in sections[:3]:  # 只取前3个section
            if section.section_type in ['header', 'paragraph'] and section.content:
                content_parts.append(section.content[:200])  # 每个section最多200字符
        
        summary = ' '.join(content_parts)
        return summary[:500] + '...' if len(summary) > 500 else summary

class MarkdownToJsonConverter:
    """Markdown到JSON转换器"""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.parser = MarkdownParser()
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def convert_all(self) -> Dict[str, Any]:
        """转换所有Markdown文件"""
        logger.info(f"开始转换 {self.input_dir} 中的所有Markdown文件")
        
        conversion_stats = {
            'total_files': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'categories': {},
            'document_types': {},
            'start_time': datetime.now().isoformat(),
        }
        
        documents = []
        
        # 遍历所有芯片类别文件夹
        for category_dir in self.input_dir.iterdir():
            if not category_dir.is_dir():
                continue
                
            chip_category = self._extract_chip_category(category_dir.name)
            logger.info(f"处理芯片类别: {chip_category}")
            
            conversion_stats['categories'][chip_category] = 0
            
            # 遍历文档类型文件夹
            for doc_type_dir in category_dir.iterdir():
                if not doc_type_dir.is_dir():
                    continue
                    
                document_type = doc_type_dir.name
                if document_type not in conversion_stats['document_types']:
                    conversion_stats['document_types'][document_type] = 0
                
                # 处理子类别文件夹或直接的MD文件
                self._process_directory(
                    doc_type_dir, chip_category, document_type, 
                    documents, conversion_stats
                )
        
        # 保存所有文档
        self._save_documents(documents, conversion_stats)
        
        conversion_stats['end_time'] = datetime.now().isoformat()
        logger.info(f"转换完成: {conversion_stats['successful_conversions']}/{conversion_stats['total_files']} 文件成功")
        
        return conversion_stats
    
    def _process_directory(self, directory: Path, chip_category: str, 
                          document_type: str, documents: List[Document], 
                          stats: Dict) -> None:
        """处理目录中的文件"""
        
        for item in directory.iterdir():
            if item.is_dir():
                # 子类别文件夹
                sub_category = item.name
                for md_file in item.glob('*.md'):
                    self._process_markdown_file(
                        md_file, chip_category, document_type, 
                        sub_category, documents, stats
                    )
            elif item.suffix.lower() == '.md':
                # 直接的Markdown文件
                self._process_markdown_file(
                    item, chip_category, document_type, 
                    None, documents, stats
                )
    
    def _process_markdown_file(self, file_path: Path, chip_category: str,
                              document_type: str, sub_category: Optional[str],
                              documents: List[Document], stats: Dict) -> None:
        """处理单个Markdown文件"""
        try:
            stats['total_files'] += 1
            logger.info(f"处理文件: {file_path}")
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 创建元数据
            metadata = DocumentMetadata(
                chip_category=chip_category,
                document_type=document_type,
                sub_category=sub_category,
                file_name=file_path.name,
                file_path=str(file_path.relative_to(self.input_dir)),
                file_size=file_path.stat().st_size,
                created_time=datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                content_hash=hashlib.md5(content.encode('utf-8')).hexdigest(),
                language='zh-cn' if self._is_chinese_content(content) else 'en'
            )
            
            # 解析文档
            document = self.parser.parse_markdown(content, metadata)
            documents.append(document)
            
            stats['successful_conversions'] += 1
            stats['categories'][chip_category] += 1
            stats['document_types'][document_type] += 1
            
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
            stats['failed_conversions'] += 1
    
    def _extract_chip_category(self, dirname: str) -> str:
        """提取芯片类别"""
        # 从文件夹名称中提取芯片类别
        patterns = [
            r'Documents \(([^)]+)\)',  # Documents (95F)
            r'([^-]+)',                # 第一个部分
        ]
        
        for pattern in patterns:
            match = re.search(pattern, dirname)
            if match:
                return match.group(1).strip()
        
        return dirname
    
    def _is_chinese_content(self, content: str) -> bool:
        """检测是否为中文内容"""
        chinese_chars = len([c for c in content if '\u4e00' <= c <= '\u9fff'])
        return chinese_chars > len(content) * 0.1
    
    def _save_documents(self, documents: List[Document], stats: Dict) -> None:
        """保存文档到JSON文件"""
        
        # 保存所有文档到一个文件
        all_docs_data = {
            'metadata': {
                'total_documents': len(documents),
                'conversion_time': datetime.now().isoformat(),
                'statistics': stats
            },
            'documents': [asdict(doc) for doc in documents]
        }
        
        all_docs_file = self.output_dir / 'all_documents.json'
        with open(all_docs_file, 'w', encoding='utf-8') as f:
            json.dump(all_docs_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"所有文档已保存到: {all_docs_file}")
        
        # 按芯片类别分别保存
        category_docs = {}
        for doc in documents:
            category = doc.metadata.chip_category
            if category not in category_docs:
                category_docs[category] = []
            category_docs[category].append(doc)
        
        for category, docs in category_docs.items():
            category_data = {
                'metadata': {
                    'chip_category': category,
                    'document_count': len(docs),
                    'conversion_time': datetime.now().isoformat()
                },
                'documents': [asdict(doc) for doc in docs]
            }
            
            category_file = self.output_dir / f'{category}_documents.json'
            with open(category_file, 'w', encoding='utf-8') as f:
                json.dump(category_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"{category} 类别文档已保存到: {category_file}")
        
        # 保存索引文件
        index_data = {
            'total_documents': len(documents),
            'categories': list(category_docs.keys()),
            'document_types': list(set(doc.metadata.document_type for doc in documents)),
            'files': {
                'all_documents': 'all_documents.json',
                'by_category': {cat: f'{cat}_documents.json' for cat in category_docs.keys()}
            },
            'statistics': stats
        }
        
        index_file = self.output_dir / 'index.json'
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"索引文件已保存到: {index_file}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='将Markdown文件转换为JSON格式')
    parser.add_argument('--input', '-i', required=True, help='输入目录路径')
    parser.add_argument('--output', '-o', required=True, help='输出目录路径')
    
    args = parser.parse_args()
    
    converter = MarkdownToJsonConverter(args.input, args.output)
    stats = converter.convert_all()
    
    print(f"转换完成!")
    print(f"总文件数: {stats['total_files']}")
    print(f"成功转换: {stats['successful_conversions']}")
    print(f"转换失败: {stats['failed_conversions']}")
    print(f"输出目录: {args.output}")

if __name__ == '__main__':
    main()

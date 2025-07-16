#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语义搜索模块
基于高质量embedding的中文语义搜索引擎
"""

import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import re
import jieba
import jieba.analyse
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import heapq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """搜索结果"""
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    chunk_type: str  # document, section, table
    
    def __lt__(self, other):
        return self.score < other.score

@dataclass
class SearchConfig:
    """搜索配置"""
    top_k: int = 10
    score_threshold: float = 0.3
    enable_rerank: bool = True
    weight_title: float = 1.5  # 标题权重
    weight_content: float = 1.0  # 内容权重
    weight_metadata: float = 0.8  # 元数据权重
    enable_keyword_boost: bool = True
    keyword_boost_factor: float = 1.2

class SemanticSearchEngine:
    """语义搜索引擎"""
    
    def __init__(self, embedding_dir: str, config: SearchConfig = None):
        self.embedding_dir = Path(embedding_dir)
        self.config = config or SearchConfig()
        
        # 加载embedding数据
        self.embeddings = {}
        self.metadata = {}
        self.embedding_config = {}
        
        self._load_embeddings()
        
        # 初始化查询处理器
        self.query_processor = QueryProcessor()
        
        # 初始化重排序器
        if self.config.enable_rerank:
            self.reranker = SemanticReranker()
    
    def _load_embeddings(self):
        """加载embedding数据"""
        logger.info(f"从 {self.embedding_dir} 加载embedding数据")
        
        # 加载配置
        config_file = self.embedding_dir / "embedding_config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                self.embedding_config = json.load(f)
            logger.info(f"加载embedding配置: {self.embedding_config}")
        
        # 加载各种类型的embedding
        chunk_types = ['documents', 'sections', 'tables']
        
        for chunk_type in chunk_types:
            embedding_file = self.embedding_dir / f"{chunk_type}_embeddings.npy"
            metadata_file = self.embedding_dir / f"{chunk_type}_metadata.json"
            
            if embedding_file.exists() and metadata_file.exists():
                # 加载embedding矩阵
                embeddings = np.load(embedding_file)
                
                # 加载元数据
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                self.embeddings[chunk_type] = embeddings
                self.metadata[chunk_type] = metadata
                
                logger.info(f"加载 {chunk_type}: {embeddings.shape[0]} 个embedding，维度 {embeddings.shape[1]}")
            else:
                logger.warning(f"未找到 {chunk_type} 的embedding文件")
        
        if not self.embeddings:
            raise FileNotFoundError(f"未在 {self.embedding_dir} 找到有效的embedding文件")
    
    def search(self, query: str, chunk_types: List[str] = None, **filters) -> List[SearchResult]:
        """执行语义搜索"""
        if not query.strip():
            return []
        
        # 默认搜索所有类型
        if chunk_types is None:
            chunk_types = list(self.embeddings.keys())
        
        logger.info(f"搜索查询: '{query}' 在类型: {chunk_types}")
        
        # 1. 查询预处理和embedding生成
        query_embedding = self.query_processor.process_query(query, self.embedding_config)
        
        if query_embedding is None:
            logger.error("无法生成查询embedding")
            return []
        
        # 2. 在各个数据类型中搜索
        all_results = []
        
        for chunk_type in chunk_types:
            if chunk_type not in self.embeddings:
                continue
            
            # 计算相似度
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                self.embeddings[chunk_type]
            )[0]
            
            # 创建初始结果
            for i, (similarity, metadata) in enumerate(zip(similarities, self.metadata[chunk_type])):
                if similarity >= self.config.score_threshold:
                    result = SearchResult(
                        chunk_id=metadata['chunk_id'],
                        text=metadata['text'],
                        score=similarity,
                        metadata=metadata['metadata'],
                        chunk_type=chunk_type
                    )
                    all_results.append(result)
        
        # 3. 应用过滤器
        if filters:
            all_results = self._apply_filters(all_results, filters)
        
        # 4. 关键词增强
        if self.config.enable_keyword_boost:
            all_results = self._apply_keyword_boost(all_results, query)
        
        # 5. 重排序
        if self.config.enable_rerank and hasattr(self, 'reranker'):
            all_results = self.reranker.rerank(query, all_results, self.config)
        
        # 6. 排序并返回top-k结果
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"找到 {len(all_results)} 个相关结果")
        return all_results[:self.config.top_k]
    
    def _apply_filters(self, results: List[SearchResult], filters: Dict[str, Any]) -> List[SearchResult]:
        """应用搜索过滤器"""
        filtered_results = []
        
        for result in results:
            include = True
            
            # 检查每个过滤条件
            for filter_key, filter_value in filters.items():
                if filter_key in result.metadata:
                    if isinstance(filter_value, list):
                        if result.metadata[filter_key] not in filter_value:
                            include = False
                            break
                    else:
                        if result.metadata[filter_key] != filter_value:
                            include = False
                            break
            
            if include:
                filtered_results.append(result)
        
        return filtered_results
    
    def _apply_keyword_boost(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """应用关键词增强"""
        query_keywords = set(jieba.cut(query.lower()))
        
        for result in results:
            text_keywords = set(jieba.cut(result.text.lower()))
            
            # 计算关键词重叠度
            overlap = len(query_keywords & text_keywords)
            total_keywords = len(query_keywords)
            
            if total_keywords > 0:
                keyword_score = overlap / total_keywords
                if keyword_score > 0:
                    result.score *= (1 + keyword_score * self.config.keyword_boost_factor)
        
        return results
    
    def get_similar_chunks(self, chunk_id: str, chunk_type: str, top_k: int = 5) -> List[SearchResult]:
        """获取相似的数据块"""
        if chunk_type not in self.embeddings:
            return []
        
        # 找到目标chunk的索引
        target_idx = None
        for i, metadata in enumerate(self.metadata[chunk_type]):
            if metadata['chunk_id'] == chunk_id:
                target_idx = i
                break
        
        if target_idx is None:
            logger.warning(f"未找到chunk: {chunk_id}")
            return []
        
        # 计算与其他chunk的相似度
        target_embedding = self.embeddings[chunk_type][target_idx].reshape(1, -1)
        similarities = cosine_similarity(target_embedding, self.embeddings[chunk_type])[0]
        
        # 获取最相似的chunk（排除自己）
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        results = []
        for idx in similar_indices:
            metadata = self.metadata[chunk_type][idx]
            result = SearchResult(
                chunk_id=metadata['chunk_id'],
                text=metadata['text'],
                score=similarities[idx],
                metadata=metadata['metadata'],
                chunk_type=chunk_type
            )
            results.append(result)
        
        return results

class QueryProcessor:
    """查询处理器"""
    
    def __init__(self):
        # 尝试导入embedding生成器
        try:
            from enhanced_vectorizer import EnhancedEmbeddingGenerator, EmbeddingConfig
            self.embedding_generator = None
            self.EmbeddingConfig = EmbeddingConfig
            self.EnhancedEmbeddingGenerator = EnhancedEmbeddingGenerator
        except ImportError:
            logger.warning("无法导入增强向量化模块")
            self.embedding_generator = None
    
    def process_query(self, query: str, embedding_config: Dict[str, Any]) -> Optional[np.ndarray]:
        """处理查询并生成embedding"""
        if not query.strip():
            return None
        
        # 查询预处理
        processed_query = self._preprocess_query(query)
        
        if not processed_query:
            return None
        
        # 生成embedding
        return self._generate_query_embedding(processed_query, embedding_config)
    
    def _preprocess_query(self, query: str) -> str:
        """预处理查询文本"""
        # 清理查询文本
        query = re.sub(r'[^\u4e00-\u9fa5\w\s]', ' ', query)
        query = re.sub(r'\s+', ' ', query).strip()
        
        # 查询扩展（可以添加同义词等）
        expanded_query = self._expand_query(query)
        
        return expanded_query
    
    def _expand_query(self, query: str) -> str:
        """查询扩展"""
        # 简单的同义词替换（可以扩展为更复杂的查询扩展）
        synonyms = {
            'MCU': ['微控制器', '单片机', '芯片'],
            '驱动': ['控制', '驱动器'],
            '传感器': ['感测器', '检测器'],
            '通信': ['通讯', '连接'],
            '低功耗': ['省电', '节能']
        }
        
        expanded_terms = [query]
        
        for term, syns in synonyms.items():
            if term in query:
                for syn in syns:
                    if syn not in query:
                        expanded_terms.append(syn)
        
        return ' '.join(expanded_terms)
    
    def _generate_query_embedding(self, query: str, embedding_config: Dict[str, Any]) -> Optional[np.ndarray]:
        """生成查询embedding"""
        if not self.embedding_generator:
            # 创建embedding生成器
            try:
                config = self.EmbeddingConfig(
                    model_name=embedding_config.get('model_name', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
                    model_type=embedding_config.get('model_type', 'sentence_transformers'),
                    device=embedding_config.get('device', 'cpu'),
                    output_dimension=embedding_config.get('output_dimension', 384)
                )
                self.embedding_generator = self.EnhancedEmbeddingGenerator(config)
            except Exception as e:
                logger.error(f"无法创建embedding生成器: {e}")
                return None
        
        try:
            embeddings = self.embedding_generator.generate_embeddings([query])
            if embeddings.size > 0:
                return embeddings[0]
        except Exception as e:
            logger.error(f"生成查询embedding失败: {e}")
        
        return None

class SemanticReranker:
    """语义重排序器"""
    
    def rerank(self, query: str, results: List[SearchResult], config: SearchConfig) -> List[SearchResult]:
        """重新排序搜索结果"""
        if not results:
            return results
        
        # 1. 基于文本类型的权重调整
        self._apply_type_weights(results, config)
        
        # 2. 基于元数据的相关性增强
        self._apply_metadata_boost(results, query, config)
        
        # 3. 基于文本长度的平衡
        self._apply_length_normalization(results)
        
        return results
    
    def _apply_type_weights(self, results: List[SearchResult], config: SearchConfig):
        """应用文本类型权重"""
        type_weights = {
            'documents': 1.0,
            'sections': 1.1,  # 章节通常更具体
            'tables': 0.9     # 表格信息密度高但可能不够详细
        }
        
        for result in results:
            weight = type_weights.get(result.chunk_type, 1.0)
            result.score *= weight
    
    def _apply_metadata_boost(self, results: List[SearchResult], query: str, config: SearchConfig):
        """基于元数据的相关性增强"""
        query_lower = query.lower()
        
        for result in results:
            boost = 1.0
            
            # 检查文件名匹配
            file_name = result.metadata.get('file_name', '').lower()
            if any(word in file_name for word in jieba.cut(query_lower)):
                boost *= 1.1
            
            # 检查芯片类别匹配
            chip_category = result.metadata.get('chip_category', '').lower()
            if chip_category and chip_category in query_lower:
                boost *= 1.2
            
            # 检查文档类型匹配
            doc_type = result.metadata.get('document_type', '').lower()
            doc_type_keywords = {
                'datasheet': ['数据手册', '规格', '参数'],
                'user manual': ['用户手册', '使用说明', '操作指南'],
                'application': ['应用', '实例', '案例'],
                'troubleshooting': ['故障', '问题', '排除', '解决']
            }
            
            for dtype, keywords in doc_type_keywords.items():
                if dtype in doc_type:
                    if any(keyword in query_lower for keyword in keywords):
                        boost *= 1.15
                        break
            
            result.score *= boost
    
    def _apply_length_normalization(self, results: List[SearchResult]):
        """应用长度归一化"""
        if not results:
            return
        
        # 计算文本长度统计
        lengths = [len(result.text) for result in results]
        avg_length = sum(lengths) / len(lengths)
        
        for result in results:
            text_length = len(result.text)
            
            # 对过短或过长的文本进行轻微惩罚
            if text_length < avg_length * 0.3:
                result.score *= 0.95  # 过短文本轻微惩罚
            elif text_length > avg_length * 3:
                result.score *= 0.98  # 过长文本轻微惩罚

class AdvancedSearchInterface:
    """高级搜索接口"""
    
    def __init__(self, embedding_dir: str):
        self.search_engine = SemanticSearchEngine(embedding_dir)
    
    def search_by_category(self, query: str, category: str, top_k: int = 10) -> List[SearchResult]:
        """按类别搜索"""
        config = SearchConfig(top_k=top_k)
        return self.search_engine.search(
            query,
            chunk_types=['documents', 'sections'],
            chip_category=category
        )
    
    def search_by_document_type(self, query: str, doc_type: str, top_k: int = 10) -> List[SearchResult]:
        """按文档类型搜索"""
        config = SearchConfig(top_k=top_k)
        return self.search_engine.search(
            query,
            chunk_types=['documents', 'sections'],
            document_type=doc_type
        )
    
    def fuzzy_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """模糊搜索（降低阈值）"""
        config = SearchConfig(top_k=top_k, score_threshold=0.1)
        self.search_engine.config = config
        return self.search_engine.search(query)
    
    def precise_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """精确搜索（提高阈值）"""
        config = SearchConfig(top_k=top_k, score_threshold=0.5)
        self.search_engine.config = config
        return self.search_engine.search(query)
    
    def multi_modal_search(self, query: str, include_tables: bool = True, top_k: int = 10) -> List[SearchResult]:
        """多模态搜索"""
        chunk_types = ['documents', 'sections']
        if include_tables:
            chunk_types.append('tables')
        
        return self.search_engine.search(query, chunk_types=chunk_types)

def main():
    """主函数 - 演示搜索功能"""
    import argparse
    
    parser = argparse.ArgumentParser(description='语义搜索演示')
    parser.add_argument('--embedding-dir', required=True, help='embedding数据目录')
    parser.add_argument('--query', required=True, help='搜索查询')
    parser.add_argument('--top-k', type=int, default=5, help='返回结果数量')
    parser.add_argument('--category', help='限制搜索的芯片类别')
    parser.add_argument('--doc-type', help='限制搜索的文档类型')
    
    args = parser.parse_args()
    
    try:
        # 创建搜索接口
        search_interface = AdvancedSearchInterface(args.embedding_dir)
        
        # 执行搜索
        if args.category:
            results = search_interface.search_by_category(args.query, args.category, args.top_k)
        elif args.doc_type:
            results = search_interface.search_by_document_type(args.query, args.doc_type, args.top_k)
        else:
            results = search_interface.multi_modal_search(args.query, top_k=args.top_k)
        
        # 显示结果
        print(f"\n搜索查询: '{args.query}'")
        print(f"找到 {len(results)} 个相关结果:\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. [{result.chunk_type}] 相似度: {result.score:.3f}")
            print(f"   ID: {result.chunk_id}")
            print(f"   文本: {result.text[:100]}...")
            if result.metadata:
                print(f"   元数据: {result.metadata}")
            print()
    
    except Exception as e:
        logger.error(f"搜索过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

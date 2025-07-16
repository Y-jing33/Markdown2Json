#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的向量化模块
使用预训练模型生成高质量中文语义embedding
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
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # 尝试导入Sentence-BERT
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("Sentence-Transformers 已安装")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence-Transformers 未安装，将使用备用方案")

try:
    # 尝试导入transformers
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers 已安装")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers 未安装，将使用TF-IDF方案")

@dataclass
class EmbeddingConfig:
    """向量化配置"""
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_type: str = "sentence_transformers"  # sentence_transformers, transformers, tfidf
    device: str = "cpu"  # cpu, cuda
    max_sequence_length: int = 512
    batch_size: int = 32
    cache_dir: str = "./model_cache"
    output_dimension: int = 384  # 输出向量维度

@dataclass 
class ChunkEmbedding:
    """文本块embedding信息"""
    chunk_id: str
    embedding: np.ndarray
    text: str
    metadata: Dict[str, Any]
    similarity_score: Optional[float] = None

class ChineseTextPreprocessor:
    """中文文本预处理器"""
    
    def __init__(self):
        # 设置jieba
        jieba.set_dictionary('./dict.txt') if Path('./dict.txt').exists() else None
        
        # 停用词列表
        self.stopwords = self._load_stopwords()
        
        # 技术词汇词典 - 针对芯片/MCU领域
        self.tech_keywords = {
            'MCU', 'ARM', 'GPIO', 'PWM', 'ADC', 'DAC', 'SPI', 'I2C', 'UART',
            'Timer', 'DMA', 'Flash', 'SRAM', 'EEPROM', 'Cortex', 'MHz', 'GHz',
            '触摸', '电机', '驱动', '控制', '传感器', '芯片', '处理器', '微控制器',
            '寄存器', '中断', '时钟', '复位', '低功耗', '睡眠模式', '唤醒',
            '通信协议', '数据手册', '用户手册', '应用指南', '故障排除'
        }
        
        # 添加自定义词典
        for keyword in self.tech_keywords:
            jieba.add_word(keyword)
    
    def _load_stopwords(self) -> set:
        """加载停用词"""
        stopwords = set([
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '这个', '那个', '这些', '那些', '他', '她', '它', '们',
            '啊', '吧', '呢', '嘛', '哦', '呀', '啦', '哈', '呵', '嗯', '噢'
        ])
        
        # 尝试加载停用词文件
        stopword_files = ['stopwords.txt', 'cn_stopwords.txt']
        for file_path in stopword_files:
            if Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    stopwords.update(line.strip() for line in f if line.strip())
                logger.info(f"从 {file_path} 加载了额外的停用词")
                break
        
        return stopwords
    
    def preprocess_text(self, text: str) -> str:
        """预处理文本"""
        if not text:
            return ""
        
        # 1. 清理特殊字符和格式
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # 移除markdown图片
        text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)  # 移除markdown链接，保留链接文本
        text = re.sub(r'[#*`]+', '', text)  # 移除markdown格式符号
        text = re.sub(r'\s+', ' ', text)  # 合并多个空白字符
        text = re.sub(r'[^\u4e00-\u9fa5\w\s.,;:!?()[\]{}""''（）【】｛｝，。；：！？]', '', text)
        
        # 2. 移除过短的文本
        if len(text.strip()) < 5:
            return ""
        
        return text.strip()
    
    def extract_keywords(self, text: str, topk: int = 20) -> List[str]:
        """提取关键词"""
        if not text:
            return []
        
        # 使用jieba提取关键词
        keywords = jieba.analyse.extract_tags(text, topK=topk, withWeight=False)
        
        # 过滤停用词和短词
        filtered_keywords = [
            kw for kw in keywords 
            if kw not in self.stopwords and len(kw) > 1
        ]
        
        return filtered_keywords
    
    def segment_text(self, text: str) -> List[str]:
        """分词"""
        if not text:
            return []
        
        # jieba分词
        words = jieba.cut(text)
        
        # 过滤停用词和短词
        filtered_words = [
            word.strip() for word in words 
            if word.strip() and word.strip() not in self.stopwords and len(word.strip()) > 1
        ]
        
        return filtered_words

class EnhancedEmbeddingGenerator:
    """增强的embedding生成器"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.preprocessor = ChineseTextPreprocessor()
        self.model = None
        self.tokenizer = None
        
        # 初始化模型
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化embedding模型"""
        logger.info(f"初始化embedding模型: {self.config.model_type}")
        
        if self.config.model_type == "sentence_transformers" and SENTENCE_TRANSFORMERS_AVAILABLE:
            self._init_sentence_transformers()
        elif self.config.model_type == "transformers" and TRANSFORMERS_AVAILABLE:
            self._init_transformers()
        else:
            logger.warning("使用TF-IDF备用方案")
            self.config.model_type = "tfidf"
            self._init_tfidf()
    
    def _init_sentence_transformers(self):
        """初始化Sentence-BERT模型"""
        try:
            # 中文语义模型推荐列表
            chinese_models = [
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "sentence-transformers/distiluse-base-multilingual-cased",
                "sentence-transformers/all-MiniLM-L6-v2"
            ]
            
            model_name = self.config.model_name
            if model_name not in chinese_models:
                model_name = chinese_models[0]
                logger.info(f"使用推荐的中文模型: {model_name}")
            
            self.model = SentenceTransformer(
                model_name,
                cache_folder=self.config.cache_dir,
                device=self.config.device
            )
            
            # 更新输出维度
            self.config.output_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Sentence-BERT模型加载成功，输出维度: {self.config.output_dimension}")
            
        except Exception as e:
            logger.error(f"Sentence-BERT模型加载失败: {e}")
            self.config.model_type = "tfidf"
            self._init_tfidf()
    
    def _init_transformers(self):
        """初始化Transformers模型"""
        try:
            # 中文BERT模型推荐
            chinese_bert_models = [
                "bert-base-chinese",
                "hfl/chinese-bert-wwm-ext",
                "hfl/chinese-roberta-wwm-ext"
            ]
            
            model_name = self.config.model_name
            if "sentence-transformers" in model_name:
                model_name = chinese_bert_models[0]
                logger.info(f"使用中文BERT模型: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir
            )
            
            if self.config.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
            
            self.config.output_dimension = self.model.config.hidden_size
            logger.info(f"Transformers模型加载成功，输出维度: {self.config.output_dimension}")
            
        except Exception as e:
            logger.error(f"Transformers模型加载失败: {e}")
            self.config.model_type = "tfidf"
            self._init_tfidf()
    
    def _init_tfidf(self):
        """初始化TF-IDF模型"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self.model = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            tokenizer=self.preprocessor.segment_text,
            lowercase=False
        )
        self.config.output_dimension = 5000
        logger.info("TF-IDF模型初始化完成")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """生成文本embedding"""
        if not texts:
            return np.array([])
        
        # 预处理文本
        processed_texts = [self.preprocessor.preprocess_text(text) for text in texts]
        processed_texts = [t for t in processed_texts if t]  # 移除空文本
        
        if not processed_texts:
            return np.array([])
        
        if self.config.model_type == "sentence_transformers":
            return self._generate_sentence_transformers_embeddings(processed_texts)
        elif self.config.model_type == "transformers":
            return self._generate_transformers_embeddings(processed_texts)
        else:
            return self._generate_tfidf_embeddings(processed_texts)
    
    def _generate_sentence_transformers_embeddings(self, texts: List[str]) -> np.ndarray:
        """使用Sentence-BERT生成embedding"""
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Sentence-BERT embedding生成失败: {e}")
            return np.array([])
    
    def _generate_transformers_embeddings(self, texts: List[str]) -> np.ndarray:
        """使用Transformers生成embedding"""
        try:
            embeddings = []
            
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                
                # 编码文本
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_sequence_length,
                    return_tensors="pt"
                )
                
                if self.config.device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # 生成embedding
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # 使用[CLS]token的表示或平均池化
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                    
                    if self.config.device == "cuda":
                        batch_embeddings = batch_embeddings.cpu()
                    
                    embeddings.append(batch_embeddings.numpy())
            
            return np.vstack(embeddings)
            
        except Exception as e:
            logger.error(f"Transformers embedding生成失败: {e}")
            return np.array([])
    
    def _generate_tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """使用TF-IDF生成embedding"""
        try:
            if not hasattr(self.model, 'vocabulary_'):
                # 首次训练
                tfidf_matrix = self.model.fit_transform(texts)
            else:
                # 已训练，直接转换
                tfidf_matrix = self.model.transform(texts)
            
            return tfidf_matrix.toarray()
        except Exception as e:
            logger.error(f"TF-IDF embedding生成失败: {e}")
            return np.array([])

class VectorizationPipeline:
    """向量化处理流水线"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.embedding_generator = EnhancedEmbeddingGenerator(config)
        self.embeddings_cache = {}
        
    def process_vectorization_data(self, input_file: str, output_dir: str) -> Dict[str, Any]:
        """处理向量化数据"""
        logger.info(f"开始处理向量化数据: {input_file}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载向量化准备数据
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        stats = {
            'total_chunks': 0,
            'embedding_dimension': self.config.output_dimension,
            'model_type': self.config.model_type,
            'model_name': self.config.model_name,
            'processing_time': 0,
            'chunks_by_type': {}
        }
        
        start_time = datetime.now()
        
        # 按类型处理不同的数据块
        all_embeddings = {}
        
        for chunk_type in ['documents', 'sections', 'tables']:
            if chunk_type in data:
                logger.info(f"处理 {chunk_type} 类型的数据块...")
                chunks = data[chunk_type]
                
                if chunks:
                    embeddings = self._process_chunks(chunks, chunk_type)
                    all_embeddings[chunk_type] = embeddings
                    
                    stats['chunks_by_type'][chunk_type] = len(chunks)
                    stats['total_chunks'] += len(chunks)
        
        # 计算处理时间
        processing_time = (datetime.now() - start_time).total_seconds()
        stats['processing_time'] = processing_time
        
        # 保存embedding结果
        self._save_embeddings(all_embeddings, output_path, stats)
        
        logger.info(f"向量化处理完成，耗时: {processing_time:.2f}秒")
        logger.info(f"总共处理 {stats['total_chunks']} 个数据块")
        
        return stats
    
    def _process_chunks(self, chunks: List[Dict[str, Any]], chunk_type: str) -> List[ChunkEmbedding]:
        """处理数据块"""
        if not chunks:
            return []
        
        # 提取文本
        texts = []
        for chunk in chunks:
            text = chunk.get('text', '')
            if not text:
                # 尝试从其他字段构建文本
                if chunk_type == 'documents':
                    text = self._build_document_text(chunk)
                elif chunk_type == 'sections':
                    text = self._build_section_text(chunk)
                elif chunk_type == 'tables':
                    text = self._build_table_text(chunk)
            texts.append(text)
        
        # 生成embedding
        embeddings_matrix = self.embedding_generator.generate_embeddings(texts)
        
        if embeddings_matrix.size == 0:
            logger.warning(f"无法为 {chunk_type} 生成embedding")
            return []
        
        # 构建结果
        chunk_embeddings = []
        for i, chunk in enumerate(chunks):
            if i < len(embeddings_matrix):
                chunk_embedding = ChunkEmbedding(
                    chunk_id=chunk.get('id', f"{chunk_type}_{i}"),
                    embedding=embeddings_matrix[i],
                    text=texts[i],
                    metadata=chunk.get('metadata', {})
                )
                chunk_embeddings.append(chunk_embedding)
        
        return chunk_embeddings
    
    def _build_document_text(self, document: Dict[str, Any]) -> str:
        """构建文档文本"""
        text_parts = []
        
        # 标题信息
        metadata = document.get('metadata', {})
        if metadata.get('file_name'):
            text_parts.append(metadata['file_name'])
        
        # 主要文本
        if document.get('text'):
            text_parts.append(document['text'])
        
        # 标签
        if document.get('tags'):
            text_parts.append(' '.join(document['tags']))
        
        return ' '.join(text_parts)
    
    def _build_section_text(self, section: Dict[str, Any]) -> str:
        """构建章节文本"""
        text_parts = []
        
        if section.get('title'):
            text_parts.append(section['title'])
        
        if section.get('text'):
            text_parts.append(section['text'])
        
        return ' '.join(text_parts)
    
    def _build_table_text(self, table: Dict[str, Any]) -> str:
        """构建表格文本"""
        text_parts = []
        
        if table.get('caption'):
            text_parts.append(table['caption'])
        
        if table.get('text'):
            text_parts.append(table['text'])
        
        return ' '.join(text_parts)
    
    def _save_embeddings(self, embeddings: Dict[str, List[ChunkEmbedding]], output_path: Path, stats: Dict[str, Any]):
        """保存embedding结果"""
        
        # 保存每种类型的embedding
        for chunk_type, chunk_embeddings in embeddings.items():
            if not chunk_embeddings:
                continue
            
            # 保存为numpy格式
            embedding_matrix = np.array([ce.embedding for ce in chunk_embeddings])
            np.save(output_path / f"{chunk_type}_embeddings.npy", embedding_matrix)
            
            # 保存元数据
            metadata_list = []
            for ce in chunk_embeddings:
                metadata_list.append({
                    'chunk_id': ce.chunk_id,
                    'text': ce.text[:200] + '...' if len(ce.text) > 200 else ce.text,  # 截断长文本
                    'metadata': ce.metadata
                })
            
            with open(output_path / f"{chunk_type}_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, ensure_ascii=False, indent=2)
            
            logger.info(f"{chunk_type} embedding已保存: {embedding_matrix.shape}")
        
        # 保存统计信息
        with open(output_path / "embedding_stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # 保存配置信息
        config_dict = {
            'model_name': self.config.model_name,
            'model_type': self.config.model_type,
            'device': self.config.device,
            'output_dimension': self.config.output_dimension,
            'max_sequence_length': self.config.max_sequence_length,
            'batch_size': self.config.batch_size
        }
        
        with open(output_path / "embedding_config.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Embedding结果已保存到: {output_path}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='增强向量化处理')
    parser.add_argument('--input', required=True, help='输入的向量化准备数据文件')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--model-type', choices=['sentence_transformers', 'transformers', 'tfidf'], 
                       default='sentence_transformers', help='模型类型')
    parser.add_argument('--model-name', default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                       help='模型名称')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='计算设备')
    parser.add_argument('--batch-size', type=int, default=32, help='批处理大小')
    
    args = parser.parse_args()
    
    # 创建配置
    config = EmbeddingConfig(
        model_name=args.model_name,
        model_type=args.model_type,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # 创建处理流水线
    pipeline = VectorizationPipeline(config)
    
    # 处理数据
    stats = pipeline.process_vectorization_data(args.input, args.output)
    
    print(f"\n向量化处理完成!")
    print(f"- 处理的数据块数量: {stats['total_chunks']}")
    print(f"- Embedding维度: {stats['embedding_dimension']}")
    print(f"- 使用的模型: {stats['model_type']} ({stats['model_name']})")
    print(f"- 处理时间: {stats['processing_time']:.2f}秒")
    print(f"- 输出目录: {args.output}")

if __name__ == "__main__":
    main()

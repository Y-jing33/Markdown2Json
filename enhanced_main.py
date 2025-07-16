#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强向量化处理脚本
使用预训练模型生成高质量中文语义embedding并进行语义搜索
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_vectorization.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """检查依赖包"""
    logger.info("检查依赖包...")
    
    dependencies = {
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'scikit-learn': 'sklearn',
        'jieba': 'jieba',
        'sentence-transformers': 'sentence_transformers',
        'transformers': 'transformers',
        'torch': 'torch'
    }
    
    missing = []
    available = []
    
    for name, import_name in dependencies.items():
        try:
            __import__(import_name)
            available.append(name)
        except ImportError:
            missing.append(name)
    
    logger.info(f"可用包: {', '.join(available)}")
    if missing:
        logger.warning(f"缺失包: {', '.join(missing)}")
        logger.info("请运行: uv add " + " ".join(missing))
    
    return len(missing) == 0

def install_required_packages():
    """安装必需的包"""
    logger.info("安装必需的包...")
    
    import subprocess
    
    packages = [
        "sentence-transformers",
        "transformers", 
        "torch"
    ]
    
    for package in packages:
        try:
            logger.info(f"安装 {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                logger.info(f"{package} 安装成功")
            else:
                logger.error(f"{package} 安装失败: {result.stderr}")
        except Exception as e:
            logger.error(f"安装 {package} 时出错: {e}")

def run_enhanced_vectorization(input_file: str, output_dir: str, model_type: str = "sentence_transformers"):
    """运行增强向量化"""
    logger.info("=" * 60)
    logger.info("开始增强向量化处理")
    logger.info("=" * 60)
    
    try:
        from enhanced_vectorizer import VectorizationPipeline, EmbeddingConfig
        
        # 创建配置
        config = EmbeddingConfig(
            model_type=model_type,
            device="cpu",  # 默认使用CPU，如果需要GPU可以改为"cuda"
            batch_size=16  # 减小批处理大小以适应CPU
        )
        
        if model_type == "sentence_transformers":
            # 使用多语言模型
            config.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        elif model_type == "transformers":
            # 使用中文BERT模型
            config.model_name = "bert-base-chinese"
        
        logger.info(f"使用模型: {config.model_name}")
        logger.info(f"模型类型: {config.model_type}")
        logger.info(f"计算设备: {config.device}")
        
        # 创建处理流水线
        pipeline = VectorizationPipeline(config)
        
        # 处理数据
        stats = pipeline.process_vectorization_data(input_file, output_dir)
        
        logger.info("=" * 60)
        logger.info("增强向量化处理完成")
        logger.info("=" * 60)
        logger.info(f"处理统计:")
        logger.info(f"  - 总数据块: {stats['total_chunks']}")
        logger.info(f"  - Embedding维度: {stats['embedding_dimension']}")
        logger.info(f"  - 处理时间: {stats['processing_time']:.2f}秒")
        logger.info(f"  - 输出目录: {output_dir}")
        
        return stats
        
    except ImportError as e:
        logger.error(f"导入模块失败: {e}")
        logger.info("请确保已安装必要的依赖包")
        return None
    except Exception as e:
        logger.error(f"向量化处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_semantic_search_demo(embedding_dir: str):
    """运行语义搜索演示"""
    logger.info("=" * 60) 
    logger.info("语义搜索演示")
    logger.info("=" * 60)
    
    try:
        from semantic_search import AdvancedSearchInterface
        
        # 创建搜索接口
        search_interface = AdvancedSearchInterface(embedding_dir)
        
        # 预设查询示例
        demo_queries = [
            "MCU低功耗模式",
            "GPIO引脚配置",
            "定时器中断",
            "SPI通信协议",
            "ADC模数转换",
            "触摸控制芯片",
            "电机驱动应用"
        ]
        
        logger.info("运行演示查询...")
        
        for query in demo_queries:
            logger.info(f"\n搜索: '{query}'")
            try:
                results = search_interface.multi_modal_search(query, top_k=3)
                
                if results:
                    for i, result in enumerate(results, 1):
                        logger.info(f"  {i}. 相似度: {result.score:.3f} [{result.chunk_type}]")
                        logger.info(f"     文本: {result.text[:80]}...")
                else:
                    logger.info("     未找到相关结果")
                    
            except Exception as e:
                logger.error(f"查询 '{query}' 失败: {e}")
        
        logger.info("\n" + "=" * 60)
        logger.info("语义搜索演示完成")
        logger.info("=" * 60)
        
    except ImportError as e:
        logger.error(f"导入搜索模块失败: {e}")
    except Exception as e:
        logger.error(f"搜索演示失败: {e}")
        import traceback
        traceback.print_exc()

def interactive_search(embedding_dir: str):
    """交互式搜索"""
    logger.info("=" * 60)
    logger.info("交互式语义搜索")
    logger.info("=" * 60)
    
    try:
        from semantic_search import AdvancedSearchInterface
        
        search_interface = AdvancedSearchInterface(embedding_dir)
        
        print("\n欢迎使用语义搜索系统!")
        print("输入查询内容，输入 'quit' 退出")
        print("-" * 40)
        
        while True:
            try:
                query = input("\n请输入搜索查询: ").strip()
                
                if query.lower() in ['quit', 'exit', '退出', 'q']:
                    break
                
                if not query:
                    continue
                
                print(f"\n搜索: '{query}'")
                print("-" * 40)
                
                results = search_interface.multi_modal_search(query, top_k=5)
                
                if results:
                    for i, result in enumerate(results, 1):
                        print(f"\n{i}. [{result.chunk_type}] 相似度: {result.score:.3f}")
                        print(f"   ID: {result.chunk_id}")
                        print(f"   文本: {result.text[:150]}...")
                        
                        # 显示元数据
                        metadata = result.metadata
                        if metadata:
                            meta_info = []
                            if 'chip_category' in metadata:
                                meta_info.append(f"类别: {metadata['chip_category']}")
                            if 'document_type' in metadata:
                                meta_info.append(f"类型: {metadata['document_type']}")
                            if 'file_name' in metadata:
                                meta_info.append(f"文件: {metadata['file_name']}")
                            
                            if meta_info:
                                print(f"   {' | '.join(meta_info)}")
                else:
                    print("   未找到相关结果")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"   搜索出错: {e}")
        
        print("\n感谢使用!")
        
    except ImportError as e:
        logger.error(f"导入搜索模块失败: {e}")
    except Exception as e:
        logger.error(f"交互式搜索失败: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='增强向量化和语义搜索系统')
    parser.add_argument('--action', choices=['vectorize', 'search', 'demo', 'interactive', 'install'], 
                       default='vectorize', help='操作类型')
    parser.add_argument('--input', help='输入的向量化准备数据文件')
    parser.add_argument('--output', help='输出目录')
    parser.add_argument('--embedding-dir', help='embedding数据目录（用于搜索）')
    parser.add_argument('--model-type', choices=['sentence_transformers', 'transformers', 'tfidf'],
                       default='sentence_transformers', help='模型类型')
    parser.add_argument('--query', help='搜索查询')
    
    args = parser.parse_args()
    
    # 默认路径
    default_input = "./output/vectorization/vectorization_ready.json"
    default_output = "./output/enhanced_embeddings"
    
    if args.action == 'install':
        install_required_packages()
        return
    
    if args.action == 'vectorize':
        input_file = args.input or default_input
        output_dir = args.output or default_output
        
        if not Path(input_file).exists():
            logger.error(f"输入文件不存在: {input_file}")
            logger.info("请先运行主处理流程生成向量化准备数据")
            return
        
        # 检查依赖
        if not check_dependencies():
            logger.info("正在安装缺失的依赖包...")
            install_required_packages()
        
        # 运行向量化
        stats = run_enhanced_vectorization(input_file, output_dir, args.model_type)
        
        if stats:
            print(f"\n✅ 增强向量化完成!")
            print(f"📊 统计信息:")
            print(f"   - 总数据块: {stats['total_chunks']}")
            print(f"   - Embedding维度: {stats['embedding_dimension']}")
            print(f"   - 处理时间: {stats['processing_time']:.2f}秒")
            print(f"📁 输出目录: {output_dir}")
            print(f"\n💡 接下来可以使用搜索功能:")
            print(f"   python enhanced_main.py --action demo --embedding-dir {output_dir}")
    
    elif args.action == 'search':
        embedding_dir = args.embedding_dir or default_output
        
        if not Path(embedding_dir).exists():
            logger.error(f"embedding目录不存在: {embedding_dir}")
            logger.info("请先运行向量化处理")
            return
        
        if args.query:
            # 单次搜索
            from semantic_search import AdvancedSearchInterface
            search_interface = AdvancedSearchInterface(embedding_dir)
            results = search_interface.multi_modal_search(args.query, top_k=5)
            
            print(f"\n搜索结果 for '{args.query}':")
            for i, result in enumerate(results, 1):
                print(f"{i}. [{result.chunk_type}] 相似度: {result.score:.3f}")
                print(f"   {result.text[:100]}...")
        else:
            logger.error("请提供搜索查询 (--query)")
    
    elif args.action == 'demo':
        embedding_dir = args.embedding_dir or default_output
        
        if not Path(embedding_dir).exists():
            logger.error(f"embedding目录不存在: {embedding_dir}")
            return
        
        run_semantic_search_demo(embedding_dir)
    
    elif args.action == 'interactive':
        embedding_dir = args.embedding_dir or default_output
        
        if not Path(embedding_dir).exists():
            logger.error(f"embedding目录不存在: {embedding_dir}")
            return
        
        interactive_search(embedding_dir)

if __name__ == "__main__":
    main()

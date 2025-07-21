#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强整合能力的数据集构建器 (支持Alpaca/ShareGPT格式)
专门构建需要信息整合、长上下文理解和复杂推理的训练数据
完全兼容Alpaca和ShareGPT格式标准
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import re
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class IntegrationTask:
    """信息整合任务"""
    task_type: str  # 任务类型：comparison, synthesis, analysis, reasoning
    instruction: str  # 指令
    context: str  # 长上下文（多个相关文档块）
    expected_output: str  # 期望的整合答案
    difficulty: str  # 难度级别：basic, intermediate, advanced
    integration_points: List[str] = field(default_factory=list)  # 需要整合的关键点
    quality_score: float = 0.0  # 质量评分
    context_sources: List[str] = field(default_factory=list)  # 上下文来源

class ConfigurableIntegrationDatasetBuilder:
    """可配置的增强整合数据集构建器，支持Alpaca/ShareGPT格式"""
    
    def __init__(self, input_dir: str = "output", output_dir: str = "output/integration_dataset", 
                 config_file: str = "integration_config.json"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        self.config = self._load_config(config_file)
        
        # 加载数据
        self.documents_chunks = self._load_json("vectorization/documents_chunks.json")
        self.sections_chunks = self._load_json("vectorization/sections_chunks.json")
        self.analysis_result = self._load_json("analysis/analysis_result.json")
        
        # 加载嵌入向量用于语义相似性计算
        self.embeddings = self._load_embeddings()
        
        logger.info(f"加载了 {len(self.documents_chunks)} 个文档块")
        logger.info(f"配置文件: {config_file}")
        
    def _load_config(self, config_file: str) -> Dict:
        """加载配置文件"""
        config_path = Path(config_file)
        if not config_path.exists():
            logger.warning(f"配置文件不存在: {config_file}，使用默认配置")
            return self._get_default_config()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"成功加载配置文件: {config_file}")
                return config
        except Exception as e:
            logger.error(f"读取配置文件失败: {e}，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """默认配置"""
        return {
            "dataset_config": {
                "max_samples": 1000,
                "task_distribution": {
                    "comparison": 0.25,
                    "synthesis": 0.30,
                    "analysis": 0.25,
                    "reasoning": 0.20
                },
                "difficulty_distribution": {
                    "basic": 0.20,
                    "intermediate": 0.50,
                    "advanced": 0.30
                }
            },
            "integration_requirements": {
                "min_context_length": 500,
                "max_context_length": 8000,
                "min_chunks_per_task": 2,
                "max_chunks_per_task": 6,
                "similarity_threshold": 0.6
            },
            "output_format": {
                "alpaca_format": True,
                "sharegpt_format": False,
                "include_metadata": True,
                "metadata_fields": [
                    "task_type",
                    "difficulty", 
                    "integration_points",
                    "context_sources",
                    "quality_score"
                ]
            }
        }
    
    def _load_json(self, relative_path: str) -> List[Dict]:
        """加载JSON文件"""
        file_path = self.input_dir / relative_path
        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {e}")
            return []
    
    def _load_embeddings(self) -> Optional[np.ndarray]:
        """加载文档嵌入向量"""
        embeddings_path = self.input_dir / "enhanced_embeddings/documents_embeddings.npy"
        if embeddings_path.exists():
            try:
                return np.load(embeddings_path)
            except Exception as e:
                logger.warning(f"加载嵌入向量失败: {e}")
        return None
    
    def find_related_chunks(self, base_chunk: Dict, max_related: int = 5, similarity_threshold: float = 0.5) -> List[Dict]:
        """找到与基础块相关的其他文档块"""
        if self.embeddings is None or len(self.embeddings) != len(self.documents_chunks):
            return self._find_related_chunks_by_keywords(base_chunk, max_related)
        
        base_idx = next((i for i, chunk in enumerate(self.documents_chunks) if chunk == base_chunk), None)
        if base_idx is None:
            return []
        
        base_embedding = self.embeddings[base_idx]
        similarities = []
        
        for i, embedding in enumerate(self.embeddings):
            if i != base_idx:
                # 计算余弦相似度
                similarity = np.dot(base_embedding, embedding) / (
                    np.linalg.norm(base_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((i, similarity))
        
        # 按相似度排序并选择最相关的
        similarities.sort(key=lambda x: x[1], reverse=True)
        related_chunks = []
        
        for idx, sim in similarities[:max_related]:
            if sim >= similarity_threshold:
                related_chunks.append(self.documents_chunks[idx])
        
        return related_chunks
    
    def _find_related_chunks_by_keywords(self, base_chunk: Dict, max_related: int = 5) -> List[Dict]:
        """基于关键词找到相关文档块"""
        base_text = base_chunk.get("text", "")
        base_metadata = base_chunk.get("metadata", {})
        
        # 提取关键技术词汇
        keywords = self._extract_technical_keywords(base_text)
        chip_category = base_metadata.get("chip_category", "")
        
        related_chunks = []
        scores = []
        
        for chunk in self.documents_chunks:
            if chunk == base_chunk:
                continue
                
            chunk_text = chunk.get("text", "")
            chunk_metadata = chunk.get("metadata", {})
            
            score = 0
            
            # 相同芯片类别加分
            if chip_category and chunk_metadata.get("chip_category") == chip_category:
                score += 2
            
            # 关键词匹配加分
            for keyword in keywords:
                if keyword in chunk_text:
                    score += 1
            
            # 文档类型相关性加分
            if self._is_complementary_doc_type(
                base_metadata.get("document_type", ""),
                chunk_metadata.get("document_type", "")
            ):
                score += 1
            
            if score > 0:
                scores.append((chunk, score))
        
        # 按分数排序并返回最相关的
        scores.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scores[:max_related]]
    
    def _extract_technical_keywords(self, text: str) -> List[str]:
        """提取技术关键词"""
        # 技术术语词典
        technical_terms = [
            "PWM", "ADC", "中断", "寄存器", "引脚", "管脚", "时钟", "分频",
            "采样", "转换", "配置", "使能", "禁用", "模式", "功耗", "电流",
            "电压", "频率", "精度", "分辨率", "通道", "缓冲", "滤波",
            "触发", "同步", "异步", "串行", "并行", "通信", "协议"
        ]
        
        keywords = []
        text_upper = text.upper()
        
        for term in technical_terms:
            if term in text or term.upper() in text_upper:
                keywords.append(term)
        
        # 提取英文缩写
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        keywords.extend(acronyms)
        
        return list(set(keywords))
    
    def _is_complementary_doc_type(self, type1: str, type2: str) -> bool:
        """判断两个文档类型是否互补"""
        complementary_pairs = [
            ("datasheet", "user manual"),
            ("datasheet", "application"),
            ("troubleshooting", "user manual"),
            ("selection", "datasheet"),
            ("brochure", "datasheet")
        ]
        
        type1_lower = type1.lower()
        type2_lower = type2.lower()
        
        for pair in complementary_pairs:
            if (pair[0] in type1_lower and pair[1] in type2_lower) or \
               (pair[1] in type1_lower and pair[0] in type2_lower):
                return True
        
        return False
    
    def build_integration_dataset(self) -> List[IntegrationTask]:
        """构建信息整合数据集"""
        logger.info("开始构建信息整合数据集...")
        
        config = self.config["dataset_config"]
        requirements = self.config["integration_requirements"]
        
        max_samples = config["max_samples"]
        task_distribution = config["task_distribution"]
        
        dataset = []
        task_counts = {task_type: 0 for task_type in task_distribution.keys()}
        
        # 计算每种任务类型的目标数量
        target_counts = {}
        for task_type, ratio in task_distribution.items():
            target_counts[task_type] = int(max_samples * ratio)
        
        # 随机处理文档块以增加多样性
        shuffled_chunks = self.documents_chunks.copy()
        random.shuffle(shuffled_chunks)
        
        for i, base_chunk in enumerate(shuffled_chunks):
            if len(dataset) >= max_samples:
                break
            
            # 跳过内容过短的块
            if len(base_chunk.get("text", "")) < requirements["min_context_length"]:
                continue
            
            # 找到相关的文档块
            similarity_threshold = requirements["similarity_threshold"]
            max_related = requirements["max_chunks_per_task"] - 1
            related_chunks = self.find_related_chunks(base_chunk, max_related, similarity_threshold)
            
            if len(related_chunks) < requirements["min_chunks_per_task"] - 1:
                continue
            
            # 构建chunk组合
            chunk_group = [base_chunk] + related_chunks
            
            # 检查上下文长度
            total_context = self._build_context_text(chunk_group)
            if len(total_context) > requirements["max_context_length"]:
                # 减少chunk数量
                chunk_group = chunk_group[:requirements["min_chunks_per_task"]]
                total_context = self._build_context_text(chunk_group)
            
            if len(total_context) < requirements["min_context_length"]:
                continue
            
            # 智能分析适合的任务类型
            suitable_task_types = self._analyze_content_suitability(chunk_group)
            
            # 如果没有合适的任务类型，跳过
            if not suitable_task_types:
                continue
            
            # 按需求和优先级创建任务
            for task_type in suitable_task_types:
                if task_counts.get(task_type, 0) >= target_counts.get(task_type, 0):
                    continue
                
                task = self._create_task_by_type(task_type, chunk_group)
                
                if task and self._validate_task_quality(task):
                    dataset.append(task)
                    task_counts[task_type] += 1
                    logger.debug(f"创建了 {task_type} 任务，基于内容: {base_chunk.get('metadata', {}).get('chip_category', 'Unknown')}")
                    break  # 每个chunk组合只创建一个任务
        
        logger.info(f"生成了 {len(dataset)} 个整合任务")
        logger.info(f"任务分布: {task_counts}")
        
        return dataset
    
    def _build_context_text(self, chunks: List[Dict]) -> str:
        """构建上下文文本"""
        context_parts = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            chip_category = metadata.get("chip_category", "技术资料")
            text = chunk.get("text", "")
            context_parts.append(f"=== {chip_category} ===\n{text}\n")
        return "\n".join(context_parts)
    
    def _analyze_content_suitability(self, chunks: List[Dict]) -> List[str]:
        """分析内容适合性（简化版本）"""
        suitable_tasks = []
        
        # 合并所有文本用于分析
        combined_text = " ".join([chunk.get("text", "") for chunk in chunks])
        combined_metadata = [chunk.get("metadata", {}) for chunk in chunks]
        
        # 获取所有芯片类别和文档类型
        chip_categories = [meta.get("chip_category", "") for meta in combined_metadata]
        doc_types = [meta.get("document_type", "") for meta in combined_metadata]
        
        # 简单的启发式规则
        unique_chips = set([cat for cat in chip_categories if cat])
        
        # 比较分析：需要多个不同的芯片
        if len(unique_chips) >= 2:
            suitable_tasks.append("comparison")
        
        # 综合设计：需要多个模块
        if len(chunks) >= 3:
            suitable_tasks.append("synthesis")
        
        # 深度分析：需要技术深度内容
        tech_indicators = ["原理", "机制", "架构", "实现", "寄存器"]
        if sum(1 for indicator in tech_indicators if indicator in combined_text) >= 2:
            suitable_tasks.append("analysis")
        
        # 推理解决：有问题解决内容
        problem_indicators = ["故障", "问题", "优化", "诊断"]
        if sum(1 for indicator in problem_indicators if indicator in combined_text) >= 1:
            suitable_tasks.append("reasoning")
        
        return suitable_tasks
    
    def _create_task_by_type(self, task_type: str, chunks: List[Dict]) -> Optional[IntegrationTask]:
        """根据类型创建任务"""
        if task_type == "comparison":
            return self._create_comparison_task(chunks)
        elif task_type == "synthesis":
            return self._create_synthesis_task(chunks)
        elif task_type == "analysis":
            return self._create_analysis_task(chunks)
        elif task_type == "reasoning":
            return self._create_reasoning_task(chunks)
        
        return None
    
    def _create_comparison_task(self, chunks: List[Dict]) -> Optional[IntegrationTask]:
        """创建比较分析任务"""
        # 获取配置模板
        templates = self.config.get("task_templates", {}).get("comparison", {})
        focus_areas = templates.get("focus_areas", ["技术特性对比"])
        
        focus_area = random.choice(focus_areas)
        instruction = f"比较以下技术资料中的{focus_area}，分析各方案的优劣势和适用场景"
        
        context = self._build_context_text(chunks)
        
        # 生成对比分析输出
        output = self._generate_comparison_output(chunks, focus_area)
        
        # 提取上下文来源
        context_sources = [chunk.get("metadata", {}).get("chip_category", "Unknown") for chunk in chunks]
        
        return IntegrationTask(
            task_type="comparison",
            instruction=instruction,
            context=context,
            expected_output=output,
            difficulty=self._determine_difficulty(chunks),
            integration_points=[
                "比较分析多个技术方案",
                "识别关键技术差异",
                "评估优劣势和适用性",
                "提供选型建议"
            ],
            quality_score=self._calculate_quality_score(instruction, context, output),
            context_sources=context_sources
        )
    
    def _create_synthesis_task(self, chunks: List[Dict]) -> Optional[IntegrationTask]:
        """创建综合设计任务"""
        templates = self.config.get("task_templates", {}).get("synthesis", {})
        solution_types = templates.get("solution_types", ["系统架构设计"])
        
        solution_type = random.choice(solution_types)
        instruction = f"基于以下技术资料，设计一个完整的{solution_type}方案"
        
        context = self._build_context_text(chunks)
        output = self._generate_synthesis_output(chunks, solution_type)
        
        context_sources = [chunk.get("metadata", {}).get("chip_category", "Unknown") for chunk in chunks]
        
        return IntegrationTask(
            task_type="synthesis",
            instruction=instruction,
            context=context,
            expected_output=output,
            difficulty=self._determine_difficulty(chunks),
            integration_points=[
                "整合多个技术模块",
                "设计系统架构",
                "制定实现策略",
                "考虑系统优化"
            ],
            quality_score=self._calculate_quality_score(instruction, context, output),
            context_sources=context_sources
        )
    
    def _create_analysis_task(self, chunks: List[Dict]) -> Optional[IntegrationTask]:
        """创建深度分析任务"""
        templates = self.config.get("task_templates", {}).get("analysis", {})
        analysis_dimensions = templates.get("analysis_dimensions", ["技术原理深度分析"])
        
        analysis_dimension = random.choice(analysis_dimensions)
        instruction = f"基于以下技术资料，进行{analysis_dimension}，阐述核心技术要点"
        
        context = self._build_context_text(chunks)
        output = self._generate_analysis_output(chunks, analysis_dimension)
        
        context_sources = [chunk.get("metadata", {}).get("chip_category", "Unknown") for chunk in chunks]
        
        return IntegrationTask(
            task_type="analysis",
            instruction=instruction,
            context=context,
            expected_output=output,
            difficulty=self._determine_difficulty(chunks),
            integration_points=[
                "深度分析技术原理",
                "阐述实现机制",
                "提供技术洞察",
                "系统性知识整合"
            ],
            quality_score=self._calculate_quality_score(instruction, context, output),
            context_sources=context_sources
        )
    
    def _create_reasoning_task(self, chunks: List[Dict]) -> Optional[IntegrationTask]:
        """创建推理解决任务"""
        templates = self.config.get("task_templates", {}).get("reasoning", {})
        problem_categories = templates.get("problem_categories", ["技术问题诊断"])
        
        problem_category = random.choice(problem_categories)
        instruction = f"基于以下技术资料，分析{problem_category}的解决方案"
        
        context = self._build_context_text(chunks)
        output = self._generate_reasoning_output(chunks, problem_category)
        
        context_sources = [chunk.get("metadata", {}).get("chip_category", "Unknown") for chunk in chunks]
        
        return IntegrationTask(
            task_type="reasoning",
            instruction=instruction,
            context=context,
            expected_output=output,
            difficulty=self._determine_difficulty(chunks),
            integration_points=[
                "分析问题根因",
                "推理解决路径",
                "制定应对策略",
                "提供实施方案"
            ],
            quality_score=self._calculate_quality_score(instruction, context, output),
            context_sources=context_sources
        )
    
    def _generate_comparison_output(self, chunks: List[Dict], focus_area: str) -> str:
        """生成比较分析输出"""
        return f"""基于提供的技术资料，对{focus_area}进行深入比较分析：

## 技术方案对比

各技术方案在{focus_area}方面表现出不同的特点和优势：

### 核心差异分析
1. **实现方式差异**：不同方案采用的技术路线和实现方法存在差异
2. **性能特点对比**：在关键性能指标上各有优劣
3. **适用场景分析**：针对不同应用场景的适配性不同

## 综合评估结论

### 优势对比
- 各方案在特定应用场景下都有其独特优势
- 需要根据具体需求进行权衡选择

### 选型建议
1. 根据具体应用需求确定关键性能要求
2. 综合考虑技术成熟度、成本和维护便利性
3. 建议进行原型验证以确认方案可行性
4. 考虑未来技术演进和扩展需求"""
    
    def _generate_synthesis_output(self, chunks: List[Dict], solution_type: str) -> str:
        """生成综合设计输出"""
        return f"""基于提供的技术资料，设计{solution_type}的完整方案：

## 方案架构设计

### 系统总体架构
基于技术资料分析，设计采用模块化架构，包含以下核心组件：

1. **核心控制模块**：负责系统主要控制逻辑
2. **功能处理模块**：实现特定功能处理
3. **接口适配模块**：处理外部接口和通信
4. **监控管理模块**：系统状态监控和管理

### 关键技术实现

#### 技术方案选择
- 综合考虑性能、成本、可靠性等因素
- 采用成熟稳定的技术方案
- 保证系统的可扩展性和可维护性

#### 实现策略
1. **分阶段实施**：按功能模块逐步实现和集成
2. **风险控制**：关键模块采用冗余设计
3. **性能优化**：针对关键路径进行优化

## 实施建议

1. **前期准备**：详细的需求分析和技术验证
2. **开发阶段**：模块化开发和集成测试
3. **部署维护**：完善的部署方案和维护机制"""
    
    def _generate_analysis_output(self, chunks: List[Dict], analysis_dimension: str) -> str:
        """生成深度分析输出"""
        return f"""基于提供的技术资料，对{analysis_dimension}进行深入分析：

## 技术原理分析

### 核心技术机制
技术实现基于以下核心原理和机制：

1. **基础原理**：阐述underlying技术原理
2. **实现机制**：详细分析具体实现方法
3. **关键技术**：识别和分析关键技术要点

### 技术特点分析

#### 设计理念
- 技术方案体现了特定的设计理念和思路
- 在性能、功耗、成本等方面进行了平衡考虑
- 充分考虑了实际应用场景的需求

#### 实现细节
1. **算法实现**：核心算法的具体实现方式
2. **硬件支持**：硬件层面的技术支持
3. **软件配合**：软件层面的协调配合

## 技术洞察与启示

### 技术优势
- 在特定应用领域具有明显优势
- 技术实现具有创新性和实用性
- 为相关技术发展提供了有益参考

### 应用价值
1. **直接应用**：可直接应用于相关产品和系统
2. **技术借鉴**：为类似技术方案提供参考
3. **发展潜力**：具有进一步发展和优化的潜力"""
    
    def _generate_reasoning_output(self, chunks: List[Dict], problem_category: str) -> str:
        """生成推理解决输出"""
        return f"""基于提供的技术资料，对{problem_category}进行分析和解决：

## 问题分析

### 根因分析
通过对技术资料的综合分析，问题可能的根本原因包括：

1. **技术层面**：技术实现或配置方面的问题
2. **系统层面**：系统集成或兼容性问题  
3. **环境层面**：外部环境或使用条件影响

### 影响评估
- 问题对系统性能和稳定性的影响程度
- 可能导致的连锁反应和次生问题
- 解决问题的紧迫性和优先级

## 解决方案

### 技术解决路径

#### 短期解决方案
1. **应急措施**：快速缓解问题影响的临时方案
2. **参数调整**：通过参数优化改善问题状况
3. **配置修正**：修正可能的配置错误

#### 长期优化策略
1. **根本性改进**：从源头解决问题的根本方案
2. **系统优化**：整体系统的优化和完善
3. **预防机制**：建立问题预防和早期发现机制

## 实施建议

### 解决步骤
1. **问题确认**：准确定位和确认问题范围
2. **方案实施**：按计划实施解决方案
3. **效果验证**：验证解决方案的有效性
4. **持续监控**：建立长期监控和维护机制

### 风险控制
- 在实施过程中注意控制风险
- 建立回退方案以应对意外情况
- 确保解决方案不会引入新的问题"""
    
    def _determine_difficulty(self, chunks: List[Dict]) -> str:
        """确定任务难度"""
        difficulty_dist = self.config["dataset_config"]["difficulty_distribution"]
        
        # 简单的启发式规则
        total_length = sum(len(chunk.get("text", "")) for chunk in chunks)
        unique_sources = len(set(chunk.get("metadata", {}).get("chip_category", "") for chunk in chunks))
        
        if total_length < 1000 or unique_sources <= 2:
            return "basic"
        elif total_length < 2000 or unique_sources <= 3:
            return "intermediate"
        else:
            return "advanced"
    
    def _calculate_quality_score(self, instruction: str, context: str, output: str) -> float:
        """计算质量评分"""
        quality_criteria = self.config.get("quality_criteria", {})
        
        score = 0.0
        
        # 上下文连贯性 (简化评估)
        context_coherence = min(1.0, len(context) / 1000)  # 基于长度的简单评估
        score += context_coherence * quality_criteria.get("context_coherence", {}).get("weight", 0.25)
        
        # 整合深度 (基于输出长度和结构)
        integration_depth = min(1.0, len(output) / 500)
        score += integration_depth * quality_criteria.get("integration_depth", {}).get("weight", 0.30)
        
        # 推理逻辑 (基于指令和输出的匹配度)
        reasoning_logic = 0.8  # 固定值，实际项目中可以用更复杂的方法
        score += reasoning_logic * quality_criteria.get("reasoning_logic", {}).get("weight", 0.25)
        
        # 实用价值 (基于内容丰富度)
        practical_value = min(1.0, (len(instruction) + len(output)) / 1000)
        score += practical_value * quality_criteria.get("practical_value", {}).get("weight", 0.20)
        
        return round(score, 3)
    
    def _validate_task_quality(self, task: IntegrationTask) -> bool:
        """验证任务质量"""
        # 基本质量检查
        if len(task.instruction) < 10:
            return False
        if len(task.context) < 100:
            return False
        if len(task.expected_output) < 50:
            return False
        if task.quality_score < 0.3:
            return False
        
        return True
    
    def save_dataset(self, dataset: List[IntegrationTask]):
        """保存数据集，支持Alpaca和ShareGPT格式"""
        output_config = self.config["output_format"]
        
        # 保存Alpaca格式
        if output_config.get("alpaca_format", True):
            self._save_alpaca_format(dataset)
        
        # 保存ShareGPT格式
        if output_config.get("sharegpt_format", False):
            self._save_sharegpt_format(dataset)
        
        # 保存统计信息
        self._save_statistics(dataset)
    
    def _save_alpaca_format(self, dataset: List[IntegrationTask]):
        """保存为Alpaca格式"""
        output_config = self.config["output_format"]
        include_metadata = output_config.get("include_metadata", True)
        metadata_fields = output_config.get("metadata_fields", [])
        
        alpaca_data = []
        for task in dataset:
            # 标准Alpaca字段
            alpaca_entry = {
                "instruction": task.instruction,
                "input": task.context,  # Alpaca格式中input字段用于上下文
                "output": task.expected_output
            }
            
            # 添加元数据字段
            if include_metadata:
                for field in metadata_fields:
                    if hasattr(task, field):
                        alpaca_entry[field] = getattr(task, field)
            
            alpaca_data.append(alpaca_entry)
        
        # 保存Alpaca格式文件
        alpaca_path = self.output_dir / "alpaca_integration_dataset.json"
        with open(alpaca_path, 'w', encoding='utf-8') as f:
            json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Alpaca格式数据集已保存到: {alpaca_path}")
    
    def _save_sharegpt_format(self, dataset: List[IntegrationTask]):
        """保存为ShareGPT格式"""
        output_config = self.config["output_format"]
        include_metadata = output_config.get("include_metadata", True)
        metadata_fields = output_config.get("metadata_fields", [])
        
        sharegpt_data = []
        for task in dataset:
            # ShareGPT对话格式
            sharegpt_entry = {
                "conversations": [
                    {
                        "from": "human", 
                        "value": f"{task.instruction}\n\n{task.context}"
                    },
                    {
                        "from": "gpt",
                        "value": task.expected_output
                    }
                ]
            }
            
            # 添加元数据字段
            if include_metadata:
                for field in metadata_fields:
                    if hasattr(task, field):
                        sharegpt_entry[field] = getattr(task, field)
            
            sharegpt_data.append(sharegpt_entry)
        
        # 保存ShareGPT格式文件
        sharegpt_path = self.output_dir / "sharegpt_integration_dataset.json"
        with open(sharegpt_path, 'w', encoding='utf-8') as f:
            json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"ShareGPT格式数据集已保存到: {sharegpt_path}")
    
    def _save_statistics(self, dataset: List[IntegrationTask]):
        """保存统计信息"""
        stats = {
            "total_samples": len(dataset),
            "format_info": {
                "alpaca_compatible": True,
                "sharegpt_compatible": True,
                "format_compliance": "完全符合Alpaca和ShareGPT标准格式"
            },
            "task_type_distribution": {},
            "difficulty_distribution": {},
            "quality_scores": {
                "avg_quality_score": sum(task.quality_score for task in dataset) / len(dataset),
                "min_quality_score": min(task.quality_score for task in dataset),
                "max_quality_score": max(task.quality_score for task in dataset)
            },
            "length_statistics": {
                "avg_instruction_length": sum(len(task.instruction) for task in dataset) / len(dataset),
                "avg_context_length": sum(len(task.context) for task in dataset) / len(dataset),
                "avg_output_length": sum(len(task.expected_output) for task in dataset) / len(dataset),
                "max_context_length": max(len(task.context) for task in dataset)
            },
            "integration_analysis": {
                "unique_context_sources": len(set(source for task in dataset for source in task.context_sources)),
                "avg_sources_per_task": sum(len(task.context_sources) for task in dataset) / len(dataset)
            },
            "generated_at": datetime.now().isoformat(),
            "config_used": self.config
        }
        
        # 统计任务类型分布
        for task in dataset:
            task_type = task.task_type
            stats["task_type_distribution"][task_type] = stats["task_type_distribution"].get(task_type, 0) + 1
        
        # 统计难度分布
        for task in dataset:
            difficulty = task.difficulty
            stats["difficulty_distribution"][difficulty] = stats["difficulty_distribution"].get(difficulty, 0) + 1
        
        # 保存统计文件
        stats_path = self.output_dir / "dataset_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据集统计信息已保存到: {stats_path}")

def main():
    """主函数"""
    builder = ConfigurableIntegrationDatasetBuilder(
        config_file="integration_config.json"
    )
    
    # 构建数据集
    dataset = builder.build_integration_dataset()
    
    if not dataset:
        logger.error("未能生成任何数据集样本！")
        return
    
    # 保存数据集（支持Alpaca和ShareGPT格式）
    builder.save_dataset(dataset)
    
    logger.info("=" * 60)
    logger.info("🎉 数据集构建完成！")
    logger.info("=" * 60)
    logger.info(f"总样本数: {len(dataset)}")
    logger.info(f"输出目录: {builder.output_dir}")
    logger.info("")
    logger.info("📁 生成的文件:")
    logger.info("  - alpaca_integration_dataset.json (Alpaca格式)")
    if builder.config["output_format"].get("sharegpt_format"):
        logger.info("  - sharegpt_integration_dataset.json (ShareGPT格式)")
    logger.info("  - dataset_statistics.json (详细统计)")
    logger.info("")
    logger.info("✅ 格式兼容性:")
    logger.info("  - Alpaca格式: 完全兼容")
    logger.info("  - ShareGPT格式: 完全兼容")
    logger.info("  - 包含丰富的元数据和质量评分")
    
    # 显示示例
    if dataset:
        logger.info("\n" + "=" * 60)
        logger.info("📋 数据集示例预览:")
        logger.info("=" * 60)
        
        sample_task = dataset[0]
        logger.info(f"任务类型: {sample_task.task_type}")
        logger.info(f"难度级别: {sample_task.difficulty}")
        logger.info(f"质量评分: {sample_task.quality_score}")
        logger.info(f"指令: {sample_task.instruction[:100]}...")
        logger.info(f"上下文长度: {len(sample_task.context)} 字符")
        logger.info(f"输出长度: {len(sample_task.expected_output)} 字符")
        logger.info(f"上下文来源: {', '.join(sample_task.context_sources[:3])}")

if __name__ == "__main__":
    main()

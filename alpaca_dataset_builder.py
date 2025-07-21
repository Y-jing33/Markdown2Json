#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpaca 格式数据集构建器
将知识库转换为适合大模型微调的 Alpaca 格式指令-响应对数据集
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AlpacaEntry:
    """Alpaca 格式数据条目"""
    instruction: str
    input: str = ""
    output: str = ""

class AlpacaDatasetBuilder:
    """Alpaca 格式数据集构建器"""
    
    def __init__(self, input_dir: str = "output", output_dir: str = "output/alpaca_dataset"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        self.documents_chunks = self._load_json("vectorization/documents_chunks.json")
        self.sections_chunks = self._load_json("vectorization/sections_chunks.json")
        self.analysis_result = self._load_json("analysis/analysis_result.json")
        
        # 指令模板
        self.instruction_templates = self._init_instruction_templates()
        
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
    
    def _init_instruction_templates(self) -> Dict[str, List[str]]:
        """初始化指令模板"""
        return {
            # 基础问答类
            "qa_basic": [
                "回答关于{chip_category}芯片的问题",
                "解释{chip_category}芯片的相关内容",
                "提供{chip_category}产品的技术信息",
                "根据技术文档回答问题",
            ],
            
            # 技术说明类
            "technical": [
                "说明{chip_category}芯片的技术特性",
                "解释{chip_category}系列的工作原理",
                "描述{chip_category}产品的功能特点",
                "介绍{chip_category}芯片的规格参数",
            ],
            
            # 应用指导类
            "application": [
                "提供{chip_category}芯片的应用指导",
                "说明{chip_category}产品的使用方法",
                "解释{chip_category}芯片的配置方法",
                "指导{chip_category}产品的开发应用",
            ],
            
            # 故障排除类
            "troubleshooting": [
                "解决{chip_category}芯片的常见问题",
                "诊断{chip_category}产品的故障现象",
                "提供{chip_category}芯片的故障排除方案",
                "分析{chip_category}产品的异常情况",
            ],
            
            # 产品选型类
            "selection": [
                "帮助选择合适的{chip_category}芯片型号",
                "比较{chip_category}系列产品的差异",
                "推荐适合的{chip_category}芯片方案",
                "分析{chip_category}产品的选型要点",
            ],
            
            # 数据手册类
            "datasheet": [
                "解读{chip_category}芯片的数据手册内容",
                "说明{chip_category}产品的电气特性",
                "解释{chip_category}芯片的管脚定义",
                "描述{chip_category}产品的封装信息",
            ],
            
            # 信息整合类（新增）
            "integration": [
                "综合分析{chip_category}芯片的多项技术特性",
                "整合{chip_category}产品的技术文档信息",
                "基于多个技术资料分析{chip_category}芯片的优势",
                "结合技术手册和应用指导，全面介绍{chip_category}产品",
            ]
        }
    
    def _get_instruction_type(self, metadata: Dict[str, Any]) -> str:
        """根据元数据确定指令类型"""
        doc_type = metadata.get("document_type", "").lower()
        
        if "troubleshooting" in doc_type:
            return "troubleshooting"
        elif "datasheet" in doc_type:
            return "datasheet"
        elif "selection" in doc_type:
            return "selection"
        elif "brochure" in doc_type:
            return "application"
        elif "manual" in doc_type:
            return "technical"
        else:
            return "qa_basic"
    
    def _generate_question_from_content(self, content: str, metadata: Dict[str, Any]) -> str:
        """根据内容生成与答案匹配的问题"""
        chip_category = metadata.get("chip_category", "芯片")
        doc_type = metadata.get("document_type", "")
        
        # 分析内容的主要特征，生成对应的问题
        questions = []
        
        # 基于实际内容特征生成问题
        if "PWM" in content and ("位编号" in content or "寄存器" in content):
            questions.extend([
                f"{chip_category}芯片的PWM寄存器有哪些位字段？",
                f"如何配置{chip_category}的PWM相关寄存器？",
                f"{chip_category}的PWM寄存器各位功能是什么？"
            ])
        elif "寄存器" in content and ("位" in content or "bit" in content.lower()):
            questions.extend([
                f"{chip_category}芯片寄存器的位定义是什么？",
                f"详细说明{chip_category}寄存器各位的功能",
                f"如何理解{chip_category}的寄存器配置？"
            ])
        elif "ADC" in content and ("采样" in content or "转换" in content):
            questions.extend([
                f"{chip_category}芯片的ADC采样时间是多少？",
                f"{chip_category}的ADC转换特性如何？",
                f"如何配置{chip_category}的ADC功能？"
            ])
        elif "复位" in content and ("电路" in content or "方式" in content):
            questions.extend([
                f"{chip_category}芯片有哪些复位方式？",
                f"{chip_category}的复位电路是如何工作的？",
                f"如何理解{chip_category}的复位机制？"
            ])
        elif "引脚" in content or "管脚" in content or "pin" in content.lower():
            questions.extend([
                f"{chip_category}芯片的引脚功能定义是什么？",
                f"如何理解{chip_category}的管脚配置？",
                f"{chip_category}芯片各引脚的作用是什么？"
            ])
        elif "工作模式" in content or "mode" in content.lower():
            questions.extend([
                f"{chip_category}芯片支持哪些工作模式？",
                f"如何切换{chip_category}的工作模式？",
                f"{chip_category}芯片各工作模式的特点是什么？"
            ])
        elif "中断" in content or "interrupt" in content.lower():
            questions.extend([
                f"{chip_category}芯片的中断机制是怎样的？",
                f"如何配置{chip_category}的中断功能？",
                f"{chip_category}芯片支持哪些中断类型？"
            ])
        elif "定时器" in content or "timer" in content.lower():
            questions.extend([
                f"{chip_category}芯片的定时器功能如何使用？",
                f"如何配置{chip_category}的定时器？",
                f"{chip_category}芯片定时器的特性是什么？"
            ])
        elif "电气特性" in content or "参数" in content:
            questions.extend([
                f"{chip_category}芯片的电气参数是什么？",
                f"{chip_category}芯片有哪些技术规格？",
                f"如何理解{chip_category}的电气特性？"
            ])
        elif "封装" in content or "package" in content.lower():
            questions.extend([
                f"{chip_category}芯片有哪些封装形式？",
                f"{chip_category}芯片的封装规格是什么？",
                f"如何选择{chip_category}芯片的封装类型？"
            ])
        
        # 基于文档类型生成问题
        if "troubleshooting" in doc_type.lower():
            questions.extend([
                f"{chip_category}芯片出现问题时如何排查？",
                f"{chip_category}芯片常见故障的解决方法？",
                f"如何诊断{chip_category}芯片的问题？"
            ])
        elif "selection" in doc_type.lower():
            questions.extend([
                f"如何选择合适的{chip_category}芯片型号？",
                f"{chip_category}系列芯片的选型要点是什么？",
                f"{chip_category}芯片选型时需要考虑哪些因素？"
            ])
        elif "datasheet" in doc_type.lower():
            questions.extend([
                f"{chip_category}芯片的技术规格说明？",
                f"如何理解{chip_category}芯片的数据手册？",
                f"{chip_category}芯片的详细技术参数？"
            ])
        
        # 如果没有找到特定问题，生成通用问题
        if not questions:
            questions.extend([
                f"请介绍{chip_category}芯片的相关信息",
                f"{chip_category}芯片的技术特点是什么？",
                f"关于{chip_category}芯片的技术说明",
                f"如何理解{chip_category}芯片的功能？"
            ])
        
        return random.choice(questions)
    
    def _clean_content(self, content: str) -> str:
        """清理文本内容"""
        if not content:
            return None
            
        # 移除HTML表格属性
        content = re.sub(r'colspan="?\d+"?', '', content)
        content = re.sub(r'rowspan="?\d+"?', '', content)
        
        # 移除HTML标签
        content = re.sub(r'<[^>]*>', '', content)
        
        # 移除表格分隔符和特殊字符
        content = re.sub(r'</?(td|tr|table|th|thead|tbody)[^>]*>', '', content)
        
        # 移除LaTeX数学符号和格式化错误
        content = re.sub(r'\$\\mathbb\{[^}]*\}\$', '', content)
        content = re.sub(r'\$[^$]*\$', '', content)  # 移除其他LaTeX公式
        content = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', content)  # 移除LaTeX命令
        
        # 修复常见的OCR错误
        content = content.replace('$\\mathbb{Q}$', 'M')  # Cortex-M 内核
        content = content.replace('Cortex $\\mathbb{Q}$ - MO+', 'Cortex-M0+')
        content = content.replace('$\\mathbb{Q}$ - MO+', 'M0+')
        content = content.replace('- MO+', '-M0+')
        
        # 清理表格残留标记
        content = re.sub(r'olspan="?\d+"?>', '', content)
        content = re.sub(r'>\s*PWMPD', 'PWMPD', content)
        
        # 移除多余的空白字符
        content = re.sub(r'\s+', ' ', content.strip())
        
        # 移除图片链接
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
        
        # 移除多余的标点符号
        content = re.sub(r'[·•]{2,}', '·', content)
        
        # 移除无意义的HTML残留
        content = re.sub(r'</?[a-zA-Z][^>]*>', '', content)
        content = re.sub(r'&[a-zA-Z]+;', '', content)  # 移除HTML实体
        
        # 处理表格数据格式
        content = self._format_table_content(content)
        
        # 英文术语中文化处理
        content = self._localize_technical_terms(content)
        
        # 移除过长的无意义重复字符
        content = re.sub(r'(-{3,}|={3,}|_{3,})', '', content)
        
        # 确保内容不为空且有意义
        if not content or len(content.strip()) < 15:
            return None
            
        # 移除开头和结尾的多余符号
        content = content.strip('- \t\n·•')
        
        return content.strip()
    
    def _format_table_content(self, content: str) -> str:
        """格式化表格内容"""
        # 处理PWM寄存器表格
        if "PWM" in content and re.search(r'PWMPD|位编号|位符号', content):
            # 提取PWM寄存器信息
            pwm_info = []
            
            # 匹配位范围和说明
            bit_matches = re.findall(r'(\d+)~(\d+)\s*([A-Z][A-Z0-9_\[\]:]*)\s*([^0-9]+?)(?=\d+~|\d+\s*-|$)', content)
            for match in bit_matches:
                start_bit, end_bit, name, description = match
                description = re.sub(r'\s+', ' ', description.strip())
                if description and len(description) > 5:
                    if start_bit == end_bit:
                        pwm_info.append(f"位{start_bit} - {name}：{description}")
                    else:
                        pwm_info.append(f"位{start_bit}~{end_bit} - {name}：{description}")
            
            # 匹配单个位定义
            single_bit_matches = re.findall(r'(\d+)\s+([A-Z][A-Z0-9_]*)\s+([^0-9]+?)(?=\d+\s+[A-Z]|$)', content)
            for match in single_bit_matches:
                bit_num, name, description = match
                description = re.sub(r'\s+', ' ', description.strip())
                if description and len(description) > 5:
                    pwm_info.append(f"位{bit_num} - {name}：{description}")
            
            if pwm_info:
                return "PWM寄存器位定义：\n" + "\n".join(pwm_info)
        
        # 检测是否包含寄存器或引脚表格数据
        if re.search(r'\d+\s+[A-Z_][A-Z0-9_]*\s+[^<]+', content):
            # 这是寄存器/引脚定义表格
            lines = []
            # 匹配表格行模式：序号 + 名称 + 描述
            table_matches = re.findall(r'(\d+)\s+([A-Z_][A-Z0-9_]*)\s+([^0-9]+?)(?=\d+\s+[A-Z_]|$)', content)
            
            for match in table_matches:
                number, name, description = match
                # 清理描述内容
                description = re.sub(r'\s+', ' ', description.strip())
                if description and len(description) > 5:
                    lines.append(f"位{number} - {name}：{description}")
            
            if lines:
                return "寄存器位定义如下：\n" + "\n".join(lines)
        
        # 检测ADC相关的数值表格
        if "ADC" in content and re.search(r'\d+\s*:\s*[^:]+', content):
            # 处理ADC配置表格
            lines = []
            config_matches = re.findall(r'(\d+)\s*:\s*([^0-9]+?)(?=\d+\s*:|其它|说明|$)', content)
            
            for match in config_matches:
                code, description = match
                description = re.sub(r'\s+', ' ', description.strip())
                if description and len(description) > 5:
                    lines.append(f"配置{code}：{description}")
            
            if lines:
                # 提取说明部分
                explanation_match = re.search(r'说明\s*:\s*([^:]+?)(?:其中|$)', content)
                result = "ADC配置选项：\n" + "\n".join(lines)
                if explanation_match:
                    explanation = explanation_match.group(1).strip()
                    result += f"\n\n说明：{explanation}"
                return result
        
        # 检测时间相关的配置表格
        if re.search(r'\d+\s*:\s*采样时间|时钟', content):
            lines = []
            time_matches = re.findall(r'(\d+)\s*:\s*([^0-9]+?)(?=\d+\s*:|其它|说明|$)', content)
            
            for match in time_matches:
                code, description = match
                description = re.sub(r'\s+', ' ', description.strip())
                if description and len(description) > 5:
                    lines.append(f"选项{code}：{description}")
            
            if lines:
                return "时序配置选项：\n" + "\n".join(lines)
        
        # 通用表格清理
        if re.search(r'colspan|rowspan|位编号|位符号', content):
            # 移除表格格式化标记
            content = re.sub(r'[a-z]+span="?\d+"?>', '', content)
            content = re.sub(r'位编号位符号说明', '寄存器位定义：', content)
            content = re.sub(r'\s*-\s*保留.*?(?=\d|$)', '', content)  # 移除保留位说明
        
        return content
    
    def _localize_technical_terms(self, content: str) -> str:
        """技术术语中文化处理"""
        # 常见英文技术术语映射
        term_mapping = {
            "Feature": "特性",
            "Built-in": "内置",
            "Hardware": "硬件", 
            "Module": "模块",
            "Configurable": "可配置",
            "Default": "默认",
            "Initial value": "初始值",
            "Supports": "支持",
            "Data units": "数据单位",
            "Programmable": "可编程",
            "Polynomial": "多项式",
            "CRC": "循环冗余校验",
            "Register": "寄存器",
            "Function": "功能",
            "Pin definition": "管脚定义",
            "Operating mode": "工作模式",
            "Configuration": "配置",
            "Interrupt": "中断",
            "Timer": "定时器",
            "ADC": "模数转换器",
            "PWM": "脉宽调制",
            "GPIO": "通用输入输出",
            "UART": "通用异步收发器",
            "SPI": "串行外设接口",
            "I2C": "集成电路总线",
            "Flash": "闪存",
            "RISC": "精简指令集",
            "CMSIS": "Cortex微控制器软件接口标准",
            "DMA": "直接存储器访问",
            "Cortex-M": "Cortex-M",
            "microcontroller": "微控制器",
            "core": "内核"
        }
        
        # 进行术语替换
        for english_term, chinese_term in term_mapping.items():
            content = re.sub(r'\b' + re.escape(english_term) + r'\b', chinese_term, content, flags=re.IGNORECASE)
        
        # 处理特殊的芯片型号和术语
        content = re.sub(r'Arm\s+Cortex', 'ARM Cortex', content, flags=re.IGNORECASE)
        content = re.sub(r'Cortex[- ]?([MmRrAa])\d*\+?', r'Cortex-\1', content)
        
        # 处理常见的英文模式
        content = re.sub(r'\b(\d+)-bit\b', r'\1位', content, flags=re.IGNORECASE)
        content = re.sub(r'\bbit\b', '位', content, flags=re.IGNORECASE)
        content = re.sub(r'\bbyte\b', '字节', content, flags=re.IGNORECASE)
        content = re.sub(r'\bMHz\b', 'MHz', content, flags=re.IGNORECASE)
        content = re.sub(r'\bkHz\b', 'kHz', content, flags=re.IGNORECASE)
        content = re.sub(r'\bmA\b', 'mA', content, flags=re.IGNORECASE)
        content = re.sub(r'\bV\b(?=\s|$)', 'V', content)
        
        # 修复特定的术语组合
        content = re.sub(r'精简指令集\s*\(\s*RISC\s*\)', '精简指令集(RISC)', content)
        content = re.sub(r'直接存储器访问控制器\s*\(\s*DMA\s*\)', '直接存储器访问控制器(DMA)', content)
        
        return content
    
    def _generate_chinese_response(self, content: str, metadata: Dict[str, Any]) -> str:
        """基于内容生成中文回答"""
        chip_category = metadata.get("chip_category", "芯片")
        doc_type = metadata.get("document_type", "")
        
        # 先进行基础的中文化处理
        chinese_content = self._localize_technical_terms(content)
        
        # 如果内容主要是英文，生成中文描述
        if self._is_mainly_english(chinese_content):
            return self._generate_chinese_description(chinese_content, chip_category, doc_type)
        
        # 直接返回清理后的内容，不添加模板格式
        return chinese_content
    
    def _is_mainly_english(self, content: str) -> bool:
        """判断内容是否主要是英文"""
        # 统计中英文字符比例
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
        english_chars = len(re.findall(r'[a-zA-Z]', content))
        total_chars = chinese_chars + english_chars
        
        if total_chars == 0:
            return False
        
        # 如果英文字符超过70%，认为是英文内容
        return english_chars / total_chars > 0.7
    
    def _generate_chinese_description(self, content: str, chip_category: str, doc_type: str) -> str:
        """为英文内容生成中文描述"""
        # 提取关键技术特征
        features = []
        
        # 检测技术特性
        if re.search(r'built.?in.*CRC', content, re.IGNORECASE):
            features.append(f"内置硬件CRC校验模块")
        
        if re.search(r'configurable.*initial.*value', content, re.IGNORECASE):
            features.append("支持可配置的初始值设置")
        
        if re.search(r'(\d+).?bit.*data.*unit', content, re.IGNORECASE):
            match = re.search(r'(\d+).?bit.*data.*unit', content, re.IGNORECASE)
            if match:
                bits = match.group(1)
                features.append(f"支持{bits}位数据单元处理")
        
        if re.search(r'programmable.*polynomial', content, re.IGNORECASE):
            features.append("具有可编程多项式功能")
        
        if re.search(r'register.*configuration', content, re.IGNORECASE):
            features.append("通过寄存器进行配置管理")
        
        if re.search(r'pin.*definition|pin.*function', content, re.IGNORECASE):
            features.append("具有明确的管脚定义和功能分配")
        
        if re.search(r'operating.*mode', content, re.IGNORECASE):
            features.append("支持多种工作模式")
        
        if re.search(r'interrupt.*handling', content, re.IGNORECASE):
            features.append("具备完善的中断处理机制")
        
        # ADC相关特性检测
        if re.search(r'ADC.*采样时间|sampling.*time', content, re.IGNORECASE):
            features.append("ADC模块支持可配置的采样时间")
        
        if re.search(r'ADC.*转换时间|conversion.*time', content, re.IGNORECASE):
            features.append("ADC具有固定的转换时间")
        
        if re.search(r'ADCEN|ADC.*启动|ADC.*enable', content, re.IGNORECASE):
            features.append("ADC模块具有电源控制功能")
        
        # 组合特性描述
        if features:
            description = "主要特性包括：" + "、".join(features) + "。"
        else:
            # 通用技术描述
            description = f"{chip_category}芯片具有先进的技术特性和可靠的性能表现，适用于各种工业和消费电子应用场景。"
        
        # 添加具体的技术参数（如果能提取到）
        technical_params = self._extract_technical_parameters(content)
        if technical_params:
            description += f" 主要技术参数：{technical_params}。"
        
        return description
    
    def _extract_technical_parameters(self, content: str) -> str:
        """提取技术参数"""
        params = []
        
        # 提取频率参数
        freq_match = re.search(r'(\d+(?:\.\d+)?)\s*(MHz|KHz|kHz)', content, re.IGNORECASE)
        if freq_match:
            freq, unit = freq_match.groups()
            params.append(f"工作频率{freq}{unit}")
        
        # 提取电压参数
        volt_match = re.search(r'(\d+(?:\.\d+)?)\s*V(?:\s|$)', content)
        if volt_match:
            voltage = volt_match.group(1)
            params.append(f"工作电压{voltage}V")
        
        # 提取电流参数
        current_match = re.search(r'(\d+(?:\.\d+)?)\s*(mA|µA)', content, re.IGNORECASE)
        if current_match:
            current, unit = current_match.groups()
            params.append(f"工作电流{current}{unit}")
        
        # 提取温度范围
        temp_match = re.search(r'(-?\d+)°C.*?(\d+)°C', content)
        if temp_match:
            min_temp, max_temp = temp_match.groups()
            params.append(f"工作温度范围{min_temp}°C至{max_temp}°C")
        
        # 提取ADC相关参数
        # 采样时间
        sample_time_match = re.search(r'采样时间[约]*(\d+)[个]*[系统]*时钟|(\d+)ns.*采样', content)
        if sample_time_match:
            time_val = sample_time_match.group(1) or sample_time_match.group(2)
            if sample_time_match.group(1):
                params.append(f"ADC采样时间约{time_val}个系统时钟")
            else:
                params.append(f"ADC采样时间约{time_val}ns")
        
        # 转换时间
        conv_time_match = re.search(r'转换时间[固定为]*(\d+)ns', content)
        if conv_time_match:
            conv_time = conv_time_match.group(1)
            params.append(f"ADC转换时间固定为{conv_time}ns")
        
        # 总转换时间公式
        if "TADC" in content and "采样时间" in content:
            params.append("ADC总转换时间 = 采样时间 + 转换时间")
        
        return "；".join(params) if params else ""
    
    def _should_include_chunk(self, chunk: Dict[str, Any]) -> bool:
        """判断是否应该包含该数据块"""
        content = chunk.get("text", "")
        
        # 过滤过短的内容
        if len(content.strip()) < 30:
            return False
        
        # 过滤纯符号或无意义内容
        if re.match(r'^[·•\-\s]*$', content.strip()):
            return False
        
        # 过滤纯数字或代码
        if re.match(r'^[\d\s\.\-\+]*$', content.strip()):
            return False
            
        # 过滤主要是HTML标签的内容
        if content.count('<') > len(content) / 10:
            return False
            
        # 过滤表格标题行或空行
        if re.match(r'^(序号|编号|说明|功能|参数|特性|型号)\s*$', content.strip()):
            return False
        
        # 确保内容有实际的技术信息
        technical_keywords = ['芯片', '寄存器', '引脚', '管脚', 'PWM', 'ADC', '中断', '定时器', 
                            '工作模式', '电路', '功能', '配置', '特性', '参数']
        
        if not any(keyword in content for keyword in technical_keywords):
            return False
        
        return True
    
    def build_qa_dataset(self, max_samples: int = None) -> List[AlpacaEntry]:
        """构建问答数据集"""
        dataset = []
        
        logger.info("开始构建文档级别的问答对...")
        
        # 处理文档块
        for chunk in self.documents_chunks:
            if not self._should_include_chunk(chunk):
                continue
            
            metadata = chunk.get("metadata", {})
            content = self._clean_content(chunk.get("text", ""))
            
            if not content:
                continue
            
            # 生成中文化的输出内容
            chinese_output = self._generate_chinese_response(content, metadata)
            
            # 生成指令
            instruction_type = self._get_instruction_type(metadata)
            instruction_templates = self.instruction_templates.get(instruction_type, self.instruction_templates["qa_basic"])
            
            chip_category = metadata.get("chip_category", "芯片")
            instruction = random.choice(instruction_templates).format(chip_category=chip_category)
            
            # 生成问题
            question = self._generate_question_from_content(content, metadata)
            
            # 创建Alpaca条目
            entry = AlpacaEntry(
                instruction=instruction,
                input=question,
                output=chinese_output
            )
            
            dataset.append(entry)
            
            if max_samples and len(dataset) >= max_samples:
                break
        
        logger.info(f"文档级别生成了 {len(dataset)} 个问答对")
        
        # 处理章节块
        section_count = 0
        logger.info("开始构建章节级别的问答对...")
        
        for chunk in self.sections_chunks:
            if not self._should_include_chunk(chunk):
                continue
            
            if max_samples and len(dataset) >= max_samples:
                break
            
            metadata = chunk.get("metadata", {})
            content = self._clean_content(chunk.get("text", ""))
            
            if not content:
                continue
            
            # 生成中文化的输出内容
            chinese_output = self._generate_chinese_response(content, metadata)
            
            # 生成指令
            instruction_type = self._get_instruction_type(metadata)
            instruction_templates = self.instruction_templates.get(instruction_type, self.instruction_templates["qa_basic"])
            
            chip_category = metadata.get("chip_category", "芯片")
            instruction = random.choice(instruction_templates).format(chip_category=chip_category)
            
            # 生成问题
            question = self._generate_question_from_content(content, metadata)
            
            # 创建Alpaca条目
            entry = AlpacaEntry(
                instruction=instruction,
                input=question,
                output=chinese_output
            )
            
            dataset.append(entry)
            section_count += 1
        
        logger.info(f"章节级别生成了 {section_count} 个问答对")
        logger.info(f"总共生成了 {len(dataset)} 个Alpaca格式的训练样本")
        
        return dataset
    
    def build_instruction_dataset(self, max_samples: int = None) -> List[AlpacaEntry]:
        """构建纯指令数据集（无input字段）"""
        dataset = []
        
        logger.info("开始构建纯指令数据集...")
        
        for chunk in self.documents_chunks + self.sections_chunks:
            if not self._should_include_chunk(chunk):
                continue
            
            if max_samples and len(dataset) >= max_samples:
                break
            
            metadata = chunk.get("metadata", {})
            content = self._clean_content(chunk.get("text", ""))
            
            if not content:
                continue
            
            # 生成中文化的输出内容
            chinese_output = self._generate_chinese_response(content, metadata)
            
            # 根据内容生成具体的、有针对性的指令
            chip_category = metadata.get("chip_category", "芯片")
            doc_type = metadata.get("document_type", "")
            
            # 根据内容生成具体指令
            if "PWM" in content and "寄存器" in content:
                instruction = f"详细说明{chip_category}芯片PWM相关寄存器的位定义和功能"
            elif "寄存器" in content and ("位" in content or "配置" in content):
                instruction = f"解释{chip_category}芯片寄存器的配置方法和各位功能"
            elif "引脚" in content or "管脚" in content:
                instruction = f"介绍{chip_category}芯片的引脚定义和功能分配"
            elif "ADC" in content and ("采样" in content or "转换" in content):
                instruction = f"说明{chip_category}芯片ADC模块的工作特性和配置方法"
            elif "复位" in content and "电路" in content:
                instruction = f"解释{chip_category}芯片的复位电路和复位方式"
            elif "中断" in content:
                instruction = f"说明{chip_category}芯片的中断处理机制和配置方法"
            elif "定时器" in content or "timer" in content.lower():
                instruction = f"介绍{chip_category}芯片定时器功能的使用方法"
            elif "工作模式" in content:
                instruction = f"说明{chip_category}芯片支持的各种工作模式"
            elif "电气特性" in content or "参数" in content:
                instruction = f"提供{chip_category}芯片的电气特性和技术参数"
            elif "封装" in content:
                instruction = f"介绍{chip_category}芯片的封装形式和规格"
            elif "troubleshooting" in doc_type.lower():
                instruction = f"提供{chip_category}芯片的故障排除指导和解决方案"
            elif "selection" in doc_type.lower():
                instruction = f"指导{chip_category}芯片的选型要点和选择方法"
            elif "datasheet" in doc_type.lower():
                instruction = f"解读{chip_category}芯片数据手册中的技术规格"
            else:
                instruction = f"介绍{chip_category}芯片的技术特性和功能"
            
            # 创建Alpaca条目（无input）
            entry = AlpacaEntry(
                instruction=instruction,
                input="",
                output=chinese_output
            )
            
            dataset.append(entry)
        
        logger.info(f"生成了 {len(dataset)} 个纯指令训练样本")
        return dataset
    
    def save_dataset(self, dataset: List[AlpacaEntry], filename: str = "alpaca_dataset.json"):
        """保存数据集"""
        output_path = self.output_dir / filename
        
        # 转换为标准Alpaca格式
        alpaca_data = []
        for entry in dataset:
            alpaca_entry = {
                "instruction": entry.instruction,
                "input": entry.input,
                "output": entry.output
            }
            alpaca_data.append(alpaca_entry)
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据集已保存到: {output_path}")
        
        # 生成统计报告
        self._generate_dataset_stats(alpaca_data, output_path.parent / f"{output_path.stem}_stats.json")
    
    def _generate_dataset_stats(self, dataset: List[Dict], stats_path: Path):
        """生成数据集统计信息"""
        stats = {
            "total_samples": len(dataset),
            "samples_with_input": sum(1 for item in dataset if item["input"].strip()),
            "samples_without_input": sum(1 for item in dataset if not item["input"].strip()),
            "avg_instruction_length": sum(len(item["instruction"]) for item in dataset) / len(dataset),
            "avg_input_length": sum(len(item["input"]) for item in dataset) / len(dataset),
            "avg_output_length": sum(len(item["output"]) for item in dataset) / len(dataset),
            "max_output_length": max(len(item["output"]) for item in dataset),
            "min_output_length": min(len(item["output"]) for item in dataset),
            "generated_at": datetime.now().isoformat()
        }
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据集统计信息已保存到: {stats_path}")
    
    def build_complete_dataset(self, max_qa_samples: int = 5000, max_instruction_samples: int = 3000):
        """构建完整的训练数据集"""
        logger.info("开始构建完整的Alpaca格式数据集...")
        
        # 构建问答数据集
        qa_dataset = self.build_qa_dataset(max_qa_samples)
        
        # 构建纯指令数据集
        instruction_dataset = self.build_instruction_dataset(max_instruction_samples)
        
        # 合并数据集
        complete_dataset = qa_dataset + instruction_dataset
        
        # 打乱数据
        random.shuffle(complete_dataset)
        
        # 保存完整数据集
        self.save_dataset(complete_dataset, "alpaca_complete_dataset.json")
        
        # 分别保存不同类型的数据集
        self.save_dataset(qa_dataset, "alpaca_qa_dataset.json")
        self.save_dataset(instruction_dataset, "alpaca_instruction_dataset.json")
        
        logger.info(f"完整数据集构建完成，总样本数: {len(complete_dataset)}")
        return complete_dataset

def main():
    """主函数"""
    builder = AlpacaDatasetBuilder()
    
    # 构建完整数据集
    dataset = builder.build_complete_dataset(
        max_qa_samples=5000,
        max_instruction_samples=3000
    )
    
    logger.info("Alpaca数据集构建完成！")
    logger.info(f"输出目录: {builder.output_dir}")
    
    # 显示示例
    if dataset:
        logger.info("\n=== 数据集示例 ===")
        for i, entry in enumerate(dataset[:3]):
            logger.info(f"\n示例 {i+1}:")
            logger.info(f"指令: {entry.instruction}")
            if entry.input:
                logger.info(f"输入: {entry.input}")
            logger.info(f"输出: {entry.output[:200]}...")

if __name__ == "__main__":
    main()
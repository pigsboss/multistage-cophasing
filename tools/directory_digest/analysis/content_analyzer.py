# -*- coding: utf-8 -*-
"""
content_analyzer.py - 基于内容特征的动态分类器
"""

import re
from typing import Dict, Any, List, Tuple
from pathlib import Path

from ..utils.complexity_analyzer import ComplexityAnalyzer


class ContentAnalyzer:
    """基于内容特征的动态分类器"""
    
    @staticmethod
    def detect_structure(content: str) -> Dict[str, Any]:
        """
        检测内容结构特征
        返回: {
            'is_tabular': bool,
            'is_natural_language': bool,
            'is_code': bool,
            'tabular_ratio': float,
            'natural_language_ratio': float,
            'structure_type': str,  # 'natural_language', 'tabular_data', 'mixed', 'code', 'config'
            'entropy': float
        }
        """
        lines = content.split('\n')
        total_lines = len(lines)
        if total_lines == 0:
            return {'is_tabular': False, 'is_natural_language': False, 'is_code': False,
                   'structure_type': 'empty', 'entropy': 0}
        
        sample = content[:10000]  # 限制分析范围提高性能
        sample_lines = lines[:min(100, total_lines)]
        
        # 1. Tabular Data 检测
        delimiter_pattern = re.compile(r'[,;\t|]{2,}')
        numeric_pattern = re.compile(r'^[\s\d\.\-\+eE,;\t|]+$')
        delimiter_lines = 0
        numeric_lines = 0
        
        for line in sample_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if delimiter_pattern.search(stripped) or stripped.count(',') > 3:
                delimiter_lines += 1
            if numeric_pattern.match(stripped) and len(stripped) > 5:
                numeric_lines += 1
        
        tabular_ratio = (delimiter_lines + numeric_lines) / max(len(sample_lines), 1)
        is_tabular = tabular_ratio > 0.3
        
        # 2. Natural Language 检测
        words = re.findall(r'\b[a-zA-Z]{2,}\b', sample.lower())
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', sample))
        
        word_diversity = 0
        stop_words_found = 0
        if words:
            unique = len(set(words))
            word_diversity = unique / len(words)
            stop_words = {'the', 'and', 'is', 'of', 'to', 'in', 'that', 'have', 'it', 
                         '的', '了', '是', '在', '有', '和', '就', '不', '人'}
            stop_words_found = sum(1 for w in words if w in stop_words)
        
        sentence_endings = sample.count('.') + sample.count('?') + sample.count('!') + \
                          sample.count('。') + sample.count('？') + sample.count('！')
        
        is_natural_language = (
            (sentence_endings > 3 and word_diversity > 0.1 and stop_words_found > 5) or
            (chinese_chars > 50 and sentence_endings > 0)
        )
        nl_ratio = min(1.0, (sentence_endings / max(len(sample_lines), 1)) * 2)
        
        # 3. Code 检测
        code_patterns = [
            r'\b(def|class|function|if|for|while|return|import|from|#include|package|public|private)\b',
            r'[{}\[\]()]+',
            r'^(\s{4}|\t)',  # 缩进
        ]
        code_matches = sum(1 for p in code_patterns if re.search(p, sample[:2000], re.M))
        is_code = code_matches > 3
        
        # 4. Config 检测（键值对结构）
        config_patterns = [
            r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=:]\s*.+$',  # key = value
            r'^[^:]+:\s+.+$',  # YAML风格
        ]
        config_lines = sum(1 for line in sample_lines if any(re.match(p, line.strip()) for p in config_patterns))
        is_config = (config_lines / max(len(sample_lines), 1)) > 0.3 and not is_code
        
        # 5. 确定结构类型
        if is_tabular and is_natural_language:
            structure_type = 'mixed'
        elif is_tabular:
            structure_type = 'tabular_data'
        elif is_code:
            structure_type = 'code'
        elif is_config:
            structure_type = 'config'
        elif is_natural_language:
            structure_type = 'natural_language'
        else:
            structure_type = 'unknown'
        
        return {
            'is_tabular': is_tabular,
            'is_natural_language': is_natural_language,
            'is_code': is_code,
            'tabular_ratio': tabular_ratio,
            'natural_language_ratio': nl_ratio,
            'structure_type': structure_type,
            'entropy': ComplexityAnalyzer.calculate_entropy(content)
        }
    
    @staticmethod
    def classify_file(content: str, filepath: Path) -> Dict[str, Any]:
        """完整文件分类"""
        # 首先尝试解码
        try:
            if isinstance(content, bytes):
                content = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                content = content.decode('latin-1', errors='ignore')
            except:
                return {'type': 'binary', 'strategy': 'metadata_only', 'reason': 'undecodable'}
        
        analysis = ContentAnalyzer.detect_structure(content)
        
        # 决策映射
        type_mapping = {
            'natural_language': ('human_readable', 'embed_with_limit'),
            'code': ('source_code', 'skeleton_or_full'),
            'config': ('config_script', 'structure_summary'),
            'tabular_data': ('tabular_data', 'header_only'),
            'mixed': ('mixed_document', 'extract_header'),
            'unknown': ('unknown_text', 'first_lines')
        }
        
        file_type, strategy = type_mapping.get(analysis['structure_type'], ('unknown', 'first_lines'))
        
        return {
            'type': file_type,
            'strategy': strategy,
            'structure_type': analysis['structure_type'],
            'entropy': analysis['entropy'],
            'metrics': analysis
        }

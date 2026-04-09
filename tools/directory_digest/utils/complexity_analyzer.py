# -*- coding: utf-8 -*-
"""
complexity_analyzer.py - 代码复杂度分析器
"""

import re
import ast
import math
from typing import Dict, Any, Optional


class ComplexityAnalyzer:
    """代码复杂度分析器"""
    
    @staticmethod
    def analyze_python(content: str) -> Dict[str, Any]:
        """分析Python代码复杂度"""
        try:
            tree = ast.parse(content)
            
            # 统计各种复杂度指标
            metrics = {
                "cyclomatic_complexity": 0,
                "function_count": 0,
                "class_count": 0,
                "average_function_length": 0,
                "max_nesting_depth": 0,
                "import_count": 0
            }
            
            # 递归分析AST
            def analyze_node(node, depth: int = 0):
                nonlocal metrics
                
                # 更新最大嵌套深度
                metrics["max_nesting_depth"] = max(metrics["max_nesting_depth"], depth)
                
                # 分析节点类型
                if isinstance(node, ast.If):
                    metrics["cyclomatic_complexity"] += 1
                    for child in ast.iter_child_nodes(node):
                        analyze_node(child, depth + 1)
                elif isinstance(node, ast.While):
                    metrics["cyclomatic_complexity"] += 1
                    for child in ast.iter_child_nodes(node):
                        analyze_node(child, depth + 1)
                elif isinstance(node, ast.For):
                    metrics["cyclomatic_complexity"] += 1
                    for child in ast.iter_child_nodes(node):
                        analyze_node(child, depth + 1)
                elif isinstance(node, ast.Try):
                    metrics["cyclomatic_complexity"] += 1
                    for child in ast.iter_child_nodes(node):
                        analyze_node(child, depth + 1)
                elif isinstance(node, ast.FunctionDef):
                    metrics["function_count"] += 1
                    # 分析函数内的语句数量
                    stmt_count = len([n for n in ast.walk(node) if isinstance(n, ast.stmt)])
                    metrics["average_function_length"] += stmt_count
                elif isinstance(node, ast.ClassDef):
                    metrics["class_count"] += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    metrics["import_count"] += 1
                else:
                    for child in ast.iter_child_nodes(node):
                        analyze_node(child, depth)
            
            # 开始分析
            analyze_node(tree)
            
            # 计算平均函数长度
            if metrics["function_count"] > 0:
                metrics["average_function_length"] = metrics["average_function_length"] / metrics["function_count"]
            
            # 评估复杂度等级
            complexity_score = metrics["cyclomatic_complexity"]
            if complexity_score < 10:
                metrics["complexity_level"] = "简单"
            elif complexity_score < 20:
                metrics["complexity_level"] = "中等"
            elif complexity_score < 30:
                metrics["complexity_level"] = "复杂"
            else:
                metrics["complexity_level"] = "非常复杂"
            
            return metrics
            
        except SyntaxError:
            return {
                "cyclomatic_complexity": 0,
                "function_count": 0,
                "class_count": 0,
                "average_function_length": 0,
                "max_nesting_depth": 0,
                "import_count": 0,
                "complexity_level": "无法分析"
            }
    
    @staticmethod
    def analyze_generic(content: str) -> Dict[str, Any]:
        """通用代码复杂度分析"""
        lines = content.split('\n')
        
        # 简单的复杂度估算
        metrics = {
            "line_count": len(lines),
            "estimated_complexity": 0
        }
        
        # 基于关键词估算复杂度
        complexity_patterns = [
            (r'\bif\b', 1),
            (r'\belse\b', 1),
            (r'\bfor\b', 2),
            (r'\bwhile\b', 2),
            (r'\btry\b', 1),
            (r'\bcatch\b', 1),
            (r'\bswitch\b', 2),
            (r'\bcase\b', 1)
        ]
        
        for line in lines:
            line_lower = line.lower()
            for pattern, weight in complexity_patterns:
                if re.search(pattern, line_lower):
                    metrics["estimated_complexity"] += weight
        
        # 评估复杂度等级
        if metrics["estimated_complexity"] < 10:
            metrics["complexity_level"] = "简单"
        elif metrics["estimated_complexity"] < 30:
            metrics["complexity_level"] = "中等"
        elif metrics["estimated_complexity"] < 50:
            metrics["complexity_level"] = "复杂"
        else:
            metrics["complexity_level"] = "非常复杂"
        
        return metrics
    
    @staticmethod
    def calculate_entropy(content: str) -> float:
        """计算香农熵"""
        if not content:
            return 0.0
        
        from collections import Counter
        
        char_counts = Counter(content)
        total = len(content)
        entropy = 0.0
        
        for count in char_counts.values():
            p = count / total
            entropy -= p * math.log2(p) if p > 0 else 0
        
        return entropy

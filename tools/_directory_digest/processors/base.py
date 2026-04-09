"""
Directory Digest - 处理器模块
包含文本文件、源代码、配置文件等各类文件的处理策略
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

# 导入基础模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from base import (
    ProcessingStrategy,
    FileType,
    FileMetadata,
    FileDigest,
    STRATEGY_CONFIGS,
)

# 尝试导入分析器模块
try:
    from analyzers.semantics.base import (
        HumanReadableSummary,
        SourceCodeAnalysis,
        SmartTextProcessor,
    )
    from analyzers.semantics.sheets import ConfigAnalysisResult
    ANALYZERS_AVAILABLE = True
except ImportError:
    ANALYZERS_AVAILABLE = False
    # 定义简单的后备数据类
    @dataclass
    class HumanReadableSummary:
        """后备人类可读摘要"""
        title: Optional[str] = None
        line_count: int = 0
        word_count: int = 0
        character_count: int = 0
        encoding: Optional[str] = None
        first_lines: List[str] = field(default_factory=list)
        last_lines: List[str] = field(default_factory=list)
        summary: Optional[str] = None
        
        def to_dict(self) -> Dict:
            return {
                "title": self.title,
                "line_count": self.line_count,
                "word_count": self.word_count,
                "character_count": self.character_count,
                "encoding": self.encoding,
                "first_lines": self.first_lines,
                "last_lines": self.last_lines,
                "summary": self.summary
            }
    
    @dataclass
    class SourceCodeAnalysis:
        """后备源代码分析"""
        language: str = "unknown"
        total_lines: int = 0
        code_lines: int = 0
        comment_lines: int = 0
        blank_lines: int = 0
        imports: List[str] = field(default_factory=list)
        functions: List[Dict] = field(default_factory=list)
        classes: List[Dict] = field(default_factory=list)
        
        def to_dict(self) -> Dict:
            return {
                "language": self.language,
                "total_lines": self.total_lines,
                "code_lines": self.code_lines,
                "comment_lines": self.comment_lines,
                "blank_lines": self.blank_lines,
                "imports": self.imports[:20],
                "functions": self.functions[:20],
                "classes": self.classes[:20]
            }
    
    @dataclass
    class ConfigAnalysisResult:
        """后备配置分析结果"""
        keys: List[str] = field(default_factory=list)
        sections: List[str] = field(default_factory=list)
        structure_summary: Optional[str] = None
        
        def to_dict(self) -> Dict:
            return {
                "keys": self.keys[:20],
                "sections": self.sections[:20],
                "structure_summary": self.structure_summary
            }


# ==================== 处理器基类 ====================

class BaseFileProcessor(ABC):
    """文件处理器基类"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.max_full_content_size = self.config.get('max_full_content_size', 1024 * 1024)  # 1MB
    
    @abstractmethod
    def can_handle(self, file_digest: FileDigest) -> bool:
        """判断是否能处理此文件"""
        pass
    
    @abstractmethod
    def process(self, file_digest: FileDigest, content: str, mode: str = "framework", 
                strategy: ProcessingStrategy = ProcessingStrategy.SUMMARY_ONLY) -> FileDigest:
        """
        处理文件内容
        
        Args:
            file_digest: 文件摘要对象
            content: 文件内容
            mode: 输出模式 ("full", "framework", "sort")
            strategy: 处理策略
            
        Returns:
            更新后的 FileDigest
        """
        pass
    
    def _should_include_full_content(self, file_digest: FileDigest, mode: str) -> bool:
        """判断是否应该包含完整内容"""
        if mode != "full":
            return False
        if file_digest.metadata.size > self.max_full_content_size:
            return False
        return True


# ==================== 文本文件处理器 ====================

class TextFileProcessor(BaseFileProcessor):
    """人类可读文本文件处理器"""
    
    TEXT_EXTENSIONS = {'.txt', '.md', '.markdown', '.rst', '.tex', '.html', '.htm', '.cmt'}
    
    def can_handle(self, file_digest: FileDigest) -> bool:
        file_type = file_digest.metadata.file_type
        if file_type in (FileType.CRITICAL_DOCS, FileType.REFERENCE_DOCS):
            return True
        
        suffix = file_digest.metadata.path.suffix.lower()
        if suffix in self.TEXT_EXTENSIONS:
            return True
        
        return False
    
    def process(self, file_digest: FileDigest, content: str, mode: str = "framework", 
                strategy: ProcessingStrategy = ProcessingStrategy.SUMMARY_ONLY) -> FileDigest:
        
        if not content:
            return file_digest
        
        filepath = file_digest.metadata.path
        
        # 处理完整内容
        if self._should_include_full_content(file_digest, mode):
            if strategy == ProcessingStrategy.FULL_CONTENT:
                file_digest.full_content = content
        
        # 生成摘要
        summary = self._generate_summary(filepath, content, strategy)
        file_digest.human_readable_summary = summary
        
        return file_digest
    
    def _generate_summary(self, filepath: Path, content: str, 
                         strategy: ProcessingStrategy) -> HumanReadableSummary:
        """生成文本摘要"""
        lines = content.split('\n')
        line_count = len(lines)
        
        # 基础统计
        words = re.findall(r'\b[\w\u4e00-\u9fff]+\b', content)
        word_count = len(words)
        
        # 提取标题
        title = self._extract_title(filepath, lines)
        
        # 提取首尾行
        first_lines = lines[:min(10, len(lines))]
        last_lines = lines[-min(5, len(lines)):] if len(lines) > 5 else []
        
        # 根据策略调整
        if strategy == ProcessingStrategy.HEADER_WITH_STATS:
            # 只保留头部
            first_lines = first_lines[:20]
            last_lines = []
        
        # 生成综合摘要
        summary_text = self._generate_summary_text(filepath, lines, strategy)
        
        return HumanReadableSummary(
            title=title,
            line_count=line_count,
            word_count=word_count,
            character_count=len(content),
            encoding=self._detect_encoding(content),
            first_lines=first_lines,
            last_lines=last_lines,
            summary=summary_text
        )
    
    def _extract_title(self, filepath: Path, lines: List[str]) -> Optional[str]:
        """提取标题"""
        # 1. 从文件名
        filename = filepath.stem
        if filename and len(filename) > 2:
            cleaned = filename.replace('_', ' ').replace('-', ' ').title()
            if 3 <= len(cleaned) <= 100:
                return cleaned
        
        # 2. 从内容
        for line in lines[:10]:
            line = line.strip()
            if not line:
                continue
            
            # Markdown 标题
            md_match = re.match(r'^#+\s+(.+)$', line)
            if md_match:
                return md_match.group(1).strip()
            
            # 其他标题模式
            if len(line) > 3 and len(line) < 100:
                if line[0].isalpha() or line[0] in ('【', '[', '*'):
                    return line
        
        return None
    
    def _detect_encoding(self, content: str) -> str:
        """检测编码"""
        try:
            content.encode('utf-8')
            return 'utf-8'
        except UnicodeEncodeError:
            return 'unknown'
    
    def _generate_summary_text(self, filepath: Path, lines: List[str], 
                               strategy: ProcessingStrategy) -> str:
        """生成摘要文本"""
        parts = []
        
        # 基础信息
        suffix = filepath.suffix.lower()
        parts.append(f"File type: {suffix[1:] if suffix else 'text'}")
        parts.append(f"Total lines: {len(lines)}")
        
        # 根据策略添加内容
        if strategy == ProcessingStrategy.FULL_CONTENT:
            parts.append("\n[FULL CONTENT INCLUDED]")
        elif strategy == ProcessingStrategy.SUMMARY_ONLY:
            # 包含前几行预览
            preview = '\n'.join(lines[:min(20, len(lines))])
            if len(lines) > 20:
                preview += f"\n... [and {len(lines) - 20} more lines]"
            parts.append(f"\nPreview:\n{preview}")
        
        return '\n'.join(parts)


# ==================== 源代码处理器 ====================

class SourceCodeProcessor(BaseFileProcessor):
    """源代码文件处理器"""
    
    CODE_EXTENSIONS = {
        '.py', '.java', '.cpp', '.c', '.h', '.hpp', '.js', '.ts', 
        '.jsx', '.tsx', '.go', '.rs', '.rb', '.php', '.swift',
        '.sh', '.bash', '.ps1', '.bat', '.cmd', '.css', '.scss'
    }
    
    def can_handle(self, file_digest: FileDigest) -> bool:
        if file_digest.metadata.file_type == FileType.SOURCE_CODE:
            return True
        
        suffix = file_digest.metadata.path.suffix.lower()
        if suffix in self.CODE_EXTENSIONS:
            return True
        
        return False
    
    def process(self, file_digest: FileDigest, content: str, mode: str = "framework", 
                strategy: ProcessingStrategy = ProcessingStrategy.CODE_SKELETON) -> FileDigest:
        
        if not content:
            return file_digest
        
        filepath = file_digest.metadata.path
        
        # 处理完整内容
        if self._should_include_full_content(file_digest, mode):
            file_digest.full_content = content
        
        # 分析代码
        analysis = self._analyze_code(filepath, content, strategy)
        file_digest.source_code_analysis = analysis
        
        # 也生成简单摘要
        if strategy != ProcessingStrategy.METADATA_ONLY:
            summary = self._generate_code_summary(filepath, content, analysis)
            file_digest.human_readable_summary = summary
        
        return file_digest
    
    def _analyze_code(self, filepath: Path, content: str, 
                      strategy: ProcessingStrategy) -> SourceCodeAnalysis:
        """分析源代码"""
        lines = content.split('\n')
        total_lines = len(lines)
        
        # 统计行类型
        blank_lines = 0
        comment_lines = 0
        code_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
            elif self._is_comment_line(stripped, filepath.suffix.lower()):
                comment_lines += 1
            else:
                code_lines += 1
        
        # 提取导入、函数、类
        imports = self._extract_imports(content, filepath.suffix.lower())
        functions = self._extract_functions(content, filepath.suffix.lower())
        classes = self._extract_classes(content, filepath.suffix.lower())
        
        # 语言识别
        language = self._identify_language(filepath)
        
        return SourceCodeAnalysis(
            language=language,
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            imports=imports,
            functions=functions,
            classes=classes
        )
    
    def _is_comment_line(self, line: str, suffix: str) -> bool:
        """判断是否为注释行"""
        if suffix in ('.py', '.sh', '.bash', '.rb'):
            return line.startswith('#')
        elif suffix in ('.java', '.cpp', '.c', '.h', '.js', '.ts'):
            return line.startswith('//') or line.startswith('/*')
        return False
    
    def _extract_imports(self, content: str, suffix: str) -> List[str]:
        """提取导入语句"""
        imports = []
        
        if suffix == '.py':
            # Python 导入
            patterns = [
                r'^import\s+([a-zA-Z_][a-zA-Z0-9_\.]*)',
                r'^from\s+([a-zA-Z_][a-zA-Z0-9_\.]*)',
            ]
        elif suffix in ('.js', '.ts'):
            # JavaScript/TypeScript 导入
            patterns = [
                r'^import\s+.*from\s+[\'"]([^\'"]+)[\'"]',
                r'^const\s+.*=\s+require\([\'"]([^\'"]+)[\'"]\)',
            ]
        elif suffix in ('.java', '.cpp', '.c'):
            # Java/C/C++ 导入/包含
            patterns = [
                r'^import\s+([a-zA-Z0-9_\.]+);',
                r'^#include\s+[<"]([^>"]+)[>"]',
            ]
        else:
            patterns = []
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            imports.extend(matches)
        
        return list(set(imports))[:50]  # 去重并限制数量
    
    def _extract_functions(self, content: str, suffix: str) -> List[Dict]:
        """提取函数定义"""
        functions = []
        
        if suffix == '.py':
            # Python 函数
            func_pattern = r'^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        elif suffix in ('.js', '.ts'):
            func_pattern = r'^\s*(?:function|const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\(|\s*=\s*(?:\([^)]*\)\s*=>|function))'
        else:
            func_pattern = r'^\s*(?:[a-zA-Z_][a-zA-Z0-9_:\*&]+\s+)+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            match = re.search(func_pattern, line)
            if match:
                func_name = match.group(1)
                if func_name not in ('if', 'for', 'while', 'switch', 'return'):
                    functions.append({
                        "name": func_name,
                        "line": i + 1
                    })
        
        return functions[:50]
    
    def _extract_classes(self, content: str, suffix: str) -> List[Dict]:
        """提取类定义"""
        classes = []
        
        class_pattern = r'^\s*(?:class|struct)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            match = re.search(class_pattern, line)
            if match:
                classes.append({
                    "name": match.group(1),
                    "line": i + 1
                })
        
        return classes[:50]
    
    def _identify_language(self, filepath: Path) -> str:
        """识别编程语言"""
        suffix_map = {
            '.py': 'python',
            '.java': 'java',
            '.cpp': 'cpp', '.cc': 'cpp', '.hpp': 'cpp_header',
            '.c': 'c', '.h': 'c_header',
            '.js': 'javascript', '.jsx': 'jsx',
            '.ts': 'typescript', '.tsx': 'tsx',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.sh': 'shell', '.bash': 'shell',
            '.ps1': 'powershell',
            '.bat': 'batch', '.cmd': 'batch',
            '.css': 'css', '.scss': 'scss', '.less': 'less',
        }
        return suffix_map.get(filepath.suffix.lower(), 'unknown')
    
    def _generate_code_summary(self, filepath: Path, content: str, 
                               analysis: SourceCodeAnalysis) -> HumanReadableSummary:
        """生成代码摘要"""
        lines = content.split('\n')
        
        summary_parts = [
            f"Language: {analysis.language}",
            f"Total lines: {analysis.total_lines}",
            f"Code lines: {analysis.code_lines}",
            f"Comment lines: {analysis.comment_lines}",
        ]
        
        if analysis.functions:
            summary_parts.append(f"Functions: {len(analysis.functions)}")
        if analysis.classes:
            summary_parts.append(f"Classes: {len(analysis.classes)}")
        if analysis.imports:
            summary_parts.append(f"Imports: {len(analysis.imports)}")
        
        return HumanReadableSummary(
            title=filepath.name,
            line_count=analysis.total_lines,
            character_count=len(content),
            first_lines=lines[:10],
            summary='\n'.join(summary_parts)
        )


# ==================== 配置文件处理器 ====================

class ConfigFileProcessor(BaseFileProcessor):
    """配置文件处理器"""
    
    CONFIG_EXTENSIONS = {
        '.yaml', '.yml', '.json', '.xml', '.toml', '.ini', 
        '.cfg', '.conf', '.env', '.properties', '.tf', '.tls'
    }
    
    def can_handle(self, file_digest: FileDigest) -> bool:
        if file_digest.metadata.file_type == FileType.TEXT_DATA:
            return True
        
        suffix = file_digest.metadata.path.suffix.lower()
        if suffix in self.CONFIG_EXTENSIONS:
            return True
        
        return False
    
    def process(self, file_digest: FileDigest, content: str, mode: str = "framework", 
                strategy: ProcessingStrategy = ProcessingStrategy.STRUCTURE_EXTRACT) -> FileDigest:
        
        if not content:
            return file_digest
        
        filepath = file_digest.metadata.path
        
        # 处理完整内容
        if self._should_include_full_content(file_digest, mode):
            file_digest.full_content = content
        
        # 分析配置结构
        config_analysis = self._analyze_config(filepath, content, strategy)
        
        # 生成摘要
        summary = self._generate_config_summary(filepath, content, config_analysis, strategy)
        file_digest.human_readable_summary = summary
        
        return file_digest
    
    def _analyze_config(self, filepath: Path, content: str, 
                       strategy: ProcessingStrategy) -> ConfigAnalysisResult:
        """分析配置文件结构"""
        suffix = filepath.suffix.lower()
        keys = []
        sections = []
        structure_summary = None
        
        if suffix in ('.yaml', '.yml'):
            keys, sections, structure_summary = self._analyze_yaml(content)
        elif suffix == '.json':
            keys, sections, structure_summary = self._analyze_json(content)
        elif suffix in ('.ini', '.cfg', '.conf'):
            keys, sections, structure_summary = self._analyze_ini(content)
        elif suffix == '.toml':
            keys, sections, structure_summary = self._analyze_toml(content)
        elif suffix == '.xml':
            keys, sections, structure_summary = self._analyze_xml(content)
        
        return ConfigAnalysisResult(
            keys=keys,
            sections=sections,
            structure_summary=structure_summary
        )
    
    def _analyze_yaml(self, content: str) -> Tuple[List[str], List[str], Optional[str]]:
        """分析 YAML 配置"""
        keys = []
        sections = []
        
        # 简单的键提取（不依赖 PyYAML）
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and ':' in line:
                key_part = line.split(':', 1)[0].strip()
                if key_part and not key_part.startswith('-'):
                    keys.append(key_part)
        
        structure_summary = f"YAML config with {len(set(keys))} top-level keys"
        return list(set(keys)), sections, structure_summary
    
    def _analyze_json(self, content: str) -> Tuple[List[str], List[str], Optional[str]]:
        """分析 JSON 配置"""
        keys = []
        sections = []
        
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                keys = list(data.keys())
                structure_summary = f"JSON object with {len(keys)} keys"
            else:
                structure_summary = f"JSON {type(data).__name__}"
        except json.JSONDecodeError:
            structure_summary = "Invalid JSON"
        
        return keys, sections, structure_summary
    
    def _analyze_ini(self, content: str) -> Tuple[List[str], List[str], Optional[str]]:
        """分析 INI 配置"""
        keys = []
        sections = []
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                sections.append(line[1:-1])
            elif line and not line.startswith('#') and '=' in line:
                key = line.split('=', 1)[0].strip()
                if key:
                    keys.append(key)
        
        structure_summary = f"INI config with {len(sections)} sections, {len(set(keys))} keys"
        return list(set(keys)), sections, structure_summary
    
    def _analyze_toml(self, content: str) -> Tuple[List[str], List[str], Optional[str]]:
        """分析 TOML 配置"""
        # 类似 INI 分析
        return self._analyze_ini(content)
    
    def _analyze_xml(self, content: str) -> Tuple[List[str], List[str], Optional[str]]:
        """分析 XML 配置"""
        keys = []
        sections = []
        
        # 简单标签提取
        tags = re.findall(r'<(\w+)[^>]*>', content)
        keys = list(set(tags))
        
        structure_summary = f"XML with {len(keys)} unique tags"
        return keys, sections, structure_summary
    
    def _generate_config_summary(self, filepath: Path, content: str,
                                 config_analysis: ConfigAnalysisResult,
                                 strategy: ProcessingStrategy) -> HumanReadableSummary:
        """生成配置文件摘要"""
        lines = content.split('\n')
        
        summary_parts = [f"Config type: {filepath.suffix[1:]}"]
        
        if config_analysis.structure_summary:
            summary_parts.append(config_analysis.structure_summary)
        
        if config_analysis.sections:
            summary_parts.append(f"Sections: {', '.join(config_analysis.sections[:10])}")
            if len(config_analysis.sections) > 10:
                summary_parts[-1] += f" (+{len(config_analysis.sections) - 10} more)"
        
        if config_analysis.keys:
            summary_parts.append(f"Keys: {', '.join(config_analysis.keys[:15])}")
            if len(config_analysis.keys) > 15:
                summary_parts[-1] += f" (+{len(config_analysis.keys) - 15} more)"
        
        return HumanReadableSummary(
            title=filepath.name,
            line_count=len(lines),
            character_count=len(content),
            first_lines=lines[:15],
            summary='\n'.join(summary_parts)
        )


# ==================== 数据文件处理器 ====================

class DataFileProcessor(BaseFileProcessor):
    """数据文件处理器（CSV、TSV、日志等）"""
    
    DATA_EXTENSIONS = {'.csv', '.tsv', '.log', '.out', '.err', '.dat', '.txt'}
    
    def can_handle(self, file_digest: FileDigest) -> bool:
        suffix = file_digest.metadata.path.suffix.lower()
        if suffix in self.DATA_EXTENSIONS:
            # 避免与其他处理器冲突
            if file_digest.metadata.file_type not in (FileType.CRITICAL_DOCS, FileType.REFERENCE_DOCS, FileType.SOURCE_CODE):
                return True
        return False
    
    def process(self, file_digest: FileDigest, content: str, mode: str = "framework", 
                strategy: ProcessingStrategy = ProcessingStrategy.HEADER_WITH_STATS) -> FileDigest:
        
        if not content:
            return file_digest
        
        filepath = file_digest.metadata.path
        
        # 数据文件一般不包含完整内容（除非特别小）
        if mode == "full" and file_digest.metadata.size < 100 * 1024:  # <100KB
            file_digest.full_content = content
        
        # 生成数据摘要
        summary = self._generate_data_summary(filepath, content, strategy)
        file_digest.human_readable_summary = summary
        
        return file_digest
    
    def _generate_data_summary(self, filepath: Path, content: str,
                               strategy: ProcessingStrategy) -> HumanReadableSummary:
        """生成数据文件摘要"""
        lines = content.split('\n')
        line_count = len(lines)
        
        suffix = filepath.suffix.lower()
        
        # 计算统计信息
        stats = self._calculate_data_stats(content, suffix)
        
        # 提取头部
        header_lines = self._extract_header(lines, suffix)
        
        summary_parts = [
            f"Data type: {suffix[1:] if suffix else 'text'}",
            f"Total lines: {line_count}"
        ]
        
        for key, value in stats.items():
            summary_parts.append(f"{key}: {value}")
        
        return HumanReadableSummary(
            title=filepath.name,
            line_count=line_count,
            character_count=len(content),
            first_lines=header_lines,
            summary='\n'.join(summary_parts)
        )
    
    def _calculate_data_stats(self, content: str, suffix: str) -> Dict[str, Any]:
        """计算数据统计信息"""
        stats = {}
        lines = content.split('\n')
        
        if suffix == '.csv':
            # CSV 统计
            non_empty_lines = [l for l in lines if l.strip()]
            if non_empty_lines:
                stats["Data rows (approx)"] = len(non_empty_lines)
                # 估算列数
                first_line = non_empty_lines[0]
                stats["Columns (approx)"] = first_line.count(',') + 1
        
        elif suffix in ('.log', '.out', '.err'):
            # 日志统计
            error_count = sum(1 for l in lines if 'error' in l.lower() or 'exception' in l.lower())
            warning_count = sum(1 for l in lines if 'warn' in l.lower())
            
            if error_count > 0:
                stats["Errors"] = error_count
            if warning_count > 0:
                stats["Warnings"] = warning_count
        
        # 通用统计
        stats["File size"] = f"{len(content)} chars"
        
        return stats
    
    def _extract_header(self, lines: List[str], suffix: str) -> List[str]:
        """提取数据文件头部"""
        header_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # 保留注释行
            if stripped and stripped.startswith(('#', '//', '/*', '*', '!')):
                header_lines.append(line)
                continue
            
            # 保留空行直到遇到数据
            if not stripped and header_lines:
                header_lines.append(line)
                continue
            
            # 数据文件的前几行
            if i < 20:
                header_lines.append(line)
            
            # 如果看起来像数据行了，停止
            if i > 10 and stripped and not stripped.startswith(('#', '//', '/*')):
                # 检查是否是纯数据行
                if len(re.findall(r'[a-zA-Z]', stripped)) / len(stripped) < 0.3:
                    break
        
        return header_lines[:30]


# ==================== 处理器注册表 ====================

class FileProcessorRegistry:
    """文件处理器注册表"""
    
    def __init__(self):
        self.processors: List[BaseFileProcessor] = []
    
    def register(self, processor: BaseFileProcessor):
        """注册处理器"""
        self.processors.append(processor)
    
    def get_processor(self, file_digest: FileDigest) -> Optional[BaseFileProcessor]:
        """获取适合此文件的处理器"""
        for processor in self.processors:
            if processor.can_handle(file_digest):
                return processor
        return None


# ==================== 公共 API ====================

def create_default_registry(config: Optional[Dict] = None) -> FileProcessorRegistry:
    """创建默认处理器注册表"""
    registry = FileProcessorRegistry()
    
    # 按优先级顺序注册（先注册的优先级高）
    registry.register(TextFileProcessor(config))
    registry.register(SourceCodeProcessor(config))
    registry.register(ConfigFileProcessor(config))
    registry.register(DataFileProcessor(config))
    
    return registry


__all__ = [
    # 基类
    'BaseFileProcessor',
    
    # 具体处理器
    'TextFileProcessor',
    'SourceCodeProcessor',
    'ConfigFileProcessor',
    'DataFileProcessor',
    
    # 注册表
    'FileProcessorRegistry',
    'create_default_registry',
]

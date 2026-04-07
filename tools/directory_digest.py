#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
directory_digest.py - 目录知识摘要器

将文件系统递归"消化"为LLM可理解的上下文摘要。

三种文件分类策略：
1. 人类可读文本文件 (HumanReadable) - .txt, .md, .rst, 配置文件等
2. 源代码文件 (SourceCode) - .py, .java, .cpp 等编程语言文件  
3. 二进制文件 (Binary) - 图像、压缩包、可执行文件等

两种输出模式：
- 全量模式 (full): 包含人类可读文本文件的完整内容
- 框架模式 (framework): 所有文件都只输出摘要/元信息
"""

import os
import sys
import json
import hashlib
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import re
from enum import Enum


# ==================== 数据类型定义 ====================

class FileType(Enum):
    """文件类型枚举"""
    HUMAN_READABLE = "human_readable"    # 人类可读文本
    SOURCE_CODE = "source_code"          # 源代码
    BINARY = "binary"                    # 二进制文件
    UNKNOWN = "unknown"                  # 未知类型


@dataclass
class FileMetadata:
    """文件元数据基类"""
    path: Path
    size: int
    modified_time: datetime
    created_time: datetime
    file_type: FileType
    mime_type: Optional[str] = None
    md5_hash: Optional[str] = None
    sha256_hash: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "path": str(self.path),
            "size": self.size,
            "modified_time": self.modified_time.isoformat(),
            "created_time": self.created_time.isoformat(),
            "file_type": self.file_type.value,
            "mime_type": self.mime_type,
            "md5_hash": self.md5_hash,
            "sha256_hash": self.sha256_hash
        }


@dataclass
class HumanReadableSummary:
    """人类可读文本摘要"""
    title: Optional[str] = None
    line_count: int = 0
    word_count: int = 0
    character_count: int = 0
    language: Optional[str] = None
    encoding: Optional[str] = None
    first_lines: List[str] = field(default_factory=list)
    last_lines: List[str] = field(default_factory=list)
    key_sections: List[Tuple[str, str]] = field(default_factory=list)
    summary: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "line_count": self.line_count,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "language": self.language,
            "encoding": self.encoding,
            "first_lines": self.first_lines,
            "last_lines": self.last_lines,
            "key_sections": [{"title": t, "content": c[:200]} for t, c in self.key_sections],
            "summary": self.summary
        }


@dataclass
class SourceCodeAnalysis:
    """源代码分析结果"""
    language: str
    total_lines: int
    code_lines: int
    comment_lines: int
    blank_lines: int
    imports: List[str] = field(default_factory=list)
    functions: List[Dict] = field(default_factory=list)
    classes: List[Dict] = field(default_factory=list)
    global_vars: List[str] = field(default_factory=list)
    constants: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "language": self.language,
            "total_lines": self.total_lines,
            "code_lines": self.code_lines,
            "comment_lines": self.comment_lines,
            "blank_lines": self.blank_lines,
            "imports": self.imports,
            "functions": [f["name"] for f in self.functions],
            "classes": [c["name"] for c in self.classes],
            "global_vars": self.global_vars,
            "constants": self.constants,
            "dependencies": self.dependencies
        }


@dataclass
class FileDigest:
    """单个文件摘要"""
    metadata: FileMetadata
    full_content: Optional[str] = None
    human_readable_summary: Optional[HumanReadableSummary] = None
    source_code_analysis: Optional[SourceCodeAnalysis] = None
    
    def to_dict(self, mode: str = "framework") -> Dict:
        """转换为字典，根据模式决定输出内容"""
        result = {
            "metadata": self.metadata.to_dict()
        }
        
        if mode == "full" and self.full_content and self.metadata.file_type == FileType.HUMAN_READABLE:
            result["full_content"] = self.full_content
        elif self.human_readable_summary:
            result["summary"] = self.human_readable_summary.to_dict()
        
        if self.source_code_analysis:
            result["source_analysis"] = self.source_code_analysis.to_dict()
        
        return result


@dataclass
class DirectoryStructure:
    """目录结构表示"""
    path: Path
    files: List[FileDigest] = field(default_factory=list)
    subdirectories: Dict[str, 'DirectoryStructure'] = field(default_factory=dict)
    
    def to_dict(self, mode: str = "framework") -> Dict:
        """转换为嵌套字典结构"""
        return {
            "path": str(self.path),
            "files": [f.to_dict(mode) for f in self.files],
            "subdirectories": {name: d.to_dict(mode) for name, d in self.subdirectories.items()}
        }


# ==================== 文件类型检测器 ====================

class FileTypeDetector:
    """智能文件类型检测器"""
    
    # 扩展名到类型的映射（优先级1）
    EXTENSION_MAPPING = {
        # 人类可读文本
        FileType.HUMAN_READABLE: [
            '.txt', '.md', '.markdown', '.rst', '.tex', '.latex',
            '.json', '.yaml', '.yml', '.xml', '.html', '.htm', '.csv',
            '.ini', '.cfg', '.conf', '.toml', '.properties',
            '.log', '.out', '.err'
        ],
        # 源代码
        FileType.SOURCE_CODE: [
            '.py', '.java', '.cpp', '.c', '.h', '.hpp', '.cc',
            '.js', '.ts', '.jsx', '.tsx', '.vue',
            '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
            '.m', '.mm', '.cs', '.fs', '.vb',
            '.sh', '.bash', '.zsh', '.fish', '.ps1',
            '.sql', '.pl', '.pm', '.r', '.lua', '.dart'
        ],
        # 二进制文件（部分常见）
        FileType.BINARY: [
            '.exe', '.dll', '.so', '.dylib', '.a', '.lib',
            '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.ico',
            '.mp3', '.mp4', '.avi', '.mkv', '.mov', '.wav',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.bin', '.dat', '.db', '.sqlite', '.sqlite3'
        ]
    }
    
    @staticmethod
    def detect_by_extension(filepath: Path) -> Optional[FileType]:
        """通过扩展名检测文件类型"""
        suffix = filepath.suffix.lower()
        
        for file_type, extensions in FileTypeDetector.EXTENSION_MAPPING.items():
            if suffix in extensions:
                return file_type
        
        return None
    
    @staticmethod
    def detect_by_content(filepath: Path) -> FileType:
        """通过内容分析检测文件类型"""
        try:
            with open(filepath, 'rb') as f:
                sample = f.read(4096)
                
                # 检测空字节（二进制文件特征）
                if b'\x00' in sample:
                    return FileType.BINARY
                
                # 检测可打印字符比例
                printable_count = 0
                for byte in sample:
                    if 32 <= byte <= 126 or byte in (9, 10, 13):
                        printable_count += 1
                
                printable_ratio = printable_count / len(sample) if sample else 0
                
                if printable_ratio < 0.7:
                    return FileType.BINARY
                
                # 检测源代码特征
                try:
                    decoded = sample.decode('utf-8', errors='ignore')
                    if FileTypeDetector._looks_like_source_code(decoded):
                        return FileType.SOURCE_CODE
                except:
                    pass
                
                return FileType.HUMAN_READABLE
                
        except Exception:
            return FileType.BINARY
    
    @staticmethod
    def _looks_like_source_code(content: str) -> bool:
        """判断内容是否像源代码"""
        patterns = [
            r'^\s*import\s+',
            r'^\s*package\s+',
            r'^\s*#include\s+',
            r'^\s*def\s+\w+\s*\(',
            r'^\s*function\s+\w+',
            r'^\s*class\s+\w+',
            r'^\s*public\s+',
            r'^\s*private\s+',
            r'^\s*protected\s+',
            r'^\s*static\s+',
            r'^\s*const\s+',
            r'^\s*let\s+\w+\s*=',
            r'^\s*var\s+\w+\s*=',
            r'^\s*console\.log',
            r'^\s*print\(',
            r'^\s*System\.out\.',
            r'^\s*//',
            r'^\s*/\*',
            r'^\s*\*/',
            r'^\s*#\s*',
        ]
        
        lines = content.split('\n')[:50]
        code_pattern_count = 0
        
        for line in lines:
            for pattern in patterns:
                if re.search(pattern, line):
                    code_pattern_count += 1
                    break
        
        return code_pattern_count >= 3
    
    @staticmethod
    def detect(filepath: Path) -> FileType:
        """综合检测文件类型"""
        type_by_ext = FileTypeDetector.detect_by_extension(filepath)
        if type_by_ext:
            return type_by_ext
        
        return FileTypeDetector.detect_by_content(filepath)


# ==================== 摘要生成器 ====================

class HumanReadableSummarizer:
    """人类可读文本摘要生成器"""
    
    @staticmethod
    def summarize(filepath: Path, content: str, max_lines: int = 10) -> HumanReadableSummary:
        """生成人类可读文本摘要"""
        # 待实现
        return HumanReadableSummary(
            line_count=len(content.split('\n')),
            word_count=len(re.findall(r'\b\w+\b', content)),
            character_count=len(content)
        )
    
    @staticmethod
    def _detect_language(content: str) -> Optional[str]:
        """简单检测文本语言"""
        return None


class SourceCodeAnalyzer:
    """源代码分析器"""
    
    @staticmethod
    def analyze(filepath: Path, content: str) -> SourceCodeAnalysis:
        """分析源代码文件"""
        # 待实现
        lines = content.split('\n')
        return SourceCodeAnalysis(
            language=filepath.suffix[1:] if filepath.suffix else "unknown",
            total_lines=len(lines),
            code_lines=len(lines),
            comment_lines=0,
            blank_lines=0
        )


# ==================== 格式转换器 ====================

class OutputFormats(Enum):
    """支持的所有输出格式"""
    JSON = "json"
    YAML = "yaml"
    MARKDOWN = "md"
    HTML = "html"
    TOML = "toml"
    PLAINTEXT = "txt"


class FormatConverter:
    """格式转换器，支持多种输出格式"""
    
    @staticmethod
    def convert(digest_data: Dict, format: str) -> str:
        """转换为指定格式"""
        if format == "json":
            return json.dumps(digest_data, indent=2, ensure_ascii=False)
        elif format == "markdown" or format == "md":
            return FormatConverter._to_markdown(digest_data)
        elif format == "yaml":
            return FormatConverter._to_yaml(digest_data)
        elif format == "html":
            return FormatConverter._to_html(digest_data)
        elif format == "toml":
            return FormatConverter._to_toml(digest_data)
        elif format == "txt" or format == "text":
            return FormatConverter._to_plaintext(digest_data)
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    @staticmethod
    def _to_markdown(digest_data: Dict) -> str:
        """转换为Markdown格式"""
        # 待实现
        return "# 目录摘要\n\n待实现\n"
    
    @staticmethod
    def _to_yaml(digest_data: Dict) -> str:
        """转换为YAML格式"""
        # 待实现
        return "metadata:\n  generated_at: ...\n"
    
    @staticmethod
    def _to_html(digest_data: Dict) -> str:
        """转换为HTML格式"""
        # 待实现
        return "<html><body>待实现</body></html>"
    
    @staticmethod
    def _to_toml(digest_data: Dict) -> str:
        """转换为TOML格式"""
        # 待实现
        return '[metadata]\ngenerated_at = "..."'
    
    @staticmethod
    def _to_plaintext(digest_data: Dict) -> str:
        """转换为纯文本格式"""
        # 待实现
        return "目录摘要 - 待实现"


# ==================== 主摘要生成器 ====================

class DirectoryDigest:
    """目录摘要生成器"""
    
    def __init__(self, 
                 root_path: Union[str, Path],
                 config: Optional[Dict] = None):
        """
        初始化摘要生成器
        
        Args:
            root_path: 根目录路径
            config: 配置字典
        """
        self.root = Path(root_path).resolve()
        self.config = config or {}
        
        # 默认配置
        self.max_file_size = self.config.get('max_file_size', 10 * 1024 * 1024)
        self.ignore_patterns = self.config.get('ignore_patterns', [
            '*.pyc', '*.pyo', '*.so', '*.dll', '__pycache__', 
            '.git', '.svn', '.hg', '.DS_Store', '*.swp', '*.swo'
        ])
        
        self.file_type_detector = FileTypeDetector()
        self.human_summarizer = HumanReadableSummarizer()
        self.source_analyzer = SourceCodeAnalyzer()
        
        # 存储结果
        self.structure: Optional[DirectoryStructure] = None
        self.stats = {
            'total_files': 0,
            'human_readable': 0,
            'source_code': 0,
            'binary': 0,
            'total_size': 0,
            'processing_time': 0
        }
    
    def create_digest(self, mode: str = "framework") -> Dict:
        """
        创建目录摘要
        
        Args:
            mode: 输出模式，"framework"（框架）或 "full"（全量）
            
        Returns:
            摘要字典
        """
        import time
        start_time = time.time()
        
        # 构建目录结构
        self.structure = self._build_directory_structure(self.root)
        
        # 处理所有文件
        self._process_directory(self.structure, mode)
        
        # 更新统计信息
        self.stats['processing_time'] = time.time() - start_time
        
        # 生成最终输出
        return self._generate_output(mode)
    
    def _build_directory_structure(self, path: Path) -> DirectoryStructure:
        """递归构建目录结构"""
        structure = DirectoryStructure(path=path)
        
        try:
            for item in path.iterdir():
                # 检查是否应该忽略
                if self._should_ignore(item):
                    continue
                
                if item.is_dir():
                    # 递归处理子目录
                    sub_structure = self._build_directory_structure(item)
                    structure.subdirectories[item.name] = sub_structure
                else:
                    # 文件，先创建空的FileDigest，稍后填充
                    structure.files.append(FileDigest(
                        metadata=FileMetadata(
                            path=item,
                            size=item.stat().st_size,
                            modified_time=datetime.fromtimestamp(item.stat().st_mtime),
                            created_time=datetime.fromtimestamp(item.stat().st_ctime),
                            file_type=FileType.UNKNOWN,
                            mime_type=mimetypes.guess_type(str(item))[0]
                        )
                    ))
                    self.stats['total_files'] += 1
                    self.stats['total_size'] += item.stat().st_size
                    
        except PermissionError:
            print(f"警告: 无权限访问目录 {path}", file=sys.stderr)
        
        return structure
    
    def _should_ignore(self, path: Path) -> bool:
        """检查路径是否应该被忽略"""
        import fnmatch
        
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(path.name, pattern):
                return True
            if pattern.startswith('*') and path.name.endswith(pattern[1:]):
                return True
        
        return False
    
    def _process_directory(self, structure: DirectoryStructure, mode: str):
        """处理目录中的所有文件"""
        # 处理当前目录的文件
        for file_digest in structure.files:
            self._process_file(file_digest, mode)
        
        # 递归处理子目录
        for subdir in structure.subdirectories.values():
            self._process_directory(subdir, mode)
    
    def _process_file(self, file_digest: FileDigest, mode: str):
        """处理单个文件"""
        filepath = file_digest.metadata.path
        
        try:
            # 1. 检测文件类型
            file_type = self.file_type_detector.detect(filepath)
            file_digest.metadata.file_type = file_type
            
            # 更新统计
            if file_type == FileType.HUMAN_READABLE:
                self.stats['human_readable'] += 1
            elif file_type == FileType.SOURCE_CODE:
                self.stats['source_code'] += 1
            elif file_type == FileType.BINARY:
                self.stats['binary'] += 1
            
            # 2. 计算哈希值
            self._calculate_hashes(file_digest)
            
            # 3. 根据文件类型处理内容
            if file_type == FileType.HUMAN_READABLE:
                self._process_human_readable(file_digest, mode)
            elif file_type == FileType.SOURCE_CODE:
                self._process_source_code(file_digest, mode)
            # 二进制文件不需要额外处理
            
        except Exception as e:
            print(f"警告: 处理文件 {filepath} 时出错: {e}", file=sys.stderr)
            file_digest.metadata.file_type = FileType.BINARY
    
    def _calculate_hashes(self, file_digest: FileDigest):
        """计算文件的哈希值"""
        filepath = file_digest.metadata.path
        
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
                
                # MD5
                md5_hash = hashlib.md5(content).hexdigest()
                file_digest.metadata.md5_hash = md5_hash
                
        except Exception:
            pass
    
    def _process_human_readable(self, file_digest: FileDigest, mode: str):
        """处理人类可读文本文件"""
        filepath = file_digest.metadata.path
        
        try:
            # 读取文件内容
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(filepath, 'rb') as f:
                    raw_content = f.read()
                    content = raw_content.decode('latin-1', errors='ignore')
            except Exception:
                content = ""
        
        # 全量模式存储完整内容
        if mode == "full":
            file_digest.full_content = content
        
        # 生成摘要
        summary = self.human_summarizer.summarize(filepath, content)
        file_digest.human_readable_summary = summary
    
    def _process_source_code(self, file_digest: FileDigest, mode: str):
        """处理源代码文件"""
        filepath = file_digest.metadata.path
        
        try:
            # 读取文件内容
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(filepath, 'rb') as f:
                    raw_content = f.read()
                    content = raw_content.decode('latin-1', errors='ignore')
            except Exception:
                content = ""
        
        # 全量模式也存储源代码内容
        if mode == "full" and file_digest.metadata.size < self.max_file_size:
            file_digest.full_content = content
        
        # 分析源代码
        analysis = self.source_analyzer.analyze(filepath, content)
        file_digest.source_code_analysis = analysis
    
    def _generate_output(self, mode: str) -> Dict:
        """生成最终输出"""
        if not self.structure:
            return {}
        
        # 基础输出结构
        output = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "root_directory": str(self.root),
                "output_mode": mode,
                "statistics": self.stats
            },
            "structure": self.structure.to_dict(mode)
        }
        
        return output
    
    def save_output(self, output: Dict, format: str = "json", output_path: Optional[Path] = None):
        """保存输出到文件"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = format.lower()
            if ext == "markdown":
                ext = "md"
            output_path = self.root / f"directory_digest_{timestamp}.{ext}"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = FormatConverter.convert(output, format)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"摘要已保存到: {output_path}")
        return output_path


# ==================== 命令行接口 ====================

def main():
    """命令行入口点"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="目录知识摘要器 - 将文件系统递归消化为LLM可理解的上下文摘要",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s /path/to/directory --mode full --output json
  %(prog)s . --mode framework --output yaml --ignore ".git,*.pyc"
        """
    )
    
    parser.add_argument("directory", help="要分析的目录路径")
    parser.add_argument("--mode", choices=["full", "framework"], default="framework",
                       help="输出模式: full(全量) 或 framework(框架)")
    parser.add_argument("--output", choices=["json", "yaml", "md", "html", "toml", "txt"], 
                       default="json", help="输出格式")
    parser.add_argument("--ignore", default=".git,__pycache__,*.pyc,*.pyo",
                       help="忽略的模式，用逗号分隔")
    parser.add_argument("--max-size", type=int, default=10,
                       help="最大文件大小(MB)，超过此大小的文件只分析元信息")
    parser.add_argument("--save", help="输出文件路径，默认为目录名_digest.json")
    parser.add_argument("--verbose", action="store_true", help="显示详细输出")
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'max_file_size': args.max_size * 1024 * 1024,
        'ignore_patterns': [p.strip() for p in args.ignore.split(',') if p.strip()]
    }
    
    # 创建摘要器
    digest = DirectoryDigest(args.directory, config)
    
    # 生成摘要
    if args.verbose:
        print(f"开始分析目录: {args.directory}")
        print(f"模式: {args.mode}, 格式: {args.output}")
    
    output = digest.create_digest(args.mode)
    
    # 保存输出
    output_path = Path(args.save) if args.save else None
    saved_path = digest.save_output(output, args.output, output_path)
    
    # 显示统计信息
    stats = output['metadata']['statistics']
    print(f"\n摘要统计:")
    print(f"  总文件数: {stats['total_files']}")
    print(f"  人类可读文本: {stats['human_readable']}")
    print(f"  源代码文件: {stats['source_code']}")
    print(f"  二进制文件: {stats['binary']}")
    print(f"  总大小: {stats['total_size'] / (1024*1024):.2f} MB")
    print(f"  处理时间: {stats['processing_time']:.2f} 秒")
    
    return saved_path


if __name__ == "__main__":
    main()

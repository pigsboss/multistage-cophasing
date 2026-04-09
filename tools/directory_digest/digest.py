# -*- coding: utf-8 -*-
"""
digest.py - 目录摘要生成器主类
"""

import os
import sys
import hashlib
import mimetypes
import concurrent.futures
from pathlib import Path
from typing import Dict, Optional, Union, List, Any
from datetime import datetime
import fnmatch

from .constants import FileType, ProcessingStrategy, OutputFormats
from .types import DirectoryStructure, FileDigest, FileMetadata, StrategyConfig, HumanReadableSummary
from .core.rule_engine import RuleEngine
from .core.context_manager import ContextManager
from .analysis.summarizer import HumanReadableSummarizer
from .analysis.code_analyzer import SourceCodeAnalyzer
from .analysis.content_analyzer import ContentAnalyzer
from .analysis.text_processor import SmartTextProcessor
from .utils.detector import FileTypeDetector
from .utils.complexity_analyzer import ComplexityAnalyzer
from .utils.format_converter import FormatConverter


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
        self.max_file_size = self.config.get('max_file_size', 10 * 1024 * 1024 * 1024)  # 10GB
        self.ignore_patterns = self.config.get('ignore_patterns', [
            '*.pyc', '*.pyo', '*.so', '*.dll', '__pycache__', 
            '.git', '.svn', '.hg', '.DS_Store', '*.swp', '*.swo'
        ])
        self.use_parallel = self.config.get('use_parallel', False)
        self.max_workers = self.config.get('max_workers', os.cpu_count() or 4)
        
        # 新配置项
        self.rules_file = self.config.get('rules_file')
        self.context_size = self.config.get('context_size', 128000)
        
        # 初始化组件
        self.rule_engine = RuleEngine(self.rules_file)
        self.context_manager = ContextManager(self.context_size)
        self.file_type_detector = FileTypeDetector()
        self.human_summarizer = HumanReadableSummarizer()
        self.source_analyzer = SourceCodeAnalyzer()
        self.text_processor = SmartTextProcessor()
        self.content_analyzer = ContentAnalyzer()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.format_converter = FormatConverter()
        
        # 存储结果
        self.structure: Optional[DirectoryStructure] = None
        self.stats = {
            'total_files': 0,
            'critical_docs': 0,
            'reference_docs': 0,
            'source_code': 0,
            'text_data': 0,
            'binary_files': 0,
            'skipped_large_files': 0,
            'skipped_by_context': 0,
            'total_size': 0,
            'processing_time': 0
        }
    
    def create_digest(self, mode: str = "framework") -> Dict:
        """
        创建目录摘要
        
        Args:
            mode: 输出模式，"framework"（框架）、"full"（全量）或 "sort"（分类排序）
        """
        import time
        start_time = time.time()
        
        # 构建目录结构
        self.structure = self._build_directory_structure(self.root)
        
        # 处理所有文件
        if self.use_parallel and len(self.structure.files) > 10:
            self._process_directory_parallel(self.structure, mode)
        else:
            self._process_directory(self.structure, mode)
        
        # 更新统计信息
        self.stats['processing_time'] = time.time() - start_time
        
        # 根据模式生成输出
        if mode == "sort":
            return self._generate_sort_output()
        else:
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
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(path.name, pattern):
                return True
            if pattern.startswith('*') and path.name.endswith(pattern[1:]):
                return True
        
        return False
    
    def _process_directory_parallel(self, structure: DirectoryStructure, mode: str):
        """并行处理目录中的所有文件"""
        # 收集所有文件
        all_files = []
        def collect_files(node: DirectoryStructure):
            all_files.extend(node.files)
            for subdir in node.subdirectories.values():
                collect_files(subdir)
        
        collect_files(structure)
        
        # 使用线程池并行处理文件
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有处理任务
            future_to_file = {
                executor.submit(self._process_file_safe, file_digest, mode): file_digest
                for file_digest in all_files
            }
            
            # 等待所有任务完成
            for future in concurrent.futures.as_completed(future_to_file):
                file_digest = future_to_file[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"警告: 并行处理文件 {file_digest.metadata.path} 时出错: {e}", file=sys.stderr)
    
    def _process_file_safe(self, file_digest: FileDigest, mode: str):
        """安全的文件处理（用于并行处理）"""
        try:
            self._process_file(file_digest, mode)
        except Exception as e:
            print(f"警告: 处理文件 {file_digest.metadata.path} 时出错: {e}", file=sys.stderr)
    
    def _process_directory(self, structure: DirectoryStructure, mode: str):
        """串行处理目录中的所有文件"""
        for file_digest in structure.files:
            self._process_file(file_digest, mode)
        
        # 递归处理子目录
        for subdir in structure.subdirectories.values():
            self._process_directory(subdir, mode)
    
    def _process_file(self, file_digest: FileDigest, mode: str):
        """基于规则和上下文限制处理文件"""
        filepath = file_digest.metadata.path
        
        try:
            # 1. 获取处理策略
            strategy, force_binary = self.rule_engine.classify_file(filepath)
            
            # 2. 估算token消耗
            estimated_tokens = self.rule_engine.estimate_token_usage(filepath, strategy)
            
            # 3. 检查文件大小限制
            if file_digest.metadata.size > self.max_file_size:
                file_digest.metadata.file_type = FileType.BINARY_FILES
                file_digest.human_readable_summary = HumanReadableSummarizer.summarize(
                    filepath, 
                    f"[SKIPPED] File size ({file_digest.metadata.size / (1024*1024):.2f} MB) exceeds limit"
                )
                self.stats['skipped_large_files'] += 1
                self.stats['binary_files'] += 1
                return
            
            # 4. 检查上下文限制
            file_record = {
                "path": str(filepath),
                "strategy": strategy.value,
                "estimated_tokens": estimated_tokens,
                "size": file_digest.metadata.size,
            }
            
            # 尝试分配token
            if not self.context_manager.can_allocate(estimated_tokens):
                # 尝试降级策略
                downgraded_strategy = self.context_manager.downgrade_strategy(strategy)
                downgraded_tokens = self.rule_engine.estimate_token_usage(filepath, downgraded_strategy)
                
                if not self.context_manager.can_allocate(downgraded_tokens):
                    # 即使降级也无法分配，跳过此文件
                    self.stats['skipped_by_context'] += 1
                    file_digest.metadata.file_type = FileType.UNKNOWN
                    file_digest.human_readable_summary = HumanReadableSummarizer.summarize(
                        filepath,
                        f"[SKIPPED] Exceeds context window (estimated {estimated_tokens} tokens)"
                    )
                    return
                
                strategy = downgraded_strategy
                estimated_tokens = downgraded_tokens
            
            # 5. 分配token
            file_record["strategy"] = strategy.value
            file_record["estimated_tokens"] = estimated_tokens
            
            if not self.context_manager.allocate(estimated_tokens, file_record):
                self.stats['skipped_by_context'] += 1
                return
            
            # 6. 根据策略处理文件
            if force_binary or strategy == ProcessingStrategy.METADATA_ONLY:
                self._process_as_binary(file_digest)
            elif strategy in [ProcessingStrategy.FULL_CONTENT, ProcessingStrategy.SUMMARY_ONLY]:
                self._process_as_text(file_digest, mode, strategy)
            elif strategy == ProcessingStrategy.CODE_SKELETON:
                self._process_as_code(file_digest, mode, strategy)
            elif strategy == ProcessingStrategy.STRUCTURE_EXTRACT:
                self._process_as_config(file_digest, mode, strategy)
            elif strategy == ProcessingStrategy.HEADER_WITH_STATS:
                self._process_as_data(file_digest, mode, strategy)
            else:
                self._process_as_text(file_digest, mode, ProcessingStrategy.SUMMARY_ONLY)
            
        except Exception as e:
            print(f"Warning: Error processing file {filepath}: {e}", file=sys.stderr)
            file_digest.metadata.file_type = FileType.UNKNOWN
            file_digest.human_readable_summary = HumanReadableSummarizer.summarize(
                filepath,
                f"[ERROR] Processing failed: {str(e)}"
            )
    
    def _process_as_binary(self, file_digest: FileDigest):
        """处理为二进制文件"""
        file_digest.metadata.file_type = FileType.BINARY_FILES
        self.stats['binary_files'] += 1
        self._calculate_hashes(file_digest)
    
    def _read_file_content(self, filepath: Path) -> Optional[str]:
        """统一读取文件内容，处理编码问题"""
        try:
            # 首先尝试UTF-8
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # 如果UTF-8失败，尝试其他编码
            try:
                with open(filepath, 'rb') as f:
                    raw_content = f.read()
                    
                    # 尝试常见编码
                    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                    for encoding in encodings_to_try:
                        try:
                            return raw_content.decode(encoding)
                        except UnicodeDecodeError:
                            continue
                    # 所有编码都失败，使用latin-1并忽略错误
                    return raw_content.decode('latin-1', errors='ignore')
            except Exception:
                return None
    
    def _process_as_text(self, file_digest: FileDigest, mode: str, strategy: ProcessingStrategy):
        """处理为文本文件"""
        filepath = file_digest.metadata.path
        
        content = self._read_file_content(filepath)
        if content is None:
            # 解码失败，降级为二进制
            self._process_as_binary(file_digest)
            return
        
        # 根据策略处理内容
        if strategy == ProcessingStrategy.FULL_CONTENT:
            file_digest.full_content = content
            file_digest.human_readable_summary = self.human_summarizer.summarize(filepath, content)
        else:  # SUMMARY_ONLY
            summary = self.human_summarizer.summarize(filepath, content)
            file_digest.human_readable_summary = summary
            
            # 如果是全量模式但使用了摘要策略，存储摘要而非全文
            if mode == "full":
                file_digest.full_content = summary.summary or f"[Summary] {summary.line_count} lines, {summary.word_count} words"
        
        # 判断是否为关键文档
        filename = filepath.name.lower()
        critical_patterns = ['readme', 'license', 'copying', 'notice', 'changelog', 'changes', 
                            'contributing', 'install', 'authors', 'news', 'todo', 'roadmap']
        
        if any(pattern in filename for pattern in critical_patterns):
            file_digest.metadata.file_type = FileType.CRITICAL_DOCS
            self.stats['critical_docs'] += 1
        else:
            file_digest.metadata.file_type = FileType.REFERENCE_DOCS
            self.stats['reference_docs'] += 1
        self._calculate_hashes(file_digest)
    
    def _process_as_code(self, file_digest: FileDigest, mode: str, strategy: ProcessingStrategy):
        """处理为源代码文件"""
        filepath = file_digest.metadata.path
        
        content = self._read_file_content(filepath)
        if content is None:
            self._process_as_binary(file_digest)
            return
        
        # 分析源代码
        analysis = self.source_analyzer.analyze(filepath, content)
        
        # 添加复杂度分析
        if filepath.suffix.lower() == '.py':
            analysis.complexity_metrics = self.complexity_analyzer.analyze_python(content)
        else:
            analysis.complexity_metrics = self.complexity_analyzer.analyze_generic(content)
        
        file_digest.source_code_analysis = analysis
        
        if mode == "full":
            file_digest.full_content = content
        
        file_digest.metadata.file_type = FileType.SOURCE_CODE
        self.stats['source_code'] += 1
        self._calculate_hashes(file_digest)
    
    def _process_as_config(self, file_digest: FileDigest, mode: str, strategy: ProcessingStrategy):
        """处理为配置文件"""
        filepath = file_digest.metadata.path
        
        content = self._read_file_content(filepath)
        if content is None:
            self._process_as_binary(file_digest)
            return
        
        # 尝试作为源代码分析
        analysis = self.source_analyzer.analyze(filepath, content)
        file_digest.source_code_analysis = analysis
        
        # 生成结构摘要
        if strategy == ProcessingStrategy.STRUCTURE_EXTRACT:
            structure = self.source_analyzer._extract_config_structure(content, filepath.suffix.lower())
            if structure:
                summary_lines = [f"[Config Structure] Type: {filepath.suffix}"]
                for item in structure[:20]:  # 最多20项
                    summary_lines.append(f"  - {item.get('name', 'unknown')}")
                if len(structure) > 20:
                    summary_lines.append(f"  ... and {len(structure) - 20} more")
                
                file_digest.human_readable_summary = HumanReadableSummarizer.summarize(
                    filepath,
                    '\n'.join(summary_lines)
                )
        
        if mode == "full":
            file_digest.full_content = content
        
        file_digest.metadata.file_type = FileType.TEXT_DATA
        self.stats['text_data'] += 1
        self._calculate_hashes(file_digest)
    
    def _process_as_data(self, file_digest: FileDigest, mode: str, strategy: ProcessingStrategy):
        """处理为数据文件"""
        filepath = file_digest.metadata.path
        
        content = self._read_file_content(filepath)
        if content is None:
            self._process_as_binary(file_digest)
            return
        
        # 使用智能截断处理器
        if self.text_processor.is_structured_data_file(filepath):
            human_content = self.text_processor.extract_human_relevant_content(content, filepath)
            
            if mode == "full":
                file_digest.full_content = human_content
            else:
                file_digest.human_readable_summary = HumanReadableSummarizer.summarize(
                    filepath,
                    f"[Structured Data] Type: {filepath.suffix}",
                    first_lines=human_content.split('\n')[:10]
                )
        else:
            # 普通文本处理
            summary = self.human_summarizer.summarize(filepath, content)
            file_digest.human_readable_summary = summary
            
            if mode == "full":
                file_digest.full_content = content
        
        file_digest.metadata.file_type = FileType.TEXT_DATA
        self.stats['text_data'] += 1
        self._calculate_hashes(file_digest)
    
    def _calculate_hashes(self, file_digest: FileDigest):
        """计算文件的哈希值（流式处理，内存高效）"""
        filepath = file_digest.metadata.path
        
        try:
            md5_hash = hashlib.md5()
            sha256_hash = hashlib.sha256()
            
            with open(filepath, 'rb') as f:
                # 使用64KB缓冲区流式读取
                for chunk in iter(lambda: f.read(65536), b''):
                    md5_hash.update(chunk)
                    sha256_hash.update(chunk)
            
            file_digest.metadata.md5_hash = md5_hash.hexdigest()
            file_digest.metadata.sha256_hash = sha256_hash.hexdigest()
            
        except (OSError, IOError) as e:
            # 权限问题、文件不存在等
            print(f"Warning: Could not read file for hash calculation: {filepath} - {e}", file=sys.stderr)
            file_digest.metadata.md5_hash = "read_error"
            file_digest.metadata.sha256_hash = "read_error"
        except Exception as e:
            # 其他意外错误
            print(f"Warning: Hash calculation failed for {filepath}: {e}", file=sys.stderr)
            file_digest.metadata.md5_hash = "hash_error"
            file_digest.metadata.sha256_hash = "hash_error"
    
    def _generate_sort_output(self) -> Dict:
        """生成分类排序输出（增强版，保留文件元数据用于 ls -l 格式）"""
        all_files = self._collect_all_files_flat()
        
        # 按类型分组，同时保留完整元数据
        by_type = {
            FileType.CRITICAL_DOCS.value: [],
            FileType.REFERENCE_DOCS.value: [],
            FileType.SOURCE_CODE.value: [],
            FileType.TEXT_DATA.value: [],
            FileType.BINARY_FILES.value: [],
            FileType.UNKNOWN.value: []
        }
        
        # 按大小分组
        large_files = []      # > 1MB
        medium_files = []     # 100KB - 1MB
        small_files = []      # < 100KB
        
        for f in all_files:
            file_type = f.metadata.file_type.value
            file_info = {
                'path': str(f.metadata.path.relative_to(self.root)),
                'size': f.metadata.size,
                'size_formatted': self._format_bytes(f.metadata.size),
                'modified': f.metadata.modified_time.isoformat(),
                'type': file_type,
                'is_binary': file_type == FileType.BINARY_FILES.value
            }
            
            if file_type in by_type:
                by_type[file_type].append(file_info)
            else:
                by_type[FileType.UNKNOWN.value].append(file_info)
            
            # 按大小分组
            size = f.metadata.size
            if size > 1024 * 1024:
                large_files.append(file_info)
            elif size > 100 * 1024:
                medium_files.append(file_info)
            else:
                small_files.append(file_info)
        
        # 构建报告
        sort_report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "root_directory": str(self.root),
                "output_mode": "sort",
                "statistics": self.stats,
                "context_usage": self.context_manager.get_summary()
            },
            "classification": {},
            "by_size": {
                "large_files": large_files,
                "medium_files": medium_files,
                "small_files": small_files
            },
            "file_listings": {
                k: sorted(v, key=lambda x: x['path']) 
                for k, v in by_type.items() if v
            }
        }
        
        # 为每种类型生成详细信息
        for type_name, files in by_type.items():
            if not files:
                continue
                
            # 按扩展名分组
            by_ext = {}
            for f in files:
                path = f['path']
                ext = Path(path).suffix.lower() or "(no extension)"
                if ext not in by_ext:
                    by_ext[ext] = []
                by_ext[ext].append(path)
            
            # 计算总大小
            total_size = sum(f['size'] for f in files)
            
            sort_report["classification"][type_name] = {
                "count": len(files),
                "total_size_bytes": total_size,
                "total_size_formatted": self._format_bytes(total_size),
                "extensions": {
                    ext: {
                        "count": len(paths),
                        "files": sorted(paths)[:10],
                        "truncated": len(paths) > 10,
                        "total_count": len(paths)
                    }
                    for ext, paths in sorted(by_ext.items(), key=lambda x: len(x[1]), reverse=True)
                }
            }
        
        # 添加建议
        recommendations = []
        if large_files:
            recommendations.append(
                f"Found {len(large_files)} large files (>1MB). "
                f"In 'full' mode, use --max-content-size to limit full content output."
            )

        if by_type.get(FileType.UNKNOWN.value, []):
            count = len(by_type[FileType.UNKNOWN.value])
            if count > 5:
                recommendations.append(
                    f"Found {count} unknown type files. Consider reviewing or adding to ignore patterns."
                )
        
        sort_report["recommendations"] = recommendations
        
        return sort_report

    def _format_bytes(self, size_bytes: int) -> str:
        """格式化字节大小为人类可读"""
        return FormatConverter._format_size(size_bytes)
    
    def _collect_all_files_flat(self) -> List[FileDigest]:
        """扁平化收集所有文件"""
        all_files = []
        
        def collect(node: DirectoryStructure):
            all_files.extend(node.files)
            for subdir in node.subdirectories.values():
                collect(subdir)
        
        if self.structure:
            collect(self.structure)
        return all_files

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
                "context_config": {
                    "max_tokens": self.context_size,
                    "rules_file": str(self.rules_file) if self.rules_file else None,
                },
                "statistics": self.stats,
                "context_usage": self.context_manager.get_summary(),
            },
            "structure": self.structure.to_dict(mode)
        }
        
        # 添加上下文分配信息
        if self.context_manager.file_records:
            output["context_allocation"] = {
                "file_records": self.context_manager.file_records,
                "strategy_distribution": self._analyze_strategy_distribution(),
            }
        
        return output

    def _analyze_strategy_distribution(self) -> Dict[str, Any]:
        """分析策略使用分布"""
        distribution = {}
        for record in self.context_manager.file_records:
            strategy = record.get("strategy", "unknown")
            distribution[strategy] = distribution.get(strategy, 0) + 1
        
        return distribution
    
    def save_output(self, output: Dict, format: str = "json", 
                   output_path: Optional[Path] = None, mode: str = None) -> Path:
        """保存输出到文件"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = format.lower()
            if ext == "markdown":
                ext = "md"
            output_path = self.root / f"directory_digest_{timestamp}.{ext}"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = self.format_converter.convert(output, format, mode)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"摘要已保存到: {output_path}")
        return output_path
    
    def save_output_with_mode(self, output: Dict, format: str = "json", 
                             output_path: Optional[Path] = None, mode: str = None) -> Path:
        """兼容方法：保存输出到文件（带模式参数）"""
        return self.save_output(output, format, output_path, mode)

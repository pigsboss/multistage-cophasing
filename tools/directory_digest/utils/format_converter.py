# -*- coding: utf-8 -*-
"""
format_converter.py - 格式转换器
"""

import json
import re
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class FormatConverter:
    """格式转换器，支持多种输出格式"""
    
    @staticmethod
    def convert(digest_data: Dict, format: str, mode: str = None) -> str:
        """转换为指定格式"""
        # 如果是 sort 模式，强制使用专门的 sort 格式
        if mode == "sort" or digest_data.get('metadata', {}).get('output_mode') == "sort":
            return FormatConverter._to_sort_format(digest_data)
        
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
        metadata = digest_data.get('metadata', {})
        structure = digest_data.get('structure', {})
        
        # 构建Markdown内容
        lines = []
        
        # 1. 标题
        lines.append(f"# 目录摘要报告")
        lines.append("")
        
        # 2. 元数据
        lines.append("## 元数据")
        lines.append("")
        lines.append(f"- **生成时间**: {metadata.get('generated_at', '未知')}")
        lines.append(f"- **根目录**: `{metadata.get('root_directory', '未知')}`")
        lines.append(f"- **输出模式**: {metadata.get('output_mode', '未知')}")
        lines.append("")
        
        # 3. 统计信息
        stats = metadata.get('statistics', {})
        lines.append("## 统计信息")
        lines.append("")
        lines.append(f"- **总文件数**: {stats.get('total_files', 0)}")
        lines.append(f"- **人类可读文件**: {stats.get('human_readable', 0)}")
        lines.append(f"- **源代码文件**: {stats.get('source_code', 0)}")
        lines.append(f"- **二进制文件**: {stats.get('binary', 0)}")
        lines.append(f"- **总大小**: {FormatConverter._format_size(stats.get('total_size', 0))}")
        lines.append(f"- **处理时间**: {stats.get('processing_time', 0):.2f}秒")
        lines.append("")
        
        # 4. 目录结构
        lines.append("## 目录结构")
        lines.append("")
        
        # 递归生成目录树
        def generate_tree(node: Dict, level: int = 0, prefix: str = "") -> List[str]:
            """递归生成目录树"""
            tree_lines = []
            indent = "  " * level
            
            # 当前目录路径（相对根目录）
            rel_path = node.get('path', '')
            root_path = metadata.get('root_directory', '')
            if rel_path.startswith(root_path):
                rel_path = rel_path[len(root_path):].lstrip('/\\')
            
            dir_name = Path(node.get('path', '')).name
            if level == 0:
                dir_name = rel_path or "."
            
            # 目录标题
            if level == 0:
                tree_lines.append(f"### {dir_name}")
                tree_lines.append("")
            else:
                tree_lines.append(f"{prefix}**{dir_name}**/")
            
            # 当前目录下的文件
            files = node.get('files', [])
            for i, file_data in enumerate(files):
                file_meta = file_data.get('metadata', {})
                file_path = Path(file_meta.get('path', ''))
                file_name = file_path.name
                
                # 文件图标和类型标识
                file_type = file_meta.get('file_type', 'unknown')
                type_icon = FormatConverter._get_file_type_icon(file_type)
                
                # 文件大小
                file_size = FormatConverter._format_size(file_meta.get('size', 0))
                
                # 是否是最后一个文件/目录
                is_last_file = i == len(files) - 1 and not node.get('subdirectories')
                
                # 前缀和连接符
                if level > 0:
                    if is_last_file and not node.get('subdirectories'):
                        file_prefix = prefix + "└── "
                    else:
                        file_prefix = prefix + "├── "
                else:
                    file_prefix = "- "
                
                tree_lines.append(f"{file_prefix}{type_icon} `{file_name}` ({file_size})")
                
                # 文件摘要信息
                if file_data.get('summary') or file_data.get('source_analysis'):
                    summary_info = FormatConverter._get_file_summary_markdown(file_data)
                    if summary_info:
                        tree_lines.append(f"{prefix}    {summary_info}")
            
            # 子目录
            subdirs = node.get('subdirectories', {})
            subdir_names = sorted(subdirs.keys())
            
            for j, subdir_name in enumerate(subdir_names):
                subdir_node = subdirs[subdir_name]
                is_last_subdir = j == len(subdir_names) - 1
                
                # 子目录前缀
                if level > 0:
                    if is_last_subdir:
                        subdir_prefix = prefix + "└── "
                        next_prefix = prefix + "    "
                    else:
                        subdir_prefix = prefix + "├── "
                        next_prefix = prefix + "│   "
                else:
                    subdir_prefix = "- "
                    next_prefix = "  "
                
                # 递归处理子目录
                subdir_lines = generate_tree(subdir_node, level + 1, next_prefix)
                
                # 替换第一行的前缀
                if subdir_lines and level == 0:
                    # 根目录下的子目录，使用标题格式
                    tree_lines.append(f"\n### {subdir_name}")
                    tree_lines.extend(subdir_lines[1:])  # 跳过第一行（已经在上面处理了）
                elif subdir_lines:
                    # 非根目录的子目录
                    first_line = subdir_lines[0]
                    if "**" in first_line:
                        # 这是一个目录标题行，替换前缀
                        tree_lines.append(f"{subdir_prefix}{first_line}")
                    else:
                        tree_lines.append(f"{subdir_prefix}{first_line}")
                    tree_lines.extend(subdir_lines[1:])
            
            return tree_lines
        
        # 生成目录树
        tree_lines = generate_tree(structure, 0, "")
        lines.extend(tree_lines)
        lines.append("")
        
        # 5. 文件详情（如果有的话）
        all_files = FormatConverter._collect_all_files(structure)
        if all_files:
            lines.append("## 文件详情")
            lines.append("")
            
            for i, file_data in enumerate(all_files[:50]):  # 限制显示50个文件详情
                file_meta = file_data.get('metadata', {})
                file_path = Path(file_meta.get('path', ''))
                
                # 获取相对于根目录的路径
                root_path = metadata.get('root_directory', '')
                full_path = str(file_path)
                if full_path.startswith(root_path):
                    rel_path = full_path[len(root_path):].lstrip('/\\')
                else:
                    rel_path = full_path
                
                lines.append(f"### {i+1}. `{rel_path}`")
                lines.append("")
                
                # 基本信息
                lines.append("#### 基本信息")
                lines.append("")
                lines.append(f"- **类型**: {file_meta.get('file_type', 'unknown')}")
                lines.append(f"- **大小**: {FormatConverter._format_size(file_meta.get('size', 0))}")
                lines.append(f"- **修改时间**: {file_meta.get('modified_time', '未知')}")
                lines.append(f"- **MD5**: `{file_meta.get('md5_hash', '无')}`")
                lines.append("")
                
                # 摘要信息
                summary = file_data.get('summary')
                if summary:
                    lines.append("#### 摘要")
                    lines.append("")
                    lines.append(f"```text")
                    lines.append(f"{summary.get('summary', '无摘要')[:500]}")  # 限制长度
                    lines.append(f"```")
                    lines.append("")
                
                # 源代码分析
                source_analysis = file_data.get('source_analysis')
                if source_analysis:
                    lines.append("#### 源代码分析")
                    lines.append("")
                    lines.append(f"- **语言**: {source_analysis.get('language', 'unknown')}")
                    lines.append(f"- **总行数**: {source_analysis.get('total_lines', 0)}")
                    lines.append(f"- **代码行**: {source_analysis.get('code_lines', 0)}")
                    lines.append(f"- **注释行**: {source_analysis.get('comment_lines', 0)}")
                    lines.append(f"- **空白行**: {source_analysis.get('blank_lines', 0)}")
                    
                    imports = source_analysis.get('imports', [])
                    if imports:
                        lines.append(f"- **导入项**: {', '.join(imports[:10])}")
                        if len(imports) > 10:
                            lines.append(f"  （还有 {len(imports) - 10} 个）")
                    
                    functions = source_analysis.get('functions', [])
                    if functions:
                        lines.append(f"- **函数**: {len(functions)}个")
                        for func in functions[:5]:
                            func_name = func.get('name', '未知')
                            func_line = func.get('line', '?')
                            lines.append(f"  - `{func_name}` (第{func_line}行)")
                        if len(functions) > 5:
                            lines.append(f"  （还有 {len(functions) - 5} 个函数）")
                    
                    classes = source_analysis.get('classes', [])
                    if classes:
                        lines.append(f"- **类**: {len(classes)}个")
                        for cls in classes[:5]:
                            cls_name = cls.get('name', '未知')
                            cls_line = cls.get('line', '?')
                            lines.append(f"  - `{cls_name}` (第{cls_line}行)")
                        if len(classes) > 5:
                            lines.append(f"  （还有 {len(classes) - 5} 个类）")
                    lines.append("")
                
                # 完整内容（如果是全量模式且内容较小）
                full_content = file_data.get('full_content')
                if full_content and len(full_content) < 5000:  # 只显示小于5000字符的内容
                    lines.append("#### 内容预览")
                    lines.append("")
                    lines.append("```")
                    # 显示前1000个字符
                    preview = full_content[:1000]
                    lines.append(preview)
                    if len(full_content) > 1000:
                        lines.append(f"...\n（完整内容共 {len(full_content)} 字符）")
                    lines.append("```")
                    lines.append("")
                
                lines.append("---")
                lines.append("")
        
        # 6. 尾部信息
        lines.append("## 报告信息")
        lines.append("")
        lines.append(f"本报告由 **Directory Digest Tool** 生成。")
        lines.append(f"生成配置：模式=`{metadata.get('output_mode', '未知')}`")
        
        # 添加时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"生成完成时间：{timestamp}")
        
        return '\n'.join(lines)
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """格式化文件大小"""
        if size_bytes == 0:
            return "0 B"
        
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {units[i]}"
    
    @staticmethod
    def _get_file_type_icon(file_type: str) -> str:
        """获取文件类型图标"""
        icons = {
            'critical_docs': '🔑',
            'reference_docs': '📚',
            'source_code': '💻',
            'text_data': '📄',
            'binary_files': '📦',
            'unknown': '❓',
        }
        return icons.get(file_type, '📄')
    
    @staticmethod
    def _get_file_summary_markdown(file_data: Dict) -> str:
        """获取文件的Markdown摘要"""
        summary = file_data.get('summary')
        source_analysis = file_data.get('source_analysis')
        
        if source_analysis:
            language = source_analysis.get('language', '')
            total_lines = source_analysis.get('total_lines', 0)
            functions = len(source_analysis.get('functions', []))
            classes = len(source_analysis.get('classes', []))
            
            info_parts = []
            if language:
                info_parts.append(f"语言: {language}")
            if total_lines:
                info_parts.append(f"行数: {total_lines}")
            if functions:
                info_parts.append(f"函数: {functions}")
            if classes:
                info_parts.append(f"类: {classes}")
            
            if info_parts:
                return f"*({', '.join(info_parts)})*"
        
        elif summary:
            line_count = summary.get('line_count', 0)
            word_count = summary.get('word_count', 0)
            info_parts = []
            
            if line_count:
                info_parts.append(f"行数: {line_count}")
            if word_count:
                info_parts.append(f"字数: {word_count}")
            
            if info_parts:
                return f"*({', '.join(info_parts)})*"
        
        return ""
    
    @staticmethod
    def _collect_all_files(node: Dict) -> List[Dict]:
        """递归收集所有文件"""
        files = []
        
        # 添加当前节点的文件
        files.extend(node.get('files', []))
        
        # 递归处理子目录
        for subdir in node.get('subdirectories', {}).values():
            files.extend(FormatConverter._collect_all_files(subdir))
        
        return files
    
    @staticmethod
    def _to_yaml(digest_data: Dict) -> str:
        """转换为YAML格式"""
        try:
            return yaml.dump(digest_data, allow_unicode=True, default_flow_style=False)
        except Exception:
            # 如果YAML库不可用，返回JSON格式
            return json.dumps(digest_data, indent=2, ensure_ascii=False)
    
    @staticmethod
    def _to_html(digest_data: Dict) -> str:
        """转换为HTML格式"""
        # 简单实现：将Markdown转换为HTML的基本结构
        md_content = FormatConverter._to_markdown(digest_data)
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>目录摘要报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #333; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
        pre {{ background-color: #f8f8f8; padding: 10px; border-radius: 5px; overflow: auto; }}
    </style>
</head>
<body>
{FormatConverter._markdown_to_html(md_content)}
</body>
</html>"""
        return html_content
    
    @staticmethod
    def _markdown_to_html(markdown_text: str) -> str:
        """将Markdown转换为HTML（简单实现）"""
        html = markdown_text
        
        # 简单的Markdown到HTML转换
        html = html.replace('# ', '<h1>').replace('\n#', '</h1>\n<h1>')
        html = html.replace('## ', '<h2>').replace('\n##', '</h2>\n<h2>')
        html = html.replace('### ', '<h3>').replace('\n###', '</h3>\n<h3>')
        
        html = html.replace('**', '<strong>').replace('**', '</strong>')
        html = html.replace('`', '<code>').replace('`', '</code>')
        html = html.replace('- ', '<li>').replace('\n-', '</li>\n<li>')
        
        # 处理代码块
        html = re.sub(r'```(\w+)?\n(.*?)\n```', r'<pre><code>\2</code></pre>', html, flags=re.DOTALL)
        
        return html
    
    @staticmethod
    def _to_toml(digest_data: Dict) -> str:
        """转换为TOML格式"""
        # 简单实现：返回JSON，因为TOML库可能需要额外安装
        return f"# TOML格式暂未完全实现，以下是JSON表示\n{json.dumps(digest_data, indent=2, ensure_ascii=False)}"
    
    @staticmethod
    def _to_sort_format(digest_data: Dict) -> str:
        """类 ls -l 格式输出"""
        lines = []
        root_dir = digest_data.get('metadata', {}).get('root_directory', '.')
        
        lines.append(f"Directory Digest: {root_dir}")
        lines.append(f"Generated: {digest_data.get('metadata', {}).get('generated_at', 'unknown')}")
        lines.append("")
        
        # 类型映射
        type_names = {
            'critical_docs': ('Critical Docs', 'C'),
            'reference_docs': ('Reference Docs', 'R'),
            'source_code': ('Source Code', 'S'),
            'text_data': ('Text Data', 'T'),
            'binary_files': ('Binary Files', 'B'),
            'unknown': ('Unknown', '?')
        }
        
        listings = digest_data.get('file_listings', {})
        
        for type_key, (type_name, type_char) in type_names.items():
            if type_key not in listings or not listings[type_key]:
                continue
            
            files = listings[type_key]
            total_size = sum(f.get('size', 0) for f in files)
            
            lines.append(f"{type_name} ({len(files)} files, {FormatConverter._format_size(total_size)})")
            lines.append("-" * 80)
            
            # 类 ls -l 格式：类型-权限 大小 日期 时间 路径
            for f in files[:100]:  # 限制显示数量
                path = f.get('path', 'unknown')
                size = f.get('size_formatted', '0 B')
                modified = f.get('modified', 'unknown')
                
                # 简化格式：- 大小 日期 路径
                if modified != 'unknown':
                    try:
                        dt = datetime.fromisoformat(modified)
                        date_str = dt.strftime("%b %d %H:%M")
                    except:
                        date_str = modified[:16]
                else:
                    date_str = "unknown"
                
                # 格式：类型 大小 日期 路径
                lines.append(f"{type_char}  {size:>10}  {date_str:>12}  {path}")
            
            if len(files) > 100:
                lines.append(f"... ({len(files) - 100} more files)")
            
            lines.append("")
        
        # 统计摘要
        stats = digest_data.get('metadata', {}).get('statistics', {})
        lines.append("Summary:")
        lines.append(f"  Total: {stats.get('total_files', 0)} files, "
                    f"{FormatConverter._format_size(stats.get('total_size', 0))}")
        lines.append(f"  Critical Docs: {stats.get('critical_docs', 0)}")
        lines.append(f"  Reference Docs: {stats.get('reference_docs', 0)}")
        lines.append(f"  Source Code: {stats.get('source_code', 0)}")
        lines.append(f"  Text Data: {stats.get('text_data', 0)}")
        lines.append(f"  Binary Files: {stats.get('binary_files', 0)}")
        lines.append(f"  Skipped (>limit): {stats.get('skipped_large_files', 0)}")
        
        return '\n'.join(lines)
    
    @staticmethod
    def _to_plaintext(digest_data: Dict) -> str:
        """转换为纯文本格式"""
        # 使用Markdown但去除格式
        md_content = FormatConverter._to_markdown(digest_data)
        
        # 去除Markdown格式
        plaintext = md_content
        
        # 去除标题标记
        plaintext = re.sub(r'#+\s+', '', plaintext)
        # 去除粗体标记
        plaintext = plaintext.replace('**', '')
        # 去除代码标记
        plaintext = plaintext.replace('`', '')
        # 去除列表标记
        plaintext = re.sub(r'^[│├└──\s]+', '', plaintext, flags=re.MULTILINE)
        
        return plaintext

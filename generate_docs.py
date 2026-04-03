#!/usr/bin/env python3
"""
MCPC 架构文档自动生成器
扫描代码目录，生成：
- 目录结构（Markdown 树）
- PlantUML 组件图（模块依赖）
- PlantUML 活动图（仿真流程）
- PlantUML 类图（继承与主要类）

用法: python generate_docs.py
"""

import os
import ast
import re
from pathlib import Path
from collections import defaultdict, Counter

# ========== 配置 ==========
SOURCE_DIR = "mission_sim"
OUTPUT_DIR = "docs"
DIAGRAM_DIR = os.path.join(OUTPUT_DIR, "diagrams")
os.makedirs(DIAGRAM_DIR, exist_ok=True)

# 忽略的目录和文件
IGNORE_DIRS = {"__pycache__", "tests", "analysis", "config", "docs", "scripts"}
IGNORE_FILES = {"__init__.py"}


# ========== 辅助函数 ==========
def should_skip(path):
    """判断是否跳过文件或目录"""
    parts = Path(path).parts
    for p in parts:
        if p in IGNORE_DIRS:
            return True
    return False


def get_tree_structure(start_path, indent=""):
    """递归生成目录树 Markdown"""
    lines = []
    items = sorted(os.listdir(start_path))
    items = [i for i in items if not i.startswith('.') and i not in IGNORE_DIRS]
    for i, item in enumerate(items):
        path = os.path.join(start_path, item)
        is_last = (i == len(items) - 1)
        prefix = "└── " if is_last else "├── "
        if os.path.isdir(path):
            lines.append(f"{indent}{prefix}📁 {item}/")
            extension = "    " if is_last else "│   "
            lines.extend(get_tree_structure(path, indent + extension))
        else:
            if item.endswith('.py') and item not in IGNORE_FILES:
                lines.append(f"{indent}{prefix}📄 {item}")
    return lines


# ========== 提取导入关系 ==========
def extract_imports(file_path):
    """返回该文件导入的模块（顶级）"""
    imports = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=file_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except Exception:
        pass
    return imports


def build_module_graph():
    """构建模块依赖图：模块名 -> 依赖的模块名集合"""
    graph = defaultdict(set)
    for root, dirs, files in os.walk(SOURCE_DIR):
        if should_skip(root):
            continue
        for file in files:
            if file.endswith('.py') and file not in IGNORE_FILES:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, SOURCE_DIR).replace(os.sep, '.')
                module_name = rel_path[:-3]  # remove .py
                imports = extract_imports(full_path)
                for imp in imports:
                    if imp.startswith('mission_sim'):
                        graph[module_name].add(imp)
    return graph


def generate_component_puml(graph):
    """生成 PlantUML 组件图"""
    lines = [
        "@startuml",
        "!theme plain",
        "skinparam componentStyle rectangle",
        "title MCPC 框架组件依赖图（高层次）",
        ""
    ]
    # 定义组件
    components = set()
    for src, deps in graph.items():
        components.add(src)
        components.update(deps)
    for comp in sorted(components):
        lines.append(f'component "{comp}" as {comp.replace(".", "_")}')
    lines.append("")
    # 依赖关系
    for src, deps in graph.items():
        src_id = src.replace(".", "_")
        for dep in deps:
            dep_id = dep.replace(".", "_")
            lines.append(f'{src_id} --> {dep_id} : uses')
    lines.append("@enduml")
    return "\n".join(lines)


# ========== 提取类与继承 ==========
def extract_classes(file_path):
    """返回文件中的类名及基类列表"""
    classes = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=file_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        bases.append(base.attr)
                classes[node.name] = bases
    except Exception:
        pass
    return classes


def build_class_hierarchy():
    """全局类名 -> 基类列表"""
    hierarchy = {}
    for root, dirs, files in os.walk(SOURCE_DIR):
        if should_skip(root):
            continue
        for file in files:
            if file.endswith('.py') and file not in IGNORE_FILES:
                full_path = os.path.join(root, file)
                classes = extract_classes(full_path)
                hierarchy.update(classes)
    return hierarchy


def generate_class_puml(hierarchy):
    """生成 PlantUML 类图，简化展示主要类及继承关系"""
    lines = [
        "@startuml",
        "!theme plain",
        "skinparam classAttributeIconSize 0",
        "title MCPC 框架核心类图（简化）",
        ""
    ]
    # 只显示重要的类（如 Simulation, Controller, Spacecraft 等）
    important_keywords = ['Simulation', 'Controller', 'Spacecraft', 'GNC', 'Formation', 'Propagator']
    important_classes = {name for name in hierarchy.keys() if any(k in name for k in important_keywords)}
    # 加上它们的基类
    for name, bases in hierarchy.items():
        for base in bases:
            if base in hierarchy or any(k in base for k in important_keywords):
                important_classes.add(name)
                important_classes.add(base)

    # 输出类
    for cls in sorted(important_classes):
        lines.append(f'class {cls} {{')
        lines.append('}')
    lines.append("")
    # 继承关系
    for cls, bases in hierarchy.items():
        if cls not in important_classes:
            continue
        for base in bases:
            if base in important_classes:
                lines.append(f'{cls} --|> {base}')
    lines.append("@enduml")
    return "\n".join(lines)


# ========== 提取活动图关键步骤 ==========
def generate_activity_puml():
    """基于代码中的关键类和方法，生成仿真主循环活动图"""
    # 从 BaseSimulation 和 FormationSimulation 中提取关键方法名
    # 这里手动构建一个通用活动图，因为自动分析控制流较复杂
    lines = [
        "@startuml",
        "!theme plain",
        "skinparam activityDiamondBackgroundColor #FFF2CC",
        "title MCPC L2 编队仿真主循环活动图",
        "",
        "|Simulation|",
        "start",
        ":Tick N 开始;",
        "",
        "|Physics|",
        ":1. 并行计算所有航天器的环境加速度;",
        ":2. 状态积分（RK4/Euler）;",
        "",
        "|Physics|",
        ":3. 生成 ISL 物理测量（天线噪声、衰减）;",
        "",
        "|Cyber|",
        ":4. 封装为网络帧（添加延迟、丢包）;",
        ":5. 各从星 GNC 接收帧并预测状态;",
        ":6. 计算控制力（LQR/MPC）;",
        "",
        "|Physics|",
        ":7. 推力器执行（死区、MIB、冲量平滑）;",
        ":8. 更新质量、累计 ΔV;",
        "",
        "|Simulation|",
        ":9. 记录数据（HDF5）;",
        ":10. 时间步进;",
        "",
        "if (仿真结束?) then (否)",
        "  -> 回到 Tick N+1;",
        "else (是)",
        "  stop;",
        "endif",
        "@enduml"
    ]
    return "\n".join(lines)


# ========== 主函数 ==========
def main():
    print("🔍 扫描代码目录...")
    # 生成目录树
    tree_lines = get_tree_structure(SOURCE_DIR)
    with open(os.path.join(OUTPUT_DIR, "structure.md"), "w", encoding="utf-8") as f:
        f.write("# MCPC 代码目录结构\n\n```\n")
        f.write("\n".join(tree_lines))
        f.write("\n```\n")
    print("✅ 目录树已保存至 docs/structure.md")

    # 生成组件图
    print("📊 分析模块依赖...")
    graph = build_module_graph()
    component_puml = generate_component_puml(graph)
    with open(os.path.join(DIAGRAM_DIR, "components.puml"), "w", encoding="utf-8") as f:
        f.write(component_puml)
    print("✅ 组件图已保存至 docs/diagrams/components.puml")

    # 生成类图
    print("🏛️ 提取类继承关系...")
    hierarchy = build_class_hierarchy()
    class_puml = generate_class_puml(hierarchy)
    with open(os.path.join(DIAGRAM_DIR, "class.puml"), "w", encoding="utf-8") as f:
        f.write(class_puml)
    print("✅ 类图已保存至 docs/diagrams/class.puml")

    # 生成活动图（基于模板）
    activity_puml = generate_activity_puml()
    with open(os.path.join(DIAGRAM_DIR, "activity.puml"), "w", encoding="utf-8") as f:
        f.write(activity_puml)
    print("✅ 活动图已保存至 docs/diagrams/activity.puml")

    print("\n🎉 文档生成完成！")
    print("提示：使用 PlantUML 渲染 .puml 文件为 PNG/SVG。")

if __name__ == "__main__":
    main()
# MCPC 框架文档化指南

## 1. 概述

### 1.1 目标

本指南旨在建立 MCPC 框架的统一文档体系，实现以下目标：

- **API 文档自动化**：通过 Sphinx 从源代码 docstring 自动生成模块级 API 参考。
- **设计文档集中化**：将架构设计、计划、复盘等手动编写的文档集中管理，便于维护和追溯。
- **图表资产可追溯**：PlantUML 源文件作为设计资产与文档紧密集成，支持 CI 自动渲染。
- **扩展指南清晰化**：为用户和贡献者提供详细的场景添加、力模型扩展等教程。
- **构建流程自动化**：支持在 CI 中自动生成文档并部署到 GitHub Pages。

### 1.2 受众

- **核心开发者**：查阅 API 文档、架构设计、复盘记录。
- **贡献者**：通过扩展指南快速上手，理解设计模式。
- **用户**：通过 README 和快速入门了解框架能力，通过 API 文档深入使用。

---

## 2. 目录结构

```text
mcpc/
├── docs/
│   ├── source/                     # Sphinx 源文件目录
│   │   ├── conf.py                 # Sphinx 配置文件
│   │   ├── index.rst               # 文档根页面
│   │   ├── api/                    # API 参考（由 sphinx.ext.autodoc 生成）
│   │   └── _static/                # 自定义 CSS/JS（可选）
│   ├── build/                      # Sphinx 构建输出（不提交，已 .gitignore）
│   ├── design/                     # 总体设想、计划类文档（手动编写）
│   │   ├── architecture.md         # 全层级架构设计总则
│   │   ├── L1_architecture.md      # L1 级架构设计
│   │   ├── roadmap.md              # 项目路线图
│   │   └── L2_design_notes.md      # L2 级预研笔记（未来）
│   ├── reviews/                    # 复盘、回顾类文档
│   │   ├── L1_completion_review.md # L1 级开发总结
│   │   ├── test_review.md          # 测试复盘与故障分析
│   │   └── performance_report.md   # 性能优化记录（可选）
│   ├── guides/                     # 扩展指南（手动编写）
│   │   ├── add_new_scenario.md     # 添加新场景指南
│   │   ├── add_new_force_model.md  # 添加新力模型指南
│   │   └── visualize_data.md       # 数据可视化指南（可选）
│   ├── diagrams/                   # PlantUML 源文件
│   │   ├── L1_activities.puml
│   │   ├── L1_architecture.puml
│   │   ├── L1_classes.puml
│   │   ├── all_levels_architecture.puml
│   │   └── (未来新增的 UML)
│   ├── README.md                   # 文档目录说明，引导用户
│   └── requirements.txt            # Sphinx 及扩展依赖
├── README.md                       # 项目根 README
├── README.en.md                    # 英文 README
└── ... (其他项目文件)
```

---

## 3. 文档编写规范

### 3.1 语言与格式

- **语言**：推荐使用英文撰写 API 文档和扩展指南，以保持国际化；设计文档和复盘可使用中文或双语。
- **格式**：
  - Sphinx 源文件使用 reStructuredText（`.rst`）。
  - 手动编写的文档（`design/`、`reviews/`、`guides/`）使用 Markdown（`.md`），便于快速编辑和版本对比。
  - PlantUML 文件使用 `.puml` 后缀。

### 3.2 API 文档规范

- **docstring 格式**：使用 Google 风格或 NumPy 风格，Sphinx 将通过 `sphinx.ext.napoleon` 解析。
  ```python
  def compute_accel(self, state: np.ndarray, epoch: float) -> np.ndarray:
      """
      Compute atmospheric drag acceleration.
  
      Args:
          state: Spacecraft state vector [x, y, z, vx, vy, vz] (ECI)
          epoch: Current simulation time (s)
  
      Returns:
          Acceleration vector [ax, ay, az] (m/s²)
      """
  ```
- **类型提示**：使用 Python 类型注解，Sphinx 可自动提取。

### 3.3 设计文档规范

- **元数据**：每个文档开头注明作者、日期、版本。
- **图表引用**：通过相对路径引用 `diagrams/` 下的 PlantUML 文件，并在 Sphinx 构建时渲染。
- **版本控制**：重要设计变更需更新文档版本号，并在 `roadmap.md` 中记录。

### 3.4 扩展指南规范

- **示例代码**：提供完整的可运行代码片段。
- **步骤清晰**：使用有序列表，每个步骤附有代码或命令。
- **预期结果**：说明操作后的预期输出或行为。

---

## 4. Sphinx 配置与构建

### 4.1 初始化 Sphinx 项目

```bash
cd docs
sphinx-quickstart
```

按提示配置：
- 根目录：`source`
- 项目名称：`MCPC Framework`
- 作者：`MCPC Team`
- 版本：`2.0`
- 语言：`en`

### 4.2 安装依赖

创建 `docs/requirements.txt`：

```txt
sphinx>=7.0
sphinx-rtd-theme>=1.3
myst-parser>=2.0
sphinxcontrib-plantuml>=0.25
sphinx-autodoc-typehints>=1.24
```

安装：
```bash
pip install -r docs/requirements.txt
```

### 4.3 配置 `conf.py`

```python
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))   # 指向项目根目录

project = 'MCPC Framework'
copyright = '2025, MCPC Team'
author = 'MCPC Team'
release = '2.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser',
    'sphinx_rtd_theme',
    'sphinxcontrib.plantuml',
    'sphinx_autodoc_typehints',
]

# 主题
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# PlantUML 配置（需安装 plantuml）
plantuml = 'plantuml'                     # 或 'java -jar /path/to/plantuml.jar'
plantuml_output_format = 'png'            # 或 'svg'

# 支持 Markdown
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# autodoc 配置
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'show-inheritance': True,
}
```

### 4.4 编写根页面 `source/index.rst`

```rst
MCPC Framework Documentation
============================

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   ../../guides/add_new_scenario.md
   ../../guides/add_new_force_model.md

.. toctree::
   :maxdepth: 2
   :caption: Design Documents

   ../../design/architecture.md
   ../../design/L1_architecture.md
   ../../design/roadmap.md

.. toctree::
   :maxdepth: 2
   :caption: Reviews

   ../../reviews/L1_completion_review.md
   ../../reviews/test_review.md

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```

### 4.5 生成 API 文档

在 `source/api/` 下创建 `modules.rst`：

```rst
API Reference
=============

.. automodule:: mission_sim
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mission_sim.core.types
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mission_sim.simulation.base
   :members:
   :undoc-members:
   :show-inheritance:

... (其他模块)
```

### 4.6 在文档中引用 PlantUML

在 `.rst` 或 `.md` 文件中：

**reStructuredText**：
```rst
.. uml:: ../../diagrams/L1_architecture.puml
   :caption: L1 级系统架构图
```

**Markdown（通过 myst_parser）**：
```markdown
```{uml} ../diagrams/L1_architecture.puml
:caption: L1 级系统架构图
```
```

### 4.7 构建文档

```bash
cd docs
sphinx-build -b html source build
```

生成的 HTML 位于 `docs/build/`。

### 4.8 部署到 GitHub Pages（可选）

在 GitHub Actions 中添加工作流：

```yaml
name: Build and Deploy Docs
on:
  push:
    branches: [main]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.14'
      - name: Install dependencies
        run: |
          pip install -r docs/requirements.txt
          sudo apt-get install -y plantuml   # 安装 PlantUML
      - name: Build docs
        run: |
          cd docs
          sphinx-build -b html source build
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: docs/build
```

---

## 5. 迁移现有文档

### 5.1 移动文件

| 原位置 | 新位置 | 操作 |
|--------|--------|------|
| `REVIEW.md` | `docs/reviews/architecture_overview.md` | 重命名并移动 |
| `PLAN.md` | `docs/design/roadmap.md` | 移动 |
| `notes/L1_architecture.md` | `docs/design/L1_architecture.md` | 移动 |
| `notes/all_levels_architecture.md` | `docs/design/architecture.md` | 移动并重命名 |
| `docs/*.puml` | `docs/diagrams/` | 移动 |

### 5.2 更新链接

- 在移动后的文档中，更新对 PlantUML 文件的引用路径（如 `../diagrams/xxx.puml`）。
- 在根目录 `README.md` 和 `README.en.md` 中添加指向 `docs/` 的链接：
  ```markdown
  ## 📚 Documentation
  - [API Reference](docs/build/index.html)
  - [Design Documents](docs/design/)
  - [Extension Guides](docs/guides/)
  ```

### 5.3 创建 `docs/README.md`


````markdown
# MCPC Documentation

This directory contains the complete documentation for the MCPC framework.

- **[Design Documents](design/)** – Architecture design, roadmap, and planning notes.
- **[Reviews](reviews/)** – Development retrospectives and test analysis.
- **[Guides](guides/)** – Step-by-step tutorials for extending the framework.
- **[Diagrams](diagrams/)** – PlantUML source files for architecture diagrams.
- **[API Reference](build/index.html)** – Auto-generated API documentation (build after `sphinx-build`).

## Building the Documentation
```bash
pip install -r requirements.txt
sphinx-build -b html source build
```
Open `build/index.html` in your browser.
````

---

## 6. 维护流程

### 6.1 日常更新

- **代码变更后**：运行 `sphinx-build` 更新 API 文档，确保与代码同步。
- **设计变更时**：更新 `design/` 下对应文档，并在 `roadmap.md` 中记录变更日志。
- **重大里程碑后**：在 `reviews/` 下撰写复盘文档，总结经验教训。

### 6.2 版本发布

- 在 `docs/design/roadmap.md` 中标记版本里程碑。
- 生成文档并部署到 GitHub Pages。
- 在项目根 `README.md` 中更新文档链接。

### 6.3 CI/CD 集成

- 每次推送到 `main` 分支时自动构建文档并部署到 GitHub Pages。
- 在 Pull Request 中自动构建预览（可选）。

---

## 7. 常见问题

### Q1: 为什么将 PlantUML 文件放在 `docs/diagrams/` 而不是 `source/`？

**A**: 保持源文件与 Sphinx 源分离，便于在 CI 中独立渲染，也方便在非 Sphinx 环境中（如 IDE 插件）直接编辑预览。

### Q2: 如何确保 Markdown 中的 PlantUML 代码块在 Sphinx 中正确渲染？

**A**: 使用 `myst_parser` 扩展，并在 `conf.py` 中启用 `myst_enable_extensions = ["colon_fence"]`，然后使用 ` ```{uml} ` 语法。

### Q3: 本地没有 PlantUML 怎么办？

**A**: 安装 PlantUML（`sudo apt install plantuml`）或使用 Docker 运行。若仅需构建文档，可在 CI 中安装；本地可暂时跳过图表渲染（在 `conf.py` 中设置 `plantuml = 'plantuml'` 并确保已安装）。

### Q4: 如何组织大型文档（如超过 10 个章节）？

**A**: 使用 Sphinx 的 `toctree` 嵌套，将大文档拆分为多个子文件，并在 `index.rst` 中引用。

---

## 8. 总结

本指南为 MCPC 框架建立了统一的文档体系，明确了目录结构、编写规范、构建流程和维护机制。通过将 API 文档自动化、设计文档集中化、图表资产可追溯，确保项目文档与代码同步演进，降低维护成本，提升协作效率。

**下一步**：按照本指南迁移现有文档，配置 Sphinx，并在 CI 中集成自动构建。完成后，MCPC 框架将拥有专业、易维护的文档系统，支撑后续 L2 级开发与社区扩展。
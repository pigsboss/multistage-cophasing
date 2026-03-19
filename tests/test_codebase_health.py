# tests/test_codebase_health.py
import unittest
import os
import re

class TestNamingConventions(unittest.TestCase):
    def test_level_designator_capitalization(self):
        """扫描项目目录，确保所有的 'l1', 'l2' 等都被正确大写为 'L1', 'L2'"""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 匹配小写的 l 加数字，且前后是下划线或点 (如 _l1_ 或 l1.py)
        bad_pattern = re.compile(r'(_l[1-5]_|_l[1-5]\.|^l[1-5]_)')
        
        violations = []
        for root, _, files in os.walk(project_root):
            # 排除隐藏目录和虚拟环境
            if '.git' in root or '__pycache__' in root or 'venv' in root:
                continue
                
            for file in files:
                if bad_pattern.search(file):
                    violations.append(os.path.join(root, file))
                    
        self.assertEqual(
            len(violations), 0, 
            f"发现违反命名规范的文件 (级别 'L' 必须大写):\n" + "\n".join(violations)
        )

if __name__ == '__main__':
    unittest.main()

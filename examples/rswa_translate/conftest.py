"""pytest 配置：把本目录加入 sys.path，使 tests/ 能直接 import text_utils / rswa_qwen25。"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

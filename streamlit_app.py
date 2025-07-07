"""
吉他谱AI - Streamlit Cloud 部署入口
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 确保ffmpeg可用（如果系统有的话）
if "/opt/homebrew/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = os.environ.get("PATH", "") + ":/opt/homebrew/bin:/usr/local/bin"

# 导入并运行主应用
if __name__ == "__main__":
    from webui.app import main
    main()
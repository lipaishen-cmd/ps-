"""启动项目 TensorBoard 监控服务，自动绑定项目结果目录（支持 UTF-8 中文路径）。"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGDIR = ROOT / "result"

if not LOGDIR.is_dir():
    print(f"[错误] 未找到结果目录: {LOGDIR}")
    sys.exit(1)

print(f"[TensorBoard] 监控目录: {LOGDIR}")
print(f"[TensorBoard] 启动端口: 6006 (自动刷新周期: 5秒)")

from tensorboard import default
from tensorboard.program import TensorBoard

tb = TensorBoard(default.get_plugins())
tb.configure(argv=[
    "tensorboard",
    "--logdir", str(LOGDIR),
    "--port", "6006",
    "--reload_interval", "5",
    "--bind_all",
])
sys.exit(tb.main())

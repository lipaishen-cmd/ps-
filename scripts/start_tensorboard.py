"""启动项目 TensorBoard 监控服务，自动绑定项目结果目录（支持 UTF-8 中文路径）。

对于旧进程正在写入但尚未内置 SummaryWriter 的 external_rl/train.csv，
自动在 tensorboard_sync 子目录同步曲线，无需中断正在跑的训练。
"""

import csv
import shutil
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGDIR = ROOT / "result"


def _sync_csv_runs(logdir: Path, writers: dict) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return

    for csv_path in sorted(logdir.glob("external_rl/*/*/*/train.csv")):
        run_dir = csv_path.parent
        native_tb = run_dir / "tensorboard"
        if native_tb.is_dir() and any(native_tb.glob("*tfevents*")):
            sync_dir = run_dir / "tensorboard_sync"
            if run_dir in writers:
                writers.pop(run_dir)[0].close()
            if sync_dir.exists():
                shutil.rmtree(sync_dir, ignore_errors=True)
            continue

        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
        except OSError:
            continue
        if not rows:
            continue

        if run_dir not in writers:
            sync_dir = run_dir / "tensorboard_sync"
            shutil.rmtree(sync_dir, ignore_errors=True)
            writers[run_dir] = [SummaryWriter(log_dir=str(sync_dir)), -1]

        writer, last_ep = writers[run_dir]
        updated = False
        for row in rows:
            try:
                ep = int(row["episode"])
            except (KeyError, ValueError):
                continue
            if ep <= last_ep:
                continue
            step = ep + 1
            cmax = row.get("final_cmax")
            if cmax not in (None, "", "None", "null"):
                writer.add_scalar("Train/Makespan_Cmax", float(cmax), step)
            loss_v = row.get("loss")
            if loss_v not in (None, "", "None", "null"):
                writer.add_scalar("Train/Total_Loss", float(loss_v), step)
            gnorm = row.get("gradient_norm")
            if gnorm not in (None, "", "None", "null"):
                writer.add_scalar("Train/Grad_Norm", float(gnorm), step)
            sims = row.get("cumulative_env_simulations")
            if sims not in (None, "", "None", "null"):
                writer.add_scalar("Train/Cumulative_Env_Simulations", int(float(sims)), step)
            staleness = row.get("parameter_staleness")
            if staleness not in (None, "", "None", "null"):
                writer.add_scalar("Train/Parameter_Staleness", float(staleness), step)
            last_ep = ep
            updated = True
        if updated:
            writer.flush()
            writers[run_dir][1] = last_ep


def main() -> int:
    if not LOGDIR.is_dir():
        print(f"[错误] 未找到结果目录: {LOGDIR}")
        return 1

    writers: dict = {}
    _sync_csv_runs(LOGDIR, writers)

    stop_event = threading.Event()

    def _watcher() -> None:
        while not stop_event.wait(10.0):
            _sync_csv_runs(LOGDIR, writers)

    threading.Thread(target=_watcher, daemon=True).start()

    print(f"[TensorBoard] 监控目录: {LOGDIR}")
    print("[TensorBoard] 浏览器打开: http://localhost:6006 (自动刷新周期: 5秒)")

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
    try:
        return int(tb.main() or 0)
    finally:
        stop_event.set()
        for writer, _ in writers.values():
            writer.close()


if __name__ == "__main__":
    sys.exit(main())

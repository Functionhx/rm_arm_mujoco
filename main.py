"""入口：运行一次视觉引导取放任务（无头模式，打印结果）。

用法:
    uv run python main.py            # 运行一次任务
    uv run python main.py --viewer   # 启动交互式可视化窗口观看
"""
from __future__ import annotations

import argparse

from src.task.pick_place_task import PickPlaceTask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true", help="启动交互式可视化窗口")
    args = ap.parse_args()

    if args.viewer:
        from scripts.view_live import run_viewer
        run_viewer()
        return

    task = PickPlaceTask()
    task.env.reset()
    result = task.run()
    print("\n===== 任务结果 =====")
    for k, v in result.items():
        if k != "history":
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()

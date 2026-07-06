"""一键运行完整取放 demo（无头，打印结果与状态迁移）。"""

import sys, pathlib as _pl
sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))


from src.task.pick_place_task import PickPlaceTask


def main():
    task = PickPlaceTask()
    task.env.reset()
    result = task.run()
    print("\n===== 取放任务结果 =====")
    print("成功:", result["success"])
    print("终止状态:", result["final_state"])
    print("放置水平误差: %.1f mm" % (result["place_xy_error"] * 1000))
    if result["vision_error"] is not None:
        print("视觉位置误差: %.1f mm" % (result["vision_error"] * 1000))
    print("状态迁移:", " -> ".join(result["history"]))


if __name__ == "__main__":
    main()

"""交互式可视化：弹出实时窗口观看取放全过程。

在 macOS 上 MuJoCo 原生 viewer 需要 mjpython，且与 uv 自带的 Python 存在
libpython 加载冲突。为“开箱即用”，这里改用 OpenCV 窗口逐帧实时显示离屏渲染画面，
用普通 `uv run python scripts/view_live.py` 即可弹出窗口。

用法:
    uv run python scripts/view_live.py                 # 默认演示相机
    uv run python scripts/view_live.py --cam topdown_cam
按 q 或 Esc 关闭窗口。
"""

import sys, pathlib as _pl
sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import argparse

import cv2
import mujoco

from src.task.pick_place_task import PickPlaceTask

WIN = "RM Arm Pick-Place (MuJoCo)"


class LiveViewer:
    """作为 recorder 传入任务：每个控制周期渲染一帧并实时显示。"""

    def __init__(self, env, camera="video_cam", width=960, height=680, every=2):
        self.env = env
        self.camera = camera
        self.every = every
        self._r = mujoco.Renderer(env.model, height=height, width=width)
        self._n = 0
        self._phase = "SEARCH"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, width, height)

    def on_state(self, name):
        self._phase = name

    def tick(self):
        self._n += 1
        if self._n % self.every:
            return
        self._r.update_scene(self.env.data, camera=self.camera)
        bgr = cv2.cvtColor(self._r.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, f"phase: {self._phase}", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(WIN, bgr)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            raise KeyboardInterrupt


def run_viewer(camera="video_cam"):
    task = PickPlaceTask()
    env = task.env
    env.reset()
    viewer = LiveViewer(env, camera=camera)
    print("实时窗口已打开：观看取放过程，按 q 或 Esc 关闭。")
    try:
        result = task.run(recorder=viewer)
        print("任务完成: success=%s  放置误差=%.1f mm" %
              (result["success"], result["place_xy_error"] * 1000))
        print("窗口保持显示，按任意键关闭…")
        cv2.waitKey(0)
    except KeyboardInterrupt:
        print("已手动关闭。")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="video_cam",
                    help="相机名: video_cam / scene_cam / topdown_cam")
    args = ap.parse_args()
    run_viewer(args.cam)

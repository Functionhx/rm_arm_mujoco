"""录制报告用素材：完整取放视频、分阶段关键帧、RGB-D、视觉检测、轨迹曲线。

产物保存到 outputs/，并复制报告所需图片到 ../figures/（供 LaTeX 直接引用）。
"""

import sys, pathlib as _pl
sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))


import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.geometry.frames import FrameManager
from src.sim.camera import RGBDCamera
from src.sim.mujoco_env import MujocoEnv, load_yaml
from src.sim.recorder import Recorder
from src.task.pick_place_task import PickPlaceTask
from src.vision.color_detector import ColorDetector
from src.vision.pose_estimator import PoseEstimator
from src.utils.debug_draw import depth_to_colormap, hstack_images

REPORT_FIG = Path("../figures")   # /Users/chen/Documents/Latex/rm4/figures
STAGE_CAPS = ["stage_1", "stage_2", "stage_3", "stage_4", "stage_5", "stage_6"]


def _copy(src: str, dst_name: str):
    REPORT_FIG.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, REPORT_FIG / dst_name)


def main():
    env = MujocoEnv()
    env.reset()
    rec = Recorder(env, camera="video_cam", width=900, height=650, fps=30, stride=2)

    # 场景总览（复位态）
    Path("outputs/screenshots").mkdir(parents=True, exist_ok=True)
    imageio.imwrite("outputs/screenshots/sim_scene_overview.png", rec.snapshot("video_cam"))

    # RGB-D + 视觉检测（任务前）
    task = PickPlaceTask(env=env, recorder=rec)
    rgb, depth = task.camera.render()
    imageio.imwrite("outputs/screenshots/cam_rgb.png", rgb)
    imageio.imwrite("outputs/screenshots/cam_depth.png", depth_to_colormap(depth))
    pose = task.pose_estimator.estimate(rgb, depth)
    det = pose.detection
    annotated = ColorDetector.annotate(rgb, det)
    vis = hstack_images(annotated, det.mask)
    imageio.imwrite("outputs/screenshots/vision_detection.png", vis)

    # 运行完整任务（录制）
    result = task.run(recorder=rec)

    # 最终帧作为第 6 阶段
    final_frame = rec.snapshot("video_cam")

    # 保存视频与轨迹
    rec.save_video("outputs/videos/pick_place.mp4")
    rec.save_trajectory("outputs/trajectories/traj.npz")

    # 保存分阶段关键帧
    stage_imgs = list(rec.stage_snaps.values())[:5] + [final_frame]
    for name, img in zip(STAGE_CAPS, stage_imgs):
        imageio.imwrite(f"outputs/screenshots/{name}.png", img)

    # 轨迹曲线图
    _plot_trajectory(rec)

    # 复制报告图片
    _copy("outputs/screenshots/sim_scene_overview.png", "sim_scene_overview.png")
    _copy("outputs/screenshots/cam_rgb.png", "cam_rgb.png")
    _copy("outputs/screenshots/cam_depth.png", "cam_depth.png")
    _copy("outputs/screenshots/vision_detection.png", "vision_detection.png")
    for name in STAGE_CAPS:
        _copy(f"outputs/screenshots/{name}.png", f"{name}.png")
    _copy("outputs/screenshots/ee_trajectory.png", "ee_trajectory.png")
    _copy("outputs/screenshots/joint_curves.png", "joint_curves.png")

    print("\n===== 录制完成 =====")
    print("成功:", result["success"], " 终止状态:", result["final_state"])
    print("放置误差: %.1f mm  视觉误差: %.1f mm" %
          (result["place_xy_error"] * 1000,
           (result["vision_error"] or 0) * 1000))
    print("视频: outputs/videos/pick_place.mp4")
    print("报告图片已复制到:", REPORT_FIG.resolve())


def _plot_trajectory(rec: Recorder):
    t = np.array(rec.t_traj)
    ee = np.array(rec.ee_traj)
    q = np.array(rec.q_traj)

    # 末端三维轨迹
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], color="#4472c4", lw=2)
    ax.scatter(ee[0, 0], ee[0, 1], ee[0, 2], c="g", s=40, label="start")
    ax.scatter(ee[-1, 0], ee[-1, 1], ee[-1, 2], c="r", s=40, label="end")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title("End-effector trajectory")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig("outputs/screenshots/ee_trajectory.png", dpi=130)
    plt.close(fig)

    # 关节角度曲线
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for j in range(q.shape[1]):
        ax.plot(t, q[:, j], lw=1.4, label=f"joint{j+1}")
    ax.set_xlabel("time (s)"); ax.set_ylabel("joint angle (rad)")
    ax.set_title("Joint angle trajectories")
    ax.legend(ncol=4, fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("outputs/screenshots/joint_curves.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()

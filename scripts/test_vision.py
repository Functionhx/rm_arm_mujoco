"""测试视觉检测与位姿估计：保存标注图与掩膜，报告位置估计误差。"""

import sys, pathlib as _pl
sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))


import numpy as np
import imageio.v2 as imageio

from src.geometry.frames import FrameManager
from src.sim.camera import RGBDCamera
from src.sim.mujoco_env import MujocoEnv, load_yaml
from src.vision.color_detector import ColorDetector
from src.vision.pose_estimator import PoseEstimator
from src.utils.debug_draw import hstack_images


def main():
    env = MujocoEnv()
    env.reset()
    task_cfg = load_yaml("configs/task.yaml")
    cam = RGBDCamera(env, "scene_cam", 640, 480)
    frames = FrameManager(env, base_body=env.base_body, camera_name="scene_cam")
    est = PoseEstimator(cam, frames, task_cfg["vision"], table_top=0.40)

    rgb, depth = cam.render()
    pose = est.estimate(rgb, depth)
    det = pose.detection
    annotated = ColorDetector.annotate(rgb, det)
    vis = hstack_images(annotated, det.mask)
    imageio.imwrite("outputs/screenshots/vision_detection.png", vis)

    gt = env.body_position("energy_unit")
    print("视觉检测:", "成功" if pose.found else "失败")
    if pose.found:
        print(f"  像素质心 = ({pose.pixel[0]:.1f}, {pose.pixel[1]:.1f})")
        print(f"  估计世界坐标 = {np.round(pose.position_world,4)}")
        print(f"  真值(body)   = {np.round(gt,4)}")
        print(f"  水平位置误差 = {np.linalg.norm(pose.position_world[:2]-gt[:2])*1000:.1f} mm")
    print("已保存 vision_detection.png 到 outputs/screenshots/")


if __name__ == "__main__":
    main()

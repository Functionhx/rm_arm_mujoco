"""测试 RGB-D 相机：渲染并保存 RGB 与深度伪彩色图，验证反投影精度。"""

import sys, pathlib as _pl
sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))


import numpy as np
import imageio.v2 as imageio

from src.geometry.frames import FrameManager
from src.geometry.transforms import invert_transform, transform_point
from src.sim.camera import RGBDCamera
from src.sim.mujoco_env import MujocoEnv
from src.utils.debug_draw import depth_to_colormap


def main():
    env = MujocoEnv()
    env.reset()
    cam = RGBDCamera(env, "scene_cam", 640, 480)
    rgb, depth = cam.render()
    imageio.imwrite("outputs/screenshots/cam_rgb.png", rgb)
    imageio.imwrite("outputs/screenshots/cam_depth.png", depth_to_colormap(depth))
    print("内参矩阵 K=\n", np.round(cam.K, 2))

    # 反投影精度校验（对物体顶部中心）
    gt = env.body_position("energy_unit") + np.array([0, 0, 0.14])
    Twc = env.camera_pose("scene_cam")
    p_mj = transform_point(invert_transform(Twc), gt)
    p_cv = np.array([p_mj[0], -p_mj[1], -p_mj[2]])
    u = cam.K[0, 0] * p_cv[0] / p_cv[2] + cam.K[0, 2]
    v = cam.K[1, 1] * p_cv[1] / p_cv[2] + cam.K[1, 2]
    est = cam.pixel_to_world(int(u), int(v), depth[int(v), int(u)])
    print(f"反投影校验: 真值={np.round(gt,4)} 估计={np.round(est,4)} 误差={np.linalg.norm(est-gt)*1000:.1f}mm")
    print("已保存 cam_rgb.png / cam_depth.png 到 outputs/screenshots/")


if __name__ == "__main__":
    main()

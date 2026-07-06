"""测试逆运动学：求解到若干目标位姿并报告误差。"""

import sys, pathlib as _pl
sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))


import numpy as np

from src.geometry.transforms import make_transform, rpy_to_matrix
from src.kinematics.inverse_kinematics import IKSolver
from src.sim.mujoco_env import MujocoEnv


def main():
    env = MujocoEnv()
    env.reset()
    ik = IKSolver(env)
    R_down = rpy_to_matrix(np.pi, 0, 0)
    targets = [
        ("物体上方", np.array([0.48, 0.02, 0.63])),
        ("抓取点", np.array([0.48, 0.02, 0.47])),
        ("放置区上方", np.array([0.50, -0.32, 0.60])),
    ]
    print("逆运动学 (DLS Jacobian) 测试:")
    for name, p in targets:
        q, ok, err = ik.solve(make_transform(R_down, p))
        print(f"  {name:10s} success={ok}  pos_err={err*1000:.2f} mm  q={np.round(q,3)}")


if __name__ == "__main__":
    main()

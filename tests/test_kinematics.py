"""运动学单元测试：FK 与 MuJoCo 一致；IK 收敛到目标。"""
import numpy as np

from src.geometry.transforms import make_transform, rpy_to_matrix
from src.kinematics.forward_kinematics import ForwardKinematics
from src.kinematics.inverse_kinematics import IKSolver
from src.sim.mujoco_env import MujocoEnv


def test_fk_matches_env():
    env = MujocoEnv(); env.reset()
    fk = ForwardKinematics(env)
    q = env.get_arm_qpos()
    T_fk = fk.compute(q)
    T_env = env.ee_pose()
    assert np.allclose(T_fk[:3, 3], T_env[:3, 3], atol=1e-6)


def test_ik_reaches_target():
    env = MujocoEnv(); env.reset()
    ik = IKSolver(env)
    R = rpy_to_matrix(np.pi, 0, 0)
    target = make_transform(R, np.array([0.48, 0.02, 0.47]))
    q, ok, err = ik.solve(target)
    assert ok
    assert err < 0.005


def test_ik_respects_limits():
    env = MujocoEnv(); env.reset()
    ik = IKSolver(env)
    R = rpy_to_matrix(np.pi, 0, 0)
    target = make_transform(R, np.array([0.45, 0.0, 0.5]))
    q, ok, err = ik.solve(target)
    assert np.all(q >= ik.q_lower - 1e-6)
    assert np.all(q <= ik.q_upper + 1e-6)

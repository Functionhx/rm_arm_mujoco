"""批量随机化实验：统计取放成功率、视觉误差、放置误差、单次耗时。"""

import sys, pathlib as _pl
sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import argparse
import numpy as np

from src.sim.mujoco_env import MujocoEnv
from src.task.pick_place_task import PickPlaceTask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = MujocoEnv()
    task = PickPlaceTask(env=env)
    rng = np.random.default_rng(args.seed)

    succ, vis_err, place_err, times = [], [], [], []
    for i in range(args.n):
        env.reset()
        task.randomize_object(rng)
        t0 = env.time
        res = task.run()
        dt = env.time - t0
        succ.append(res["success"])
        if res["vision_error"] is not None:
            vis_err.append(res["vision_error"] * 1000)
        place_err.append(res["place_xy_error"] * 1000)
        times.append(dt)
        print(f"[{i+1:2d}/{args.n}] success={res['success']} "
              f"vis_err={ (res['vision_error'] or 0)*1000:5.1f}mm "
              f"place_err={res['place_xy_error']*1000:5.1f}mm  t={dt:4.1f}s")

    vis_err = np.array(vis_err); place_err = np.array(place_err); times = np.array(times)
    ok = np.array(succ)
    print("\n===== 统计 (N=%d) =====" % args.n)
    print("成功率:            %.0f%% (%d/%d)" % (100 * ok.mean(), ok.sum(), args.n))
    print("视觉位置误差 均值/最大: %.1f / %.1f mm" % (vis_err.mean(), vis_err.max()))
    print("放置水平误差 均值/最大: %.1f / %.1f mm" % (place_err.mean(), place_err.max()))
    print("单次任务耗时 均值:     %.1f s" % times.mean())


if __name__ == "__main__":
    main()

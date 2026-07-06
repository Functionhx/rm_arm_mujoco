<a name="top"></a>
<h1 align="center">rm_arm_mujoco</h1>
<p align="center">
  <b>Vision-Guided Robotic-Arm Pick-and-Place in MuJoCo</b><br>
  <sub>For the RoboMaster 2026 University League Engineering Challenge</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/MuJoCo-3.10-EE7733">
  <img src="https://img.shields.io/badge/package_manager-uv-DE5FE9">
  <img src="https://img.shields.io/badge/tests-10%20passing-3FB950">
  <img src="https://img.shields.io/badge/license-MIT-2EA043">
</p>

<p align="center">
  <img src="docs/media/pick_place.gif" width="70%" alt="pick-and-place demo">
</p>

<p align="center"><a href="README.md">简体中文</a> ｜ <b>English</b></p>

---

`rm_arm_mujoco` is a **simulation-first** implementation of a full vision-guided
manipulation pipeline for an automatic **pick-and-place** task, built on the
[MuJoCo](https://mujoco.org/) physics engine. The task is modeled after the
**RoboMaster 2026 University League (RMUL) Engineering Challenge**, in which a robot
must grasp an *energy unit* and place it into a target receptacle.

The system perceives the target with an RGB-D camera, estimates its pose, plans and
executes a smooth grasp trajectory via inverse kinematics, and drives the whole
pick → transport → place sequence with a finite-state machine — closing the loop
from **perception** to **planning** to **control**. The core value proposition is
that visual perception lets the arm *adapt online* to a target whose position varies,
where a fixed replay trajectory would fail.

> A companion experiment report (in Chinese) is included at
> [`docs/report.pdf`](docs/report.pdf).

## ✨ Key Features

- **RGB-D perception** — off-screen RGB + depth rendering, intrinsics from FoV, pixel
  back-projection and ray–plane localization on the known support plane.
- **Color-segmentation detection** — HSV thresholding + morphology + largest-contour
  centroid (with an optional ArUco/`solvePnP` pose estimator for comparison).
- **Hand–eye coordinate transforms** — clean SE(3) utilities and a frame manager for
  `world / base / camera / object / ee`.
- **Damped least-squares (DLS) inverse kinematics** — MuJoCo Jacobian, singularity-robust,
  joint-limit-aware.
- **Smooth trajectories** — quintic-polynomial joint interpolation and Cartesian
  straight-line motion; arm gravity compensation removes steady-state droop.
- **FSM task controller** — `SEARCH → APPROACH → DESCEND → GRASP → LIFT → MOVE →
  INSERT → RELEASE → RETREAT → DONE` with retries.
- **Reproducible tooling** — [`uv`](https://docs.astral.sh/uv/)-managed environment,
  unit + integration tests, one-command demo, video/figure recording, and a
  randomized benchmark harness.

## 🔬 Method

<p align="center"><img src="docs/media/scene.png" width="49%" alt="scene">
<img src="docs/media/vision.png" width="49%" alt="vision detection"></p>

```
RGB-D camera ─▶ color detection ─▶ pose estimation ─▶ hand-eye transform
                                                             │
       joint servo ◀── DLS-IK ◀── grasp & trajectory planning
                                                             │
                                          finite-state machine (task logic)
```

The pick-and-place sequence executed by the state machine:

<p align="center"><img src="docs/media/pipeline.png" width="92%" alt="pipeline stages"></p>

<p align="center"><sub>
(a) detect &amp; align · (b) descend &amp; grasp · (c) lift · (d) move to target ·
(e) place · (f) release &amp; retreat
</sub></p>

## 📊 Results

Measured over **20 randomized trials** (object initial position randomized on the table):

| Metric | Result |
| :-- | :-- |
| Pick-and-place success rate | **100 %** (20/20) |
| Vision position error (mean / max) | 4.9 / 16.5 mm |
| Placement error (mean / max) | 20.9 / 43.0 mm |
| Perception time (detect + pose) | ~0.6 ms |
| Task duration (sim time) | ~6.7 s |
| Robustness to target displacement | 100 % within ±100 mm horizontal |

Reproduce with `uv run python scripts/benchmark.py -n 20`.

## 🚀 Installation

Requires [`uv`](https://docs.astral.sh/uv/). The Franka Panda model (from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)) is vendored
under `assets/robots/panda/`, so no extra download is needed.

```bash
git clone https://github.com/Functionhx/rm_arm_mujoco.git
cd rm_arm_mujoco
uv sync                     # create .venv and install dependencies
```

## 🕹️ Usage

```bash
uv run python scripts/run_demo.py         # run one full pick-and-place task (headless)
uv run python scripts/view_live.py        # live OpenCV window of the task
uv run python scripts/record_demo.py      # record video + figures to outputs/
uv run python scripts/benchmark.py -n 20  # randomized success-rate benchmark

uv run python scripts/test_camera.py      # RGB-D render + back-projection check
uv run python scripts/test_vision.py      # detection + pose-estimation check
uv run python scripts/test_ik.py          # inverse-kinematics check

uv run pytest                             # unit + integration tests (10 tests)
```

> **macOS note:** MuJoCo's native `launch_passive` viewer requires `mjpython`, which
> conflicts with uv's bundled Python. `scripts/view_live.py` therefore uses a live
> OpenCV window that works under plain `uv run python`.

## ⚙️ Configuration

All tunables live in `configs/` (no code edits needed):

| File | Contents |
| :-- | :-- |
| `configs/sim.yaml`   | scene path, control frequency, camera & render settings |
| `configs/robot.yaml` | joint/actuator names, home pose, gripper, IK parameters |
| `configs/task.yaml`  | object/target bodies, HSV thresholds, grasp/place heights, FSM params |

## 📁 Repository Structure

```
rm_arm_mujoco/
├── configs/            # sim.yaml / robot.yaml / task.yaml
├── assets/
│   ├── robots/panda/   # Franka Panda (vendored from MuJoCo Menagerie, Apache-2.0)
│   ├── objects/        # energy_unit (cube.xml), tray.xml
│   └── scenes/         # pick_place_scene.xml
├── src/
│   ├── sim/            # mujoco_env, camera (RGB-D), recorder
│   ├── vision/         # color_detector, pose_estimator, aruco_pose
│   ├── geometry/       # transforms (SE3), frames
│   ├── kinematics/     # forward/inverse kinematics, jacobian
│   ├── planning/       # grasp_planner, trajectory, collision_check
│   ├── control/        # arm_controller, gripper_controller
│   ├── task/           # state_machine, pick_place_task
│   └── utils/          # logger, debug_draw
├── scripts/            # run_demo, view_live, record_demo, benchmark, test_*
├── tests/              # test_transforms / test_kinematics / test_state_machine
└── docs/               # report.pdf, media/
```

## 📝 Notes on the Simulation Proxy

The real RoboMaster energy unit is a Ø95/Ø80 mm dumbbell that would be grasped by a
custom large gripper. To match the **Franka Panda's 80 mm gripper stroke**, the
simulated object keeps the dumbbell topology and 150 mm height but uses a reduced top
cap (Ø50) and a graspable central waist (Ø30) so the parallel gripper can grasp it
top-down. This proxy does not affect validation of the vision-guided pick-and-place
*algorithms*, which transfer to a real arm at the control level. Grasp firmness during
transport is enforced by a weld equality activated on gripper close (equivalent to a
firm physical grip). See the report for details.

## 📌 Citation

```bibtex
@software{fan2026_rm_arm_mujoco,
  author  = {Fan, Yuchen},
  title   = {rm\_arm\_mujoco: Vision-Guided Robotic-Arm Pick-and-Place in MuJoCo},
  year    = {2026},
  url     = {https://github.com/Functionhx/rm_arm_mujoco}
}
```

## 🙏 Acknowledgements

- [MuJoCo](https://mujoco.org/) and [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Franka Panda model) by Google DeepMind.
- Task setting inspired by the **DJI RoboMaster 2026** University League Engineering Challenge.

## 📄 License

Released under the [MIT License](LICENSE). The vendored Panda model is under Apache-2.0
(see `assets/robots/panda/LICENSE`).

<p align="right"><a href="#top">⬆ Back to top</a></p>

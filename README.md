# tabletenniszed — Table Tennis Ball Prediction & Planning (C++17 / ROS2)

Real-time table tennis ball trajectory prediction and racket-hit planning,
driven by a ZED stereo camera.
Refactored from the original Python/ROS1 implementation into C++17/ROS2 Humble.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [File Structure](#file-structure)
4. [File Descriptions](#file-descriptions)
5. [Dependencies & Installation](#dependencies--installation)
6. [Build](#build)
7. [Run](#run)
8. [Parameter Tuning (no recompile)](#parameter-tuning-no-recompile)
9. [ROS2 Topics](#ros2-topics)

---

## System Overview

```
ZED Camera (remote PC)
        │  UDP JSON packets
        ▼
  ┌─────────────┐   ball pos / vel   ┌───────────────────┐
  │ ZedReceiver │ ─────────────────► │ TableTennisNode   │
  │  (3 threads)│                    │                   │
  └─────────────┘                    │  BallPredictor    │
        │                            │  (trajectory sim) │
        │ torso pose                 │                   │
        ▼                            │  Planner          │
  /torso_pose_origin_zed             │  (hit planning)   │
                                     └────────┬──────────┘
                                              │
                        ┌─────────────────────┼──────────────────────┐
                        ▼                     ▼                      ▼
              /predicted_ball_position  /predicted_racket_normal  /predicted_ball_predict_time
              /predicted_ball_velocity  /predicted_racket_velocity
```

**Data flow:**

1. The ZED camera PC sends JSON packets over UDP (position + pose data).
2. `ZedReceiver` receives, filters, and Kalman-filters the ball position.
3. `TableTennisNode` detects the incoming ball, predicts its trajectory,
   and plans the required racket pose to hit it to a target point.
4. All results are published as ROS2 topics at the planning rate (100 Hz).

---

## Architecture

### Module separation

The codebase is split into three independent layers:

| Layer | CMake target | ROS2 dependency | Description |
|-------|-------------|-----------------|-------------|
| Core algorithms | `tt_core` (static lib) | ✗ | KF, AKF, predictor, planner |
| ZED receiver | `tt_zed_receiver` (static lib) | ✓ | UDP + Kalman wrappers |
| ROS2 node | `table_tennis_node` (executable) | ✓ | Orchestration, publishing |

### Threading model

`ZedReceiver` runs three background threads:

```
Thread 1 — UDP receive:   recv() ──► JSON parse ──► deque (maxlen=2)
Thread 2 — Ball proc (60 Hz):  deque ──► spatial filter ──► KF/AKF ──► BallData
Thread 3 — Torso proc (60 Hz): deque ──► AprilTag pose ──► ROS2 publish
```

`TableTennisNode` runs a single timer callback at `planning_rate` Hz (default 100 Hz)
on the ROS2 executor thread.

### Parameter system

All hyperparameters live in `config/params.yaml` and are loaded at startup
via the ROS2 parameter server — **no recompilation needed to change them**.
The C++ `Params` struct (`include/tabletenniszed/params.hpp`) is the in-memory
container that gets populated from the YAML file by `loadParamsFromNode()`
inside `table_tennis_node.cpp`.

```
config/params.yaml
       │
       │  ros2 launch --params-file
       ▼
  ROS2 parameter server
       │
       │  declare_parameter() / get_parameter()
       ▼
  Params struct  ──►  ZedReceiver / AKF / BallPredictor / TableTennisNode
```

---

## File Structure

```
tabletenniszed_cpp/
├── CMakeLists.txt                          # Build definition
├── package.xml                             # ROS2 package manifest
├── .gitignore
│
├── config/
│   └── params.yaml                         # ← All tunable hyperparameters
│
├── include/tabletenniszed/
│   ├── params.hpp                          # Params struct (runtime config container)
│   ├── kalman_filter_3d.hpp                # Standard KF interface
│   ├── adaptive_kalman_filter_3d.hpp       # Physics-based AKF interface
│   ├── ball_predictor.hpp                  # Trajectory predictor interface
│   ├── planner.hpp                         # Hit planner interface
│   └── zed_receiver.hpp                    # ZED UDP receiver interface
│
├── src/
│   ├── kalman_filter_3d.cpp
│   ├── adaptive_kalman_filter_3d.cpp
│   ├── ball_predictor.cpp
│   ├── planner.cpp
│   ├── zed_receiver.cpp
│   └── table_tennis_node.cpp               # main() + ROS2 node
│
└── launch/
    └── tabletenniszed.launch.py            # Launch file
```

---

## File Descriptions

### `config/params.yaml`

The single source of truth for all tunable values.
Edit this file and re-launch to change behaviour without recompiling.

Key sections:

| Section | Parameters |
|---------|-----------|
| Communication | `udp_host`, `udp_port`, `planning_rate` |
| Physics | `gravity`, `k_drag`, `ch` (horiz. restitution), `cv` (vert. restitution) |
| Kalman filter | `kalman_*` (noise, initial covariance) |
| Adaptive KF | `akf_*` (physics-based EKF settings) |
| Marker filter | Spatial bounding box in table frame |
| Strike region | Where the racket is expected to intercept the ball |
| Coordinate transforms | `table_in_world`, `camera_to_torso`, `t_origin_to_table` (calibration matrices) |

---

### `include/tabletenniszed/params.hpp`

Defines the `Params` struct — a plain C++ data container with one field per
parameter.  It is **not** a config file itself; it is the in-memory type used
to pass runtime config between components.

```cpp
struct Params {
    std::string zed_udp_host = "172.16.2.101";
    double      k_drag       = 0.20;
    // ... (all fields have conservative defaults as fallback)
    Eigen::Matrix<double,6,6> buildAkfPInit() const;  // helper
};
```

The default values serve only as fallbacks if no YAML file is loaded.
The canonical values are in `config/params.yaml`.

---

### `include/tabletenniszed/kalman_filter_3d.hpp` + `src/kalman_filter_3d.cpp`

**Standard 6-state constant-velocity Kalman filter.**

- State: `x = [px, py, pz, vx, vy, vz]`
- Observation: position only `[px, py, pz]`
- Per-axis noise tuning via constructor parameters
- Auto-resets on: reversed timestamp, `dt > dt_max`, or position jump `> reset_threshold`
- Used as a fallback when `use_adaptive_kalman = false`

Key API:
```cpp
Eigen::Vector3d update(pos_observed, timestamp);
std::optional<Eigen::Vector3d> getEstimatedVelocity();
```

---

### `include/tabletenniszed/adaptive_kalman_filter_3d.hpp` + `src/adaptive_kalman_filter_3d.cpp`

**Physics-based Adaptive Extended Kalman Filter (AKF).**

- Uses the same 6-state model as KF but replaces linear prediction with a
  physics model: gravity + quadratic drag + table bounce reflection
- Jacobian of the nonlinear transition is computed analytically for
  covariance propagation (Extended KF)
- Process/measurement noise scaled adaptively with `dt` and camera distance
- Detects "return hit" events (ball direction reversal) and resets automatically
- **Default filter** (`use_adaptive_kalman = true`)

Key API:
```cpp
// Returns (filtered_pos, filtered_vel), both nullopt on first call or after reset
std::pair<std::optional<Vector3d>, std::optional<Vector3d>>
    update(pos_observed, timestamp, camera_distance);
```

---

### `include/tabletenniszed/ball_predictor.hpp` + `src/ball_predictor.cpp`

**Forward trajectory simulator.**

Integrates ball motion under:
- Quadratic aerodynamic drag: `a_drag = -k * ||v|| * v`
- Gravity: `a_grav = [0, 0, -g]`
- Table bounce (horizontal/vertical restitution `Ch`, `Cv`)

Integration method: Euler forward at configurable step `dt` (default 2 ms).

Key API:
```cpp
// Single target time
PredictResult predictAtTime(initial_pos, initial_vel, target_time, dt);

// Multiple target times — single simulation pass (efficient)
PredictResultList predictAtTimes(initial_pos, initial_vel, times, dt);
```

Public members `k`, `Ch`, `Cv` can be modified at runtime.

---

### `include/tabletenniszed/planner.hpp` + `src/planner.cpp`

**Inverse dynamics hit planner.**

Given:
- Impact point `p_hit` and incoming ball velocity `v_in`
- Desired flight time `T_flight` to land at `p_target`
- Drag coefficient `k_drag` and racket restitution `e`

Computes:
- Required ball exit velocity `v_out` (analytical linear-drag inverse)
- Racket surface normal `n` and racket velocity `Vr` (frictionless collision model)

Sub-functions are all public for testability:

```cpp
HitPlan planPingpongHit(p_hit, v_in, T_flight, p_target, k_drag, e);
Vector3d solveOutgoingVelocityFixedTime(p_hit, p_target, T, k_drag);
RacketPose solveRacketPoseAndSpeed(v_in, v_out, e);
```

---

### `include/tabletenniszed/zed_receiver.hpp` + `src/zed_receiver.cpp`

**ZED camera UDP receiver.**

Listens on a UDP socket and processes JSON packets from the ZED camera PC.
Expected JSON fields per packet:

```json
{
  "timestamp": 1712345678.123,
  "ball_camera": {"x": 0.1, "y": 0.0, "z": 1.5, "valid": true},
  "ball_table":  {"x": 0.3, "y": 0.0, "z": 0.1, "valid": true},
  "camera_pose": {
    "valid": true,
    "position": {"x": ..., "y": ..., "z": ...},
    "rotation": [r00, r01, ..., r22]   // 3x3 row-major flat array
  }
}
```

Processing pipeline:
1. Parse JSON → `ParsedUdpData`
2. Spatial bounding-box filter (in table frame)
3. Kalman or AKF filter → `BallData` (filtered pos + vel + raw pos + world pos)
4. Reject ball on table surface (`z < 0.08 m` while in table bounds)
5. For torso: AprilTag pose → camera→torso transform → origin frame → publish

Coordinate transforms applied:
```
table frame ──► world frame      (table_in_world)
table frame ──► origin frame     (table_in_origin)
camera frame ──► torso frame     (camera_to_torso)
```

---

### `src/table_tennis_node.cpp`

**Main ROS2 node: `table_tennis_predictor`.**

Initialization (two-phase to avoid `shared_from_this()` in constructor):
```
main()
  └─ make_shared<TableTennisNode>()   ← constructor: load params, create publishers
       └─ node->initialize()          ← create ZedReceiver + planning timer
```

Planning loop (`executePlanning()`, called at `planning_rate` Hz):

```
1. Poll ZedReceiver for latest BallData
2. If new data:
   a. detectStart()  — check vx < -0.1 m/s, scan trajectory for strike-region entry
   b. If detected: refine time_to_strike in a ±0.1 s window at 5 ms steps
   c. Predict ball state at time_to_strike
   d. planPingpongHit() → racket normal + velocity
3. Clamp outputs to strike region
4. Decay time_to_strike by 1/rate per cycle
5. Publish all topics
6. Append row to prediction_data{N}.csv
```

CSV log files are auto-numbered (`data/prediction_data1.csv`, `data/prediction_data2.csv`, …).

---

### `launch/tabletenniszed.launch.py`

```bash
# Default launch (uses config/params.yaml)
ros2 launch tabletenniszed tabletenniszed.launch.py

# Override UDP address on command line
ros2 launch tabletenniszed tabletenniszed.launch.py udp_host:=192.168.1.10 udp_port:=9000

# Use a custom parameter file
ros2 launch tabletenniszed tabletenniszed.launch.py params_file:=/path/to/my_params.yaml
```

Command-line arguments (`udp_host`, `udp_port`, `planning_rate`) **override**
the values in the YAML file.

---

## Dependencies & Installation

### Required

| Dependency | Version | Install |
|------------|---------|---------|
| Ubuntu | 22.04 LTS | — |
| ROS2 | Humble | See below |
| Eigen3 | ≥ 3.3 | `sudo apt install libeigen3-dev` |
| nlohmann/json | any | `sudo apt install nlohmann-json3-dev` |
| colcon | — | installed with ROS2 |
| Python3 | ≥ 3.10 | system default |

### ROS2 Humble installation (Ubuntu 22.04)

```bash
# Add ROS2 apt repository
sudo apt install software-properties-common curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list

sudo apt update
sudo apt install ros-humble-desktop python3-colcon-common-extensions
```

### ROS2 message packages

```bash
sudo apt install \
    ros-humble-geometry-msgs \
    ros-humble-std-msgs
```

### All system deps at once

```bash
sudo apt install \
    libeigen3-dev \
    nlohmann-json3-dev \
    ros-humble-desktop \
    python3-colcon-common-extensions \
    ros-humble-geometry-msgs \
    ros-humble-std-msgs
```

---

## Build

```bash
# Clone the repository
git clone git@github.com:Yinghui-Lee/tabletennisplanner.git
cd tabletennisplanner

# Source ROS2 environment
source /opt/ros/humble/setup.bash

# Build (Release mode for best performance)
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source the local install overlay
source install/setup.bash
```

Build output:
- `install/tabletenniszed/lib/tabletenniszed/table_tennis_node` — executable
- `install/tabletenniszed/share/tabletenniszed/config/params.yaml` — installed config
- `install/tabletenniszed/share/tabletenniszed/launch/` — launch files

---

## Run

```bash
# Source both ROS2 and the install overlay (always required in a new terminal)
source /opt/ros/humble/setup.bash
source install/setup.bash

# Launch with default parameters
ros2 launch tabletenniszed tabletenniszed.launch.py

# Launch with a different ZED camera address
ros2 launch tabletenniszed tabletenniszed.launch.py udp_host:=172.16.2.50

# Launch with a custom parameter file (e.g. for a different table calibration)
ros2 launch tabletenniszed tabletenniszed.launch.py \
    params_file:=/home/user/my_calibration.yaml
```

**Check that the node is running:**
```bash
ros2 node list
# → /table_tennis_predictor

ros2 topic list
# → /predicted_ball_position
# → /predicted_ball_velocity
# → /predicted_ball_predict_time
# → /predicted_racket_normal
# → /predicted_racket_velocity
# → /torso_pose_origin_zed

# Monitor predicted ball position
ros2 topic echo /predicted_ball_position
```

---

## Parameter Tuning (no recompile)

Edit `config/params.yaml`, then re-launch.  No rebuilding needed.

**Common tuning scenarios:**

### Change camera / table calibration
```yaml
# Recalibrate: replace these 4×4 matrices (row-major, 16 floats)
table_in_world:    [...]   # table frame → world frame
camera_to_torso:   [...]   # ZED camera → torso frame
t_origin_to_table: [...]   # origin frame → table frame (pure translation)
```
`origin_in_world` and `table_in_origin` are derived automatically at startup.

### Adjust Kalman filter aggressiveness
```yaml
# More aggressive smoothing (larger Q = trust model more, smaller R = trust sensor less)
akf_q_pos_base: 0.0005   # increase for faster response
akf_r_pos:      0.00005  # decrease to trust sensor more
```

### Change strike region
```yaml
strike_region_x_min: 0.35
strike_region_x_max: 0.55
strike_region_z_min: 0.80
strike_region_z_max: 1.50
```

### Switch from AKF to standard KF
```yaml
use_adaptive_kalman: false   # use the simpler constant-velocity KF
```

### Change ball drag / restitution model
```yaml
k_drag: 0.25    # aerodynamic drag coefficient
ch:     0.80    # horizontal restitution at table bounce
cv:     0.90    # vertical restitution at table bounce
```

---

## ROS2 Topics

| Topic | Type | Frame | Description |
|-------|------|-------|-------------|
| `/predicted_ball_position` | `geometry_msgs/PointStamped` | `origin_frame_w` | Predicted ball position at strike time |
| `/predicted_ball_velocity` | `geometry_msgs/TwistStamped` | `origin_frame_w` | Predicted ball velocity at strike time |
| `/predicted_ball_predict_time` | `geometry_msgs/PointStamped` | `origin_frame_w` | Time-to-strike in `point.x` (seconds) |
| `/predicted_racket_normal` | `geometry_msgs/Vector3Stamped` | `origin_frame_w` | Required racket surface normal |
| `/predicted_racket_velocity` | `geometry_msgs/TwistStamped` | `origin_frame_w` | Required racket velocity |
| `/torso_pose_origin_zed` | `geometry_msgs/PoseStamped` | `origin_frame_w` | Torso pose from AprilTag (ZED) |

**Coordinate frames:**

- `origin_frame_w` — robot body origin frame (defined by `t_origin_to_table` calibration)
- Table frame — table surface centre, X along table length, Z pointing up
- World frame — absolute world frame (defined by `table_in_world` calibration)

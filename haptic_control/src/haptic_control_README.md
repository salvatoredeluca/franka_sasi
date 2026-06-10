# haptic_control — Teleoperation Node

This document describes the `haptic_control` package in detail, the C++ node `haptic_teleop_node` and the launch file `dual_haptic_teleop.launch.py` for Franka FR3 teleoperation with two Haption Desktop 6D haptic devices.

---

## Package Structure

```
haptic_control/
├── CMakeLists.txt
├── package.xml
├── src/
│   └── haptic_teleop_node.cpp    ← main teleoperation node
└── launch/
    ├── haptic_teleop.launch.py       ← launches the teleoperation node
    └── dual_haptic.launch.py  ← launches both devices
```

---

## haptic_teleop_node.cpp

### Overview

The node reads in real time the **pose** and **velocity** of the Haption end-effector, transforms them into the Franka base frame and publishes:

- The absolute pose (current position + orientation)
- The relative pose (delta with respect to the initial position at launch)
- The Cartesian velocity (linear + angular)

The node is written in C++ to ensure low latency, compatible with the Franka operating frequency (~1kHz).

---

### Configurable Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `base_frame` | string | `fr3_link0` | Franka base frame in the TF tree |
| `haptic_base_frame` | string | `haptic_base` | Parent TF frame for the haptic EE (for RViz visualization) |
| `ee_frame` | string | `haptic_ee` | TF frame name for the Haption EE |
| `linear_scale` | double | `1.0` | Scale factor for linear velocity. With `0.5` the Franka moves at half speed |
| `angular_scale` | double | `1.0` | Scale factor for angular velocity |
| `pose_mode` | string | `both` | Pose publishing mode: `absolute`, `relative`, or `both` |
| `haptic_rotation_matrix` | double[9] | identity | Row-major 3x3 rotation matrix mapping Haption frame → Franka frame. Used for velocity commands and PoseStamped topics |

---

### Topics

### Subscribe

| Topic | Type | Description |
| --- | --- | --- |
| `out_virtuose_pose` | `raptor_api_interfaces/OutVirtuosePose` | Absolute EE pose of the Haption in its base frame |
| `out_virtuose_speed` | `raptor_api_interfaces/OutVirtuoseSpeed` | Cartesian EE velocity of the Haption |

> With namespace `left` the topics become `/left/out_virtuose_pose` and `/left/out_virtuose_speed`.
> 

### Publish

| Topic | Type | Description |
| --- | --- | --- |
| `haptic_control/target_pose_absolute` | `geometry_msgs/PoseStamped` | Absolute pose transformed into Franka frame |
| `haptic_control/target_pose_relative` | `geometry_msgs/PoseStamped` | Pose delta with respect to initial home |
| `haptic_control/ee_twist_command` | `geometry_msgs/Twist` | Cartesian velocity in Franka frame |
| `/tf` | `tf2_msgs/TFMessage` | Transform `haptic_base_frame → ee_frame` |

---

### Code Details

### Initialization and Parameters

```cpp
this->declare_parameter("linear_scale",          1.0);
this->declare_parameter("angular_scale",         1.0);
this->declare_parameter("base_frame",            std::string("fr3_link0"));
this->declare_parameter("haptic_base_frame",     std::string("haptic_base"));
this->declare_parameter("ee_frame",              std::string("haptic_ee"));
this->declare_parameter("pose_mode",             std::string("both"));
this->declare_parameter("haptic_rotation_matrix",
  std::vector<double>{1,0,0, 0,1,0, 0,0,1});
```

Parameters are declared with default values and can be overridden from the launch file. This makes the node reusable for both devices by simply changing the namespace, `haptic_base_frame` and `ee_frame`.

### Rotation Matrix Haption → Franka

```cpp
auto m = this->get_parameter("haptic_rotation_matrix").as_double_array();
R_h2f_ << m[0], m[1], m[2],
          m[3], m[4], m[5],
          m[6], m[7], m[8];
```

This matrix maps vectors from the Haption frame to the Franka base frame. It is passed as a flat row-major 9-value array from the launch file and can be set independently for each device.

RPY is not used because axis swaps (det = -1) cannot be expressed as a pure rotation. The full 3x3 matrix covers any linear remapping.

`quat_h2f_` expresses the same rotation as a quaternion, used to rotate orientation in the PoseStamped topics.

> **Note**: `R_h2f_` is applied **only** to velocity commands and PoseStamped topics. The TF is published raw (without `R_h2f_`) relative to `haptic_base_frame`, so that `haptic_ee` is correctly a child of the haptic device base frame in the TF tree.
> 

### Rotation Matrix Calibration

To determine the correct `haptic_rotation_matrix` for each device, set it to identity in the launch file and move the device one axis at a time while observing the `haptic_ee` frame in RViz (Fixed Frame = `world`):

1. Move only along X → note which RViz arrow moves (red=X, green=Y, blue=Z)
2. Move only along Y → note the result
3. Move only along Z → note the result

Build the matrix row by row: each row specifies which Haption axis maps to that Franka axis.

### Conditional Publishers

```cpp
if (pose_mode_ == "absolute" || pose_mode_ == "both") {
  abs_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
    "haptic_control/target_pose_absolute", 10);
}
if (pose_mode_ == "relative" || pose_mode_ == "both") {
  rel_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
    "haptic_control/target_pose_relative", 10);
}
```

Publishers are created only if needed based on `pose_mode`, avoiding unnecessary topics and reducing computational overhead.

### poseCallback — Absolute and Relative Pose

```cpp
Eigen::Quaterniond quat(
  msg->virtuose_pose.rotation.w,  // w first!
  msg->virtuose_pose.rotation.x,
  msg->virtuose_pose.rotation.y,
  msg->virtuose_pose.rotation.z);
quat.normalize();
```

Eigen constructs the quaternion with order `(w, x, y, z)` while ROS uses `(x, y, z, w)` — swapping the order would produce a completely wrong orientation. `normalize()` ensures the quaternion is unit length.

```cpp
// Initialize home pose on first message
if (!home_initialized_) {
  home_pos_  = pos;
  home_quat_ = quat;
  home_initialized_ = true;
}
```

On the first received message, the current pose is saved as **home**. This happens automatically at node launch — whatever position the device is in at that moment becomes the `(0, 0, 0)` reference for the relative pose.

```cpp
// Relative pose
Eigen::Vector3d delta_pos = R_h2f_ * (pos - home_pos_);
Eigen::Quaterniond delta_quat = (quat_h2f_ * home_quat_).inverse() * quat_franka;
```

`delta_pos` is the vector difference between the current pose and home, rotated into the Franka frame. `delta_quat` is the rotation from `home` to `current`, expressed in Franka frame — computed by composing the inverse of home with the current orientation.

### TF Broadcaster

```cpp
tf_msg.header.frame_id = haptic_base_frame_;  // "haptic_left_base" or "haptic_right_base"
tf_msg.child_frame_id  = ee_frame_;           // "haptic_ee_1" or "haptic_ee_2"
tf_broadcaster_.sendTransform(tf_msg);
```

Publishes on `/tf` the raw transform from `haptic_base_frame` to `haptic_ee`. This allows RViz to display the haptic device position in real time. The static transforms `world → haptic_left_base` and `world → haptic_right_base` are defined in the launch file.

### speedCallback — Velocity with Deadband

```cpp
if (v_lin.norm() < linear_deadband_) v_lin.setZero();
if (v_ang.norm() < angular_deadband_) v_ang.setZero();
```

The Haption generates small noise velocities even when stationary (~1e-4 m/s). The deadband zeroes velocities below the threshold, preventing the Franka from receiving unintended motion commands. The Euclidean norm is used instead of comparing individual components, making the threshold isotropic.

---

## haptic_teleop.launch.py

### Overview

Launches two instances of `haptic_teleop_node` simultaneously, one for each haptic device, with distinct namespaces, `haptic_base_frame` and `ee_frame`. Assumes that wrappers and impedance nodes are already running (launched with `dual_haptic.launch.py`).

### Launch Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `base_frame` | `fr3_link0` | Franka base frame, shared between both nodes |
| `linear_scale` | `1.0` | Linear velocity scale, applied to both |
| `angular_scale` | `1.0` | Angular velocity scale, applied to both |
| `pose_mode` | `both` | Pose mode: `absolute` | `relative` | `both` |

### Code Details

### Common Parameters Declared Once

```python
base_frame_arg = DeclareLaunchArgument(
  'base_frame',
  default_value='fr3_link0',
  description='Franka base frame'
)
```

Parameters are declared once and passed to both nodes via `LaunchConfiguration`. Changing a parameter from the command line applies it to both devices simultaneously.

### Static TF: world → haptic base frames

```python
tf_haptic_left_base = Node(
  package='tf2_ros',
  executable='static_transform_publisher',
  name='haptic_left_base',
  arguments=['0', '0', '0', '0', '0', '0', '1', 'world', 'haptic_left_base']
)

tf_haptic_right_base = Node(
  package='tf2_ros',
  executable='static_transform_publisher',
  name='haptic_right_base',
  arguments=['0', '0', '0', '0', '0', '0', '1', 'world', 'haptic_right_base']
)
```

These static transforms connect the haptic base frames to `world`, enabling RViz visualization with Fixed Frame = `world`. The translation and rotation values should be updated to reflect the real physical position of the devices in the lab.

### Two Instances with Distinct Namespace, haptic_base_frame and ee_frame

```python
# Device 1
haptic_teleop_1 = Node(
  namespace='left',
  parameters=[{
    'haptic_base_frame':      'haptic_left_base',
    'ee_frame':               'haptic_ee_1',
    'haptic_rotation_matrix': [1.0, 0.0, 0.0,
                                0.0, 1.0, 0.0,
                                0.0, 0.0, 1.0],
    ...
  }],
)

# Device 2
haptic_teleop_2 = Node(
  namespace='right',
  parameters=[{
    'haptic_base_frame':      'haptic_right_base',
    'ee_frame':               'haptic_ee_2',
    'haptic_rotation_matrix': [1.0, 0.0, 0.0,
                                0.0, 1.0, 0.0,
                                0.0, 0.0, 1.0],
    ...
  }],
)
```

The `namespace` ensures subscribers automatically connect to the correct device topics:

- `left` → subscribes to `/left/out_virtuose_pose` and `/left/out_virtuose_speed`
- `right` → subscribes to `/right/out_virtuose_pose` and `/right/out_virtuose_speed`

The distinct `haptic_base_frame` and `ee_frame` produce two separate TF frames in the TF tree, one per device. The `haptic_rotation_matrix` is set independently per device to account for their different physical orientations relative to the Franka.

---

## Usage

### Prerequisites

Before launching the teleoperation node, make sure `dual_haptic.launch.py` is running and both devices are calibrated.

### Launch

```bash
ros2 launch haptic_control dual_haptic_teleop.launch.py
```

With custom parameters:

```bash
ros2 launch haptic_control dual_haptic_teleop.launch.py \
  pose_mode:=relative \
  linear_scale:=0.5 \
  angular_scale:=0.5
```

### Verification

```bash
# List published topics
ros2 topic list | grep haptic_control

# Check relative pose device 1
ros2 topic echo /left/haptic_control/target_pose_relative --once

# Check velocity device 2
ros2 topic echo /right/haptic_control/target_cartesian_velocity

# Check TF tree
ros2 run tf2_ros tf2_echo world haptic_left_base
ros2 run tf2_ros tf2_echo haptic_left_base haptic_ee_1
ros2 run tf2_ros tf2_echo haptic_right_base haptic_ee_2
```
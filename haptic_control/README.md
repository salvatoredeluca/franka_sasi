# Haptic Teleoperation — Dual Device Setup

This document describes the complete procedure to launch and use two Haption Desktop 6D haptic devices (n70 and n71) simultaneously for Franka FR3 v2 teleoperation.

---

## System Architecture

```
Host (Ubuntu 20.04)
├── SvcHaptic_Desktop_6D_n70  (daemon - started by Docker entrypoint)
├── SvcHaptic_Desktop_6D_n71  (daemon - started by Docker entrypoint)
└── Docker Container (ros:humble)
    └── haption_ws
        ├── haption_raptor_api     — ROS2 driver for haptic devices
        ├── raptor_api_interfaces  — custom messages and services
        ├── test_impedance         — impedance test node
        └── haptic_control         — teleoperation node (haptic_teleop_node)
```

### Main Topics

| Topic | Type | Description |
| --- | --- | --- |
| `/left/out_virtuose_pose` | `OutVirtuosePose` | EE pose device n70 |
| `/left/out_virtuose_speed` | `OutVirtuoseSpeed` | EE velocity device n70 |
| `/right/out_virtuose_pose` | `OutVirtuosePose` | EE pose device n71 |
| `/right/out_virtuose_speed` | `OutVirtuoseSpeed` | EE velocity device n71 |
| `/left/haptic_control/target_pose_absolute` | `PoseStamped` | Absolute pose device 1 in Franka frame |
| `/left/haptic_control/target_pose_relative` | `PoseStamped` | Relative pose from home device 1 |
| `/left/haptic_control/target_cartesian_velocity` | `Twist` | Cartesian velocity device 1 |
| `/right/haptic_control/target_pose_absolute` | `PoseStamped` | Absolute pose device 2 in Franka frame |
| `/right/haptic_control/target_pose_relative` | `PoseStamped` | Relative pose from home device 2 |
| `/right/haptic_control/target_cartesian_velocity` | `Twist` | Cartesian velocity device 2 |

### Network Configuration

| Element | IP | Port |
| --- | --- | --- |
| Host | `192.168.9.101/24` | — |
| Host (virtual second IP) | `192.168.9.102/24` | — |
| Device n70 | `192.168.9.16` | UDP 5000 |
| Device n71 | `192.168.9.17` | UDP 5000 |
| Wrapper left (local bind) | `192.168.9.101` | UDP 12120 |
| Wrapper right (local bind) | `192.168.9.102` | UDP 12120 |

> **Note**: the second local IP (`192.168.9.102`) is required because the RaptorAPI hardcodes port `12120`. Having two distinct local IPs avoids the `EADDRINUSE` conflict between the two wrappers.
> 


# LD_LIBRARY_PATH for RaptorAPI

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/<workspace>/src/haption_raptor_api/Dependencies/RaptorAPI/bin/Linux/glibc-2.35/
```

# Verify that daemons are running

```bash
ps aux | grep -i svc
```

Both processes must be present:

```
./SvcHaptic_Desktop_6D_n70
./SvcHaptic_Desktop_6D_n71
```

If not running, start them manually:

```bash
cd ~/6DOFs_haptic_devices/n70/7.2.0000
./SvcHaptic_Desktop_6D_n70 &

cd ~/6DOFs_haptic_devices/n71/7.2.0000
./SvcHaptic_Desktop_6D_n71 &
```

### Calibrate the haptic devices 

```bash
# Calibrate n71
cd ~/6DOFs_haptic_devices/n71/7.2.0000
sudo ./calibrate_Desktop_6D_n71

# Calibrate n70
cd ~/6DOFs_haptic_devices/n70/7.2.0000
sudo ./calibrate_Desktop_6D_n70
```

### Add the second local IP (on host)

> **Note**: this command must be re-run after every reboot as the virtual IP is not persistent.
> 

```bash
sudo ip addr add 192.168.9.102/24 dev eno1
```

Verify:

```bash
ip addr show eno1
```

Both `192.168.9.101` and `192.168.9.102` must be present.

### Launch dual haptic (wrapper + impedance)

```bash
ros2 launch haptic_control dual_haptic.launch.py
```

This starts:

- `/left/raptor_api_wrapper` — ROS2 wrapper for n70
- `/left/test_impedance` — impedance node for n70
- `/right/raptor_api_wrapper` — ROS2 wrapper for n71
- `/right/test_impedance` — impedance node for n71

### Launch the teleoperation node

In a second terminal:

```bash
ros2 launch haptic_control dual_haptic_teleop.launch.py
```

Optional parameters:

```bash
ros2 launch haptic_control dual_haptic_teleop.launch.py \
  pose_mode:=relative \       # absolute | relative | both (default: both)
  linear_scale:=1.0 \         # linear velocity scale (default: 1.0)
  angular_scale:=1.0 \        # angular velocity scale (default: 1.0)
```

---

## Verification

### Check active topics

```bash
ros2 topic list | grep haptic
```

### Check incoming data

```bash
# Absolute pose device 1
ros2 topic echo /left/haptic_control/target_pose_absolute 

# Relative pose device 2
ros2 topic echo /right/haptic_control/target_pose_relative 

# Velocity device 1
ros2 topic echo /left/haptic_control/target_cartesian_velocity
```

### Check frequency

```bash
ros2 topic hz /left/haptic_control/target_cartesian_velocity
ros2 topic hz /right/haptic_control/target_cartesian_velocity
```

### Check TF

```bash
ros2 run tf2_ros tf2_echo fr3_link0 haptic_ee_1
ros2 run tf2_ros tf2_echo fr3_link0 haptic_ee_2
```

---

## Troubleshooting

### Error 13 (Virtuose Init returned 13)

- Check that daemons are running: `ps aux | grep -i svc`
- Check `LD_LIBRARY_PATH`: `echo $LD_LIBRARY_PATH | grep haption`
- Try: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ros2_ws/src/haption_raptor_api/Dependencies/RaptorAPI/bin/Linux/glibc-2.35/`
- Check connectivity: `ping -c 3 192.168.9.16` and `ping -c 3 192.168.9.17`

### Error 98 EADDRINUSE (bind failed)

- The second local IP has not been added: `sudo ip addr add 192.168.9.102/24 dev eno1`

### Error 99 EADDRNOTAVAIL

- Same as error 98 — the second IP is not present on the interface

### Device not calibrated (x: 0.000 y: 0.000 z: 0.000)

- Run the calibration scripts in the device folder

### Emergency stop active

- Check status: `ros2 topic echo /left/out_virtuose_status --once`
- Physically restart the device
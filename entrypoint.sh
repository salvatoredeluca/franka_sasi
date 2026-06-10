#!/bin/bash
source ~/.bashrc
export PYTHONPATH="/opt/openrobots/lib/python3.10/site-packages:${PYTHONPATH}"
/ros2_ws/src/6DOFs_haptic_devices/n71/7.2.0000/SvcHaptic_Desktop_6D_n71 &
/ros2_ws/src/6DOFs_haptic_devices/n70/7.2.0000/SvcHaptic_Desktop_6D_n70 &
sudo ip addr add 192.168.9.102/24 dev eno1 2>/dev/null || true
echo "[entrypoint] Environment ready."
exec bash

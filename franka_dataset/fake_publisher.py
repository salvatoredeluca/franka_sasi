#!/usr/bin/env python3
"""
fake_publisher.py

Publishes synthetic data on all topics consumed by record_lerobot_dataset.py
so you can verify dataset recording without real hardware.

Topics published:
  /left/ee_pose              geometry_msgs/PoseStamped   (100 Hz)
  /left/ee_twist_command     geometry_msgs/TwistStamped  (100 Hz)
  /left/gripper_status       std_msgs/Bool               (10 Hz, toggles)
  /right/ee_pose             geometry_msgs/PoseStamped   (100 Hz)
  /right/ee_twist_command    geometry_msgs/TwistStamped  (100 Hz)
  /right/gripper_status      std_msgs/Bool               (10 Hz, toggles)
  /camera_left/color/image_raw   sensor_msgs/Image       (30 Hz, red tint)
  /camera_right/color/image_raw  sensor_msgs/Image       (30 Hz, blue tint)

Poses and twists follow sine/cosine trajectories so each frame is distinct
and easy to spot regressions (constant data would mask frame-copy bugs).

Usage:
  python3 fake_publisher.py
  python3 fake_publisher.py --fps 30 --image_size 224
"""

import argparse
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Header


class FakePublisher(Node):

    def __init__(self, fps: int, image_size: int):
        super().__init__("fake_publisher")
        self.fps        = fps
        self.image_size = image_size
        self._t0        = time.time()

        qos = rclpy.qos.QoSProfile(depth=10)

        # ── publishers ────────────────────────────────────────────────────────
        self._pub = {
            "left_pose":    self.create_publisher(PoseStamped,  "/left/ee_pose",            qos),
            "left_twist":   self.create_publisher(TwistStamped, "/left/ee_twist_command",   qos),
            "left_grip":    self.create_publisher(Bool,         "/left/gripper_status",     qos),
            "right_pose":   self.create_publisher(PoseStamped,  "/right/ee_pose",           qos),
            "right_twist":  self.create_publisher(TwistStamped, "/right/ee_twist_command",  qos),
            "right_grip":   self.create_publisher(Bool,         "/right/gripper_status",    qos),
            "cam_left":     self.create_publisher(Image,        "/camera_left/color/image_raw",  qos),
            "cam_right":    self.create_publisher(Image,        "/camera_right/color/image_raw", qos),
        }

        # ── timers ────────────────────────────────────────────────────────────
        self.create_timer(1.0 / 100,  self._pub_state)    # pose + twist at 100 Hz
        self.create_timer(1.0 / fps,  self._pub_cameras)  # cameras at dataset fps
        self.create_timer(1.0 / 10,   self._pub_grippers) # gripper toggle at 10 Hz

        self._gripper_left  = True   # start open
        self._gripper_right = True
        self._grip_counter  = 0      # toggle every 20 calls (~2 s at 10 Hz)

        self.get_logger().info(
            f"FakePublisher started — cameras at {fps} Hz, "
            f"image size {image_size}x{image_size}.")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _now(self):
        return self.get_clock().now().to_msg()

    def _elapsed(self) -> float:
        return time.time() - self._t0

    def _make_pose(self, offset: float) -> PoseStamped:
        t   = self._elapsed() + offset
        msg = PoseStamped()
        msg.header = Header(stamp=self._now(), frame_id="world")
        p = msg.pose.position
        p.x = 0.4 + 0.05 * math.sin(t * 0.5)
        p.y = offset * 0.3 + 0.03 * math.cos(t * 0.7)
        p.z = 0.3 + 0.02 * math.sin(t * 1.1)
        # unit quaternion rotating slowly around z
        angle       = t * 0.2
        o           = msg.pose.orientation
        o.x         = 0.0
        o.y         = 0.0
        o.z         = math.sin(angle / 2)
        o.w         = math.cos(angle / 2)
        return msg

    def _make_twist(self, offset: float) -> TwistStamped:
        t   = self._elapsed() + offset
        msg = TwistStamped()
        msg.header = Header(stamp=self._now(), frame_id="world")
        l = msg.twist.linear
        l.x = 0.01 * math.sin(t * 1.0)
        l.y = 0.01 * math.cos(t * 1.3)
        l.z = 0.005 * math.sin(t * 0.8)
        a = msg.twist.angular
        a.x = 0.005 * math.cos(t * 0.6)
        a.y = 0.005 * math.sin(t * 0.9)
        a.z = 0.002 * math.cos(t * 1.2)
        return msg

    def _make_image(self, tint: tuple[int, int, int]) -> Image:
        t   = self._elapsed()
        H   = W = self.image_size
        # base: slow animated gradient so consecutive frames are distinct
        xx, yy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
        wave   = (0.5 + 0.5 * np.sin(xx * 6.0 + t * 2.0)) * 200
        data   = np.zeros((H, W, 3), dtype=np.uint8)
        data[..., 0] = (wave * tint[0] / 255).astype(np.uint8)
        data[..., 1] = (wave * tint[1] / 255).astype(np.uint8)
        data[..., 2] = (wave * tint[2] / 255).astype(np.uint8)

        msg              = Image()
        msg.header       = Header(stamp=self._now(), frame_id="camera")
        msg.height       = H
        msg.width        = W
        msg.encoding     = "rgb8"
        msg.is_bigendian = False
        msg.step         = W * 3
        msg.data         = data.tobytes()
        return msg

    # ── timer callbacks ───────────────────────────────────────────────────────

    def _pub_state(self):
        self._pub["left_pose"].publish(self._make_pose(offset=0.0))
        self._pub["left_twist"].publish(self._make_twist(offset=0.0))
        self._pub["right_pose"].publish(self._make_pose(offset=1.0))
        self._pub["right_twist"].publish(self._make_twist(offset=1.0))

    def _pub_cameras(self):
        # left camera: red tint; right camera: blue tint — easy to tell apart
        self._pub["cam_left"].publish(self._make_image((255, 80, 80)))
        self._pub["cam_right"].publish(self._make_image((80, 80, 255)))

    def _pub_grippers(self):
        self._grip_counter += 1
        if self._grip_counter % 20 == 0:   # toggle every ~2 s
            self._gripper_left  = not self._gripper_left
        if self._grip_counter % 30 == 0:   # different period for right arm
            self._gripper_right = not self._gripper_right

        left_msg  = Bool(data=self._gripper_left)
        right_msg = Bool(data=self._gripper_right)
        self._pub["left_grip"].publish(left_msg)
        self._pub["right_grip"].publish(right_msg)


def main():
    parser = argparse.ArgumentParser(
        description="Fake ROS2 publisher for record_lerobot_dataset.py testing.")
    parser.add_argument("--fps",        type=int, default=30,
                        help="Camera publish rate (should match recorder --fps).")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Image width and height in pixels.")
    args = parser.parse_args()

    rclpy.init()
    node     = FakePublisher(fps=args.fps, image_size=args.image_size)
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        print("Publishing fake data. Press Ctrl+C to stop.")
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
record_lerobot_dataset.py

ROS2 Humble node that records a dual Franka FR3 teleoperation setup
into a LeRobot dataset for fine-tuning with openpi (pi0 / pi0-FAST).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SYNCHRONIZATION STRATEGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

openpi builds action chunks using:
    delta_timestamps = [t / dataset.fps for t in range(action_horizon)]

This requires every frame to be temporally equidistant at exactly
1/fps seconds apart. Irregular spacing causes index lookup errors
during training.

Therefore the architecture is:

  CAMERA CALLBACKS (async, at camera publish rate)
  ├── _cb_cam_left  → decode image + snapshot left.pose  atomically
  └── _cb_cam_right → decode image + snapshot right.pose atomically

    Pose is snapshotted INSIDE the camera callback so that each image
    is paired with the arm pose read at the same instant.
    Temporal coherence: img_left <-> left.pose  guaranteed.
                        img_right <-> right.pose guaranteed.

  TIMER @ 1/fps Hz  (controls dataset frequency)
  └── _capture_frame → reads camera snapshots + latest twist + gripper
                       appends one frame with equidistant timestamps

  LATEST-VALUE CALLBACKS (async)
  ├── twist  → updated continuously by haptic device
  │            valid even when operator is still (last command holds)
  └── gripper Bool → event-driven, latest value is always current state

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET LAYOUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

observation.state (16,):
  [0..6]  left  EE pose (x,y,z,qx,qy,qz,qw) <- snapshot in cam_left  cb
  [7]     left  gripper_state {0.0, 1.0}      <- latest Bool value
  [8..14] right EE pose                       <- snapshot in cam_right cb
  [15]    right gripper_state                 <- latest Bool value

action (14,):
  [0..5]  left  EE twist (dx,dy,dz,dwx,dwy,dwz) <- latest haptic value
  [6]     left  gripper_cmd  <- next-state trick applied in stop_episode
  [7..12] right EE twist                         <- latest haptic value
  [13]    right gripper_cmd  <- next-state trick applied in stop_episode

  next-state trick (applied offline before saving to disk):
    action[t].gripper_cmd = state[t+1].gripper_state
    last frame: gripper_cmd = current gripper_state (hold)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOPICS SUBSCRIBED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  /left/ee_pose              geometry_msgs/PoseStamped
  /left/ee_twist_command     geometry_msgs/TwistStamped  (haptic device)
  /left/gripper_status       std_msgs/Bool  (True=open, False=closed)

  /right/ee_pose             geometry_msgs/PoseStamped
  /right/ee_twist_command    geometry_msgs/TwistStamped
  /right/gripper_status      std_msgs/Bool

  /camera_left/color/image_raw   sensor_msgs/Image
  /camera_right/color/image_raw  sensor_msgs/Image

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  python3 record_lerobot_dataset.py \
      --repo_id    my_org/dual-fr3-dataset \
      --output_dir ./lerobot_data \
      --fps        30 \
      --image_size 224

Controls:
  Enter      ->  start / stop episode
  q + Enter  ->  quit and save dataset
"""

import argparse
import shutil         
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from cv_bridge import CvBridge
from PIL import Image as PILImage

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ── Gripper binary encoding ───────────────────────────────────────────────────
# Source topic is std_msgs/Bool — no physical width available.
# {0.0, 1.0} is consistent with ACT / LeRobot public dataset conventions.
GRIPPER_OPEN  = 1.0   # Bool True  -> open
GRIPPER_CLOSE = 0.0   # Bool False -> closed

# ── Dataset dimensions ────────────────────────────────────────────────────────
#   state:  pose(7) + gripper_state(1) = 8 per arm   
#   action: twist(6) + gripper_cmd(1)  = 7 per arm  
STATE_DIM  = 16
ACTION_DIM = 14

# observation.state indices — gripper_state
IDX_LEFT_GRIPPER_STATE  = 7
IDX_RIGHT_GRIPPER_STATE = 15

# action indices — gripper_cmd (placeholder 0.0, filled in stop_episode)
IDX_LEFT_GRIPPER_CMD  = 6
IDX_RIGHT_GRIPPER_CMD = 13


# ── Feature schema ────────────────────────────────────────────────────────────
def build_features(image_size: int) -> dict:
    H = W = image_size
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": [
                # left arm (8)
                "left_x",  "left_y",  "left_z",
                "left_qx", "left_qy", "left_qz", "left_qw",
                "left_gripper_state",
                # right arm (8)
                "right_x",  "right_y",  "right_z",
                "right_qx", "right_qy", "right_qz", "right_qw",
                "right_gripper_state",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": [
                # left arm (7)
                "left_dx",  "left_dy",  "left_dz",
                "left_dwx", "left_dwy", "left_dwz",
                "left_gripper_cmd",
                # right arm (7)
                "right_dx",  "right_dy",  "right_dz",
                "right_dwx", "right_dwy", "right_dwz",
                "right_gripper_cmd",
            ],
        },
        "observation.images.camera_left": {
            "dtype": "video",
            "shape": (3, H, W),
            "names": ["channel", "height", "width"],
        },
        "observation.images.camera_right": {
            "dtype": "video",
            "shape": (3, H, W),
            "names": ["channel", "height", "width"],
        },
    }


# ── Camera snapshot ───────────────────────────────────────────────────────────
@dataclass
class CameraSnapshot:
    """
    Stores image and arm pose captured atomically inside a camera callback.

    Because pose is read at the same instant the image arrives, each
    snapshot guarantees temporal coherence between image and pose for
    one arm. The timer then reads the latest snapshot at fixed intervals.
    """
    image:         Optional[PILImage.Image] = None
    pose:          Optional[np.ndarray]     = None   # (7,) copy at cb time
    gripper_state: Optional[float]          = None   # latest Bool at cb time

    def ready(self) -> bool:
        return (self.image is not None and
                self.pose  is not None and
                self.gripper_state is not None)


# ── Per-arm latest-value cache ────────────────────────────────────────────────
@dataclass
class ArmBuffer:
    """
    Holds the most recent value for twist and gripper.
    Updated asynchronously by topic callbacks.

    twist:         last haptic command — valid even when operator is still
    gripper_state: last Bool received  — correct by definition (event-driven)
    """
    twist:         Optional[np.ndarray] = None   # (6,) [dx,dy,dz,dwx,dwy,dwz]
    gripper_state: Optional[float]      = None   # 1.0=open / 0.0=closed

    def ready(self) -> bool:
        return (self.twist         is not None and
                self.gripper_state is not None)

    def action_vec(self) -> np.ndarray:
        """(7,) -> [twist(6) | gripper_cmd_placeholder(1)]
        gripper_cmd is 0.0 here — filled by next-state trick in stop_episode.
        """
        return np.concatenate(
            [self.twist, [0.0]]
        ).astype(np.float32)


# ── ROS2 recorder node ────────────────────────────────────────────────────────
class DualFR3Recorder(Node):

    def __init__(self, dataset: LeRobotDataset, fps: int, image_size: int):
        super().__init__("dual_fr3_lerobot_recorder")

        self.dataset    = dataset
        self.fps        = fps
        self.image_size = image_size
        self.bridge     = CvBridge()

        self._lock     = threading.Lock()
        self.recording = False
        self.frames_buf: list[dict] = []

        # ── camera snapshots — updated in camera callbacks ────────────────────
        # Each snapshot holds the image AND the arm pose read at that instant.
        # This guarantees temporal coherence between image and pose per arm.
        self.snap_left  = CameraSnapshot()
        self.snap_right = CameraSnapshot()

        # ── arm buffers — twist and gripper latest values ─────────────────────
        # gripper_state defaults to OPEN: the Bool topic is event-driven and
        # may never publish if the gripper state never changes at startup.
        self.left  = ArmBuffer(gripper_state=GRIPPER_OPEN)
        self.right = ArmBuffer(gripper_state=GRIPPER_OPEN)

        # ── QoS ──────────────────────────────────────────────────────────────
        # depth=1  for state/gripper: only the latest message matters
        # depth=2  for cameras: small buffer to absorb brief publish bursts
        qos_state  = rclpy.qos.qos_profile_sensor_data
        qos_camera = rclpy.qos.qos_profile_sensor_data

        # ── left arm ──────────────────────────────────────────────────────────
        # pose is read inside the camera callback (see _cb_cam_left)
        # so we still subscribe to get the latest pose into a buffer
        self.create_subscription(
            PoseStamped,  "/left/ee_pose",
            self._cb_pose_left,    qos_state)
        self.create_subscription(
            Twist, "/left/ee_twist_command",
            self._cb_twist_left,   qos_state)
        self.create_subscription(
            Bool,         "/left/gripper_status",
            self._cb_gripper_left, qos_state)

        # ── right arm ─────────────────────────────────────────────────────────
        self.create_subscription(
            PoseStamped,  "/right/ee_pose",
            self._cb_pose_right,    qos_state)
        self.create_subscription(
            Twist, "/right/ee_twist_command",
            self._cb_twist_right,   qos_state)
        self.create_subscription(
            Bool,         "/right/gripper_status",
            self._cb_gripper_right, qos_state)

        # ── cameras — update snapshots asynchronously ─────────────────────────
        self.create_subscription(
            Image, "/left/camera_left/color/image_raw",
            self._cb_cam_left,  qos_camera)
        self.create_subscription(
            Image, "/right/camera_right/color/image_raw",
            self._cb_cam_right, qos_camera)

        # ── timer — controls dataset frequency ───────────────────────────────
        # Fires at exactly 1/fps Hz and reads the latest snapshots.
        # This guarantees equidistant frame timestamps required by openpi:
        #   delta_timestamps = [t / fps for t in range(action_horizon)]
        self.create_timer(1.0 / fps, self._capture_frame)

        self.get_logger().info(
            f"DualFR3Recorder ready — recording at {fps} Hz.")

    # ── static helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _bool_to_gripper(is_open: bool) -> float:
        """Bool -> {1.0=open, 0.0=closed}."""
        return GRIPPER_OPEN if is_open else GRIPPER_CLOSE

    def _ros_img_to_pil(self, msg: Image) -> PILImage.Image:
        """Decode ROS Image -> resized PIL. Always called OUTSIDE the lock."""
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        pil    = PILImage.fromarray(cv_img)
        if pil.size != (self.image_size, self.image_size):
            pil = pil.resize(
                (self.image_size, self.image_size), PILImage.LANCZOS)
        return pil

    # ── pose callbacks ────────────────────────────────────────────────────────
    # Pose is stored in a dedicated field so camera callbacks can copy
    # it atomically at image-arrival time.
    def _cb_pose_left(self, msg: PoseStamped):
        p, o = msg.pose.position, msg.pose.orientation
        with self._lock:
            self._latest_pose_left = np.array(
                [p.x, p.y, p.z, o.x, o.y, o.z, o.w], dtype=np.float32)

    def _cb_pose_right(self, msg: PoseStamped):
        p, o = msg.pose.position, msg.pose.orientation
        with self._lock:
            self._latest_pose_right = np.array(
                [p.x, p.y, p.z, o.x, o.y, o.z, o.w], dtype=np.float32)

    # initialise pose fields so camera callbacks can check for None
    _latest_pose_left:  Optional[np.ndarray] = None
    _latest_pose_right: Optional[np.ndarray] = None

    # ── twist callbacks (latest value — haptic device) ────────────────────────
    def _cb_twist_left(self, msg: Twist):
        l, a = msg.linear, msg.angular
        with self._lock:
            self.left.twist = np.array(
                [l.x, l.y, l.z, a.x, a.y, a.z], dtype=np.float32)

    def _cb_twist_right(self, msg: Twist):
        l, a = msg.linear, msg.angular
        with self._lock:
            self.right.twist = np.array(
                [l.x, l.y, l.z, a.x, a.y, a.z], dtype=np.float32)

    # ── gripper callbacks (latest value — Bool event-driven) ──────────────────
    def _cb_gripper_left(self, msg: Bool):
        with self._lock:
            self.left.gripper_state = self._bool_to_gripper(msg.data)

    def _cb_gripper_right(self, msg: Bool):
        with self._lock:
            self.right.gripper_state = self._bool_to_gripper(msg.data)

    # ── camera callbacks — update snapshots ───────────────────────────────────
    def _cb_cam_left(self, msg: Image):
        """
        1. Decode and resize image OUTSIDE the lock.
        2. Inside the lock: copy image + left pose atomically.
           -> img_left and snap_left.pose are temporally coherent.
        """
        pil = self._ros_img_to_pil(msg)   # outside lock — slow operation

        with self._lock:
            self.snap_left.image = pil
            self.snap_left.pose  = (
                self._latest_pose_left.copy()
                if self._latest_pose_left is not None else None)
            self.snap_left.gripper_state = self.left.gripper_state

    def _cb_cam_right(self, msg: Image):
        """
        Mirror of _cb_cam_left for the right arm.
        """
        pil = self._ros_img_to_pil(msg)   # outside lock

        with self._lock:
            self.snap_right.image = pil
            self.snap_right.pose  = (
                self._latest_pose_right.copy()
                if self._latest_pose_right is not None else None)
            self.snap_right.gripper_state = self.right.gripper_state

    # ── timer callback — frame capture at fixed fps ───────────────────────────
    def _capture_frame(self):
        """
        Called by ROS2 timer at exactly 1/fps Hz.

        Reads the latest camera snapshots and arm buffers and appends
        one frame to frames_buf.

        Temporal properties of the saved frame:
          img_left   <-> left.pose   : coherent — snapshotted together
                                       in _cb_cam_left
          img_right  <-> right.pose  : coherent — snapshotted together
                                       in _cb_cam_right
          twist_left / twist_right   : latest haptic command — valid
                                       even when operator is still
          gripper_left/right         : latest Bool — correct by definition

        Frame timestamps are equidistant at 1/fps seconds, satisfying
        the openpi requirement:
          delta_timestamps = [t / fps for t in range(action_horizon)]
        """
        with self._lock:
            if not self.recording:
                return

            # wait until camera snapshots and arm buffers are populated
            if not (self.snap_left.ready() and
                    self.snap_right.ready() and
                    self.left.ready() and
                    self.right.ready()):
                missing = []
                if not self.snap_left.ready():
                    missing.append(
                        f"snap_left(img={self.snap_left.image is not None},"
                        f" pose={self.snap_left.pose is not None},"
                        f" grip={self.snap_left.gripper_state is not None})")
                if not self.snap_right.ready():
                    missing.append(
                        f"snap_right(img={self.snap_right.image is not None},"
                        f" pose={self.snap_right.pose is not None},"
                        f" grip={self.snap_right.gripper_state is not None})")
                if not self.left.ready():
                    missing.append(
                        f"left(twist={self.left.twist is not None},"
                        f" grip={self.left.gripper_state is not None})")
                if not self.right.ready():
                    missing.append(
                        f"right(twist={self.right.twist is not None},"
                        f" grip={self.right.gripper_state is not None})")
                self.get_logger().warn(
                    f"Skipping frame — not ready: {', '.join(missing)}",
                    throttle_duration_sec=2.0)
                return

            # ── observation.state (16,) ───────────────────────────────────────
            # left  pose @ cam_left  callback time -> coherent with img_left
            # right pose @ cam_right callback time -> coherent with img_right
            # gripper_state: latest Bool value for each arm
            state = np.array([
                *self.snap_left.pose,  self.snap_left.gripper_state,    # (8,)
                *self.snap_right.pose, self.snap_right.gripper_state,   # (8,)
            ], dtype=np.float32)   # (16,)

            # ── action (14,) ──────────────────────────────────────────────────
            # twist: latest haptic command (valid when operator is still)
            # gripper_cmd: placeholder 0.0, filled by next-state trick
            #              in stop_episode() before writing to disk
            action = np.concatenate(
                [self.left.action_vec(),    # (7,)
                 self.right.action_vec()]   # (7,)
            )   # (14,)

            self.frames_buf.append({
                "observation.state":               state,
                "action":                          action,
                "observation.images.camera_left":  self.snap_left.image,
                "observation.images.camera_right": self.snap_right.image,
            })

    # ── episode control ───────────────────────────────────────────────────────
    def start_episode(self):
        with self._lock:
            self.frames_buf.clear()
            self.recording = True
        self.get_logger().info("Episode started.")

    def stop_episode(self, task: str = "manipulation task") -> int:
        with self._lock:
            self.recording = False
            buf = list(self.frames_buf)
            self.frames_buf.clear()

        if not buf:
            self.get_logger().warn("No frames captured — episode discarded.")
            return 0

        # ── next-state gripper shift ──────────────────────────────────────────
        #
        # The Bool topic gives the observed gripper state, not a command.
        # Standard practice in imitation learning (ACT, Diffusion Policy, pi0)
        # defines the gripper action as the state the gripper transitions INTO:
        #
        #   action[t].gripper_cmd = state[t+1].gripper_state
        #
        # Last frame: gripper_cmd = current gripper_state (hold — no future).
        #
        # Indices:
        #   state  [7]  = left_gripper_state  ->  action  [6] = left_gripper_cmd
        #   state  [15] = right_gripper_state ->  action [13] = right_gripper_cmd
        # ─────────────────────────────────────────────────────────────────────
        for i, frame in enumerate(buf):
            next_state = (
                buf[i + 1]["observation.state"]
                if i < len(buf) - 1
                else frame["observation.state"]   # last frame: hold
            )
            action = frame["action"].copy()
            action[IDX_LEFT_GRIPPER_CMD]  = next_state[IDX_LEFT_GRIPPER_STATE]
            action[IDX_RIGHT_GRIPPER_CMD] = next_state[IDX_RIGHT_GRIPPER_STATE]
            frame["action"] = action

        self.get_logger().info(
            f"Episode stopped — {len(buf)} frames -> writing to dataset...")

        for frame in buf:
            self.dataset.add_frame(frame, task=task)

        self.dataset.save_episode()
        self.get_logger().info("Episode saved.")
        return len(buf)


# ── keyboard control (main thread) ───────────────────────────────────────────
def keyboard_loop(recorder: DualFR3Recorder,
                  shutdown_event: threading.Event) -> None:
    print("\n" + "-" * 60)
    print("  Dual FR3 LeRobot Recorder")
    print("  Enter      ->  start / stop an episode")
    print("  q + Enter  ->  quit and save the dataset")
    print("-" * 60 + "\n")

    episode_active = False
    ep_count       = 0

    while not shutdown_event.is_set():
        try:
            key = input()
        except EOFError:
            break

        if key.strip().lower() == "q":
            print("Quitting...")
            shutdown_event.set()
            break

        if not episode_active:
            recorder.start_episode()
            episode_active = True
            ep_count += 1
            print(f"Episode {ep_count} recording...  (press Enter to stop)")
        else:
            task = input(
                "Task description (leave blank for default): ").strip()
            if not task:
                task = "Grasp the bottle with the right arm and uncork it with the left one," \
                " then fill the cup with water using the right arm"
            # strip lone surrogates caused by non-UTF-8 terminals
            task = task.encode("utf-8", errors="replace").decode("utf-8")
            n = recorder.stop_episode(task=task)
            print(f"Episode {ep_count} saved — {n} frames.")
            episode_active = False
            print("\nPress Enter to start a new episode  |  q + Enter to quit")


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Record dual Franka FR3 teleoperation demos "
            "as a LeRobot dataset via ROS2 Humble."
        )
    )
    parser.add_argument(
        "--repo_id",     type=str,
        default="local/dual-fr3-dataset",
        help="HuggingFace repo ID used as dataset name.")
    parser.add_argument(
        "--output_dir",  type=Path,
        default=Path("./lerobot_data"),
        help="Local directory to write the dataset.")
    parser.add_argument(
        "--fps",         type=int, default=30,
        help=(
            "Recording frequency in Hz. "
            "Must match the fps used in the openpi TrainConfig "
            "(delta_timestamps = [t/fps for t in range(action_horizon)])."))
    parser.add_argument(
        "--image_size",  type=int, default=224,
        help="Images resized to image_size x image_size pixels.")
    args = parser.parse_args()

    # ── create LeRobot dataset ────────────────────────────────────────────────
    features = build_features(args.image_size)
 
    meta_ok = (args.output_dir / "meta" / "tasks.jsonl").exists()

    if args.output_dir.exists() and not meta_ok:
        shutil.rmtree(args.output_dir)
        print(f"Removed incomplete dataset directory at {args.output_dir}.")

    if args.output_dir.exists() and meta_ok:
        # count existing episodes from parquet files
        existing = sorted(
            (args.output_dir / "data").rglob("episode_*.parquet"))
        n_existing = len(existing)
 
        print(f"\nExisting dataset found at {args.output_dir}")
        print(f"  Existing episodes : {n_existing}")
        print(f"  [a] append new episodes to existing dataset")
        print(f"  [o] overwrite — delete everything and start fresh")
        print(f"  [q] quit")
 
        while True:
            choice = input("Choice: ").strip().lower()
            if choice in ("a", "o", "q"):
                break
            print("  Invalid choice — enter a, o or q")
 
        if choice == "q":
            print("Exiting.")
            return
 
        elif choice == "o":
            shutil.rmtree(args.output_dir)
            print("Previous dataset removed.")
            dataset = LeRobotDataset.create(
                repo_id=args.repo_id,
                fps=args.fps,
                features=features,
                root=args.output_dir,
                image_writer_threads=4,
            )
            print(f"New dataset created at {args.output_dir}")
 
        else:  # choice == "a"
            dataset = LeRobotDataset(
                repo_id=args.repo_id,
                root=args.output_dir,
                local_files_only=True,
            )
            print(f"Dataset opened — {dataset.num_episodes} existing episode(s).")
 
    else:
        # first run — create a fresh dataset
        print(f"Creating new dataset at {args.output_dir} ...")
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            features=features,
            root=args.output_dir,
            image_writer_threads=4,
        )

    # ── spin ROS2 ─────────────────────────────────────────────────────────────
    rclpy.init()
    recorder = DualFR3Recorder(
        dataset, fps=args.fps, image_size=args.image_size)

    executor       = MultiThreadedExecutor()
    executor.add_node(recorder)
    shutdown_event = threading.Event()

    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    try:
        keyboard_loop(recorder, shutdown_event)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down ROS2...")
        executor.shutdown()
        recorder.destroy_node()
        rclpy.shutdown()
        ros_thread.join(timeout=5.0)
        print(f"\nDataset saved — {dataset.num_episodes} episode(s) at {args.output_dir}")

    if dataset.num_episodes > 0 and not args.repo_id.startswith("local/"):
        print("Pushing dataset to HuggingFace Hub...")
        try:
            dataset.push_to_hub()
            print(f"Done. View at https://huggingface.co/datasets/{args.repo_id}")
        except Exception as e:
            print(f"Push failed: {e}")
            print("Data is safe locally. Run 'huggingface-cli login' then push manually.")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
// Copyright (c) 2023 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <string>

#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/bool.hpp>

#include <std_srvs/srv/set_bool.hpp>
#include <mutex>
#include <atomic>
#include <franka_msgs/action/grasp.hpp>
#include <franka_msgs/action/move.hpp>
#include <franka_msgs/srv/set_full_collision_behavior.hpp>

#include <franka_example_controllers/fr3/dynamics.hpp>
#include <tf2_ros/transform_broadcaster.h>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_example_controllers {

/**
 * Joint impedance controller with full dynamics via Pinocchio.
 *
 * Uses the Dynamics class for gravity, mass matrix, Coriolis, and
 * Jacobian computations.  When "gazebo" is true the gravity torque is
 * added to the commanded effort; on real Franka hardware it is omitted
 * (the firmware compensates internally).
 */
class CustomController : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Vector3d = Eigen::Matrix<double, 3, 1>;
  using Matrix6x7d = Eigen::Matrix<double, 6, 7>;
  using Matrix7x6d = Eigen::Matrix<double, 7, 6>;
  using Matrix6x6d = Eigen::Matrix<double, 6, 6>;
  using Matrix7x7d = Eigen::Matrix<double, 7, 7>;
  using Matrix3x3d = Eigen::Matrix<double, 3, 3>;
  using Matrix4x4d = Eigen::Matrix<double, 4, 4>;




  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

 protected:
  /// Full dynamics object — available to subclasses.
  std::unique_ptr<Dynamics> dynamics_;

 private:
  std::string robot_type_;
  std::string arm_prefix_;
  std::string robot_description_;

  const int num_joints = 7;
  Vector7d q_;
  Vector7d dq_;
  Vector7d dq_filtered_;
  Vector3d p_d_; //Desired ee position
  Eigen::Quaterniond q_d_quat_; //Desired ee orientation
  Vector3d p_d_dot_; //Desired ee linear velocity
  Vector3d omega_d_; //Desired ee angular velocity
  Vector6d k_gains_;
  Vector6d d_gains_;
  Vector7d d_null_gains_;
  double null_stiffness_gain_;
  double null_sigma_gain_;

    // --- Twist command ---
  geometry_msgs::msg::Twist twist_cmd_;
  std::mutex twist_mutex_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_sub_;
  std::chrono::steady_clock::time_point last_twist_time_;
  double twist_timeout_{0.5};  // seconds
  bool first_twist_received_ = false;

  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  //Error
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr pose_error_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_error_pub_;

  //Pose publisher
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr ee_pose_pub_;



  // --- Gripper ---
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr gripper_service_;  std::shared_ptr<rclcpp_action::Client<franka_msgs::action::Grasp>> gripper_grasp_client_;
  std::shared_ptr<rclcpp_action::Client<franka_msgs::action::Move>> gripper_move_client_;
  std::atomic<bool> gripper_is_closed_{false};
  std::atomic<bool> gripper_action_in_progress_{false};

  // --- Collision behavior thresholds ---
  std::vector<double> lower_torque_thresholds_accel_;
  std::vector<double> upper_torque_thresholds_accel_;
  std::vector<double> lower_torque_thresholds_nominal_;
  std::vector<double> upper_torque_thresholds_nominal_;
  std::vector<double> lower_force_thresholds_accel_;
  std::vector<double> upper_force_thresholds_accel_;
  std::vector<double> lower_force_thresholds_nominal_;
  std::vector<double> upper_force_thresholds_nominal_;

  rclcpp_action::Client<franka_msgs::action::Move>::SendGoalOptions move_goal_options_;
  rclcpp_action::Client<franka_msgs::action::Grasp>::SendGoalOptions grasp_goal_options_;

  // Gripper parameters
  double grasp_width_;
  double grasp_speed_;
  double grasp_force_;
  double grasp_epsilon_inner_;
  double grasp_epsilon_outer_;
  double open_width_;
  double open_speed_;

  // Gripper status publisher
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr gripper_status_pub_;



  void twistCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
void gripperServiceCallback(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                               std::shared_ptr<std_srvs::srv::SetBool::Response> response);  void openGripper();
  void graspGripper();
  void assignMoveGoalOptionsCallbacks();
  void assignGraspGoalOptionsCallbacks();

// --- Safety ---
  double sigma_min_threshold_;    // soglia per regolarizzazione e forza repulsiva
  double sigma_min_critical_;     // soglia per blocco twist
  double lambda_max_;             // max damping per regolarizzazione (Nakamura-Hanafusa 1986)
  double manip_task_gain_;        // gain forza repulsiva nel task space (Dimeas 2016)
  double manip_task_max_accel_;   // saturazione forza repulsiva
  double null_joint_limit_gain_;
  bool twist_blocked_ = false;

  // Cached per-cycle SVD results (computed once in update, reused in impedance_ctrl)
  double sigma_min_{1.0};
  Vector6d u_min_cached_;
  Eigen::Matrix<double, 7, 1> v_min_cached_;  // right singular vector
  Vector7d sigma_min_grad_cached_;             // ∂σ_min/∂q
  Matrix6x7d J_cached_;

  // FR3 joint limits (rad) - from URDF
  static constexpr double q_min_[7] = {-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159};
  static constexpr double q_max_[7] = { 2.7437,  1.7837,  2.9007, -0.1518,  2.8065,  4.5169,  3.0159};
  static constexpr double joint_safety_margin_ = 0.0872665;  // ~5 degrees
  static constexpr double joint_limit_delta_ = 0.01;   // numerical safety margin
// FR3 home position (high manipulability configuration)
  static constexpr double q_home_[7] = {0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785};


  void updateJointStates();

  Vector7d impedance_ctrl();
};

}  // namespace franka_example_controllers

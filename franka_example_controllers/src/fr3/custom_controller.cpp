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

#include <franka_example_controllers/fr3/custom_controller.hpp>

#include <cassert>
#include <cmath>
#include <exception>
#include <string>

#include <Eigen/Eigen>

namespace franka_example_controllers {

controller_interface::InterfaceConfiguration
CustomController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_prefix_ + robot_type_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
CustomController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_prefix_ + robot_type_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_prefix_ + robot_type_ + "_joint" + std::to_string(i) + "/velocity");
  }
  return config;
}

controller_interface::return_type CustomController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& period) {
  updateJointStates();

  double dt = period.seconds();

  const double kAlpha = 0.99;
  dq_filtered_ = (1 - kAlpha) * dq_filtered_ + kAlpha * dq_;

  // --- Read twist command and check timeout ---
  geometry_msgs::msg::Twist twist_cmd;
  {
    std::lock_guard<std::mutex> lock(twist_mutex_);
    twist_cmd = twist_cmd_;
  }

  // --- Update desired velocities ---
  p_d_dot_ << twist_cmd.linear.x, twist_cmd.linear.y, twist_cmd.linear.z;
  omega_d_ << twist_cmd.angular.x, twist_cmd.angular.y, twist_cmd.angular.z;

// --- Compute Jacobian, SVD, and sigmaMin gradient (once per cycle) ---
  J_cached_ = dynamics_->worldJacobian(q_);
  Eigen::JacobiSVD<Matrix6x7d> svd(J_cached_, Eigen::ComputeFullU | Eigen::ComputeFullV);
  sigma_min_ = svd.singularValues()(5);
  u_min_cached_ = svd.matrixU().col(5);
  v_min_cached_ = svd.matrixV().col(5);
  sigma_min_grad_cached_ = dynamics_->sigmaMinGradient(q_, J_cached_, u_min_cached_, v_min_cached_);

  if(!twist_blocked_){
    do {
      // --- Safety: block controller if sigma_min below critical threshold ---
      if (sigma_min_ < sigma_min_critical_) {
        RCLCPP_ERROR_ONCE(get_node()->get_logger(),
            "sigma_min=%.4f below critical %.4f - blocking controller - torques set to gravity compensation.", sigma_min_, sigma_min_critical_);
        twist_blocked_ = true;
        break;
      }
      // --- Safety: block controller if one joint violates the limits buffer (more conservative of about 5 degrees) ---
      for (int i = 0; i < num_joints; ++i) {
        double dist_to_min = q_(i) - q_min_[i];
        double dist_to_max = q_max_[i] - q_(i);
        if (dist_to_min < joint_safety_margin_ || dist_to_max < joint_safety_margin_) {
            RCLCPP_ERROR_ONCE(get_node()->get_logger(),
                "Joint %d near limit - blocking controller.", i+1);
            twist_blocked_ = true;
            break;
        }
      }
      if (twist_blocked_) break;

      // --- Safety: block controller if timeout is violated
      if (first_twist_received_) {
        double diff = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - last_twist_time_).count();
        if (diff > twist_timeout_) {
          RCLCPP_ERROR_ONCE(get_node()->get_logger(),
              "Watchdog: no twist for %.4f s - blocking controller.", diff);
          twist_blocked_ = true;
          break;
        }
      }
    } while(false);
  }

if (twist_blocked_) {
    for (int i = 0; i < num_joints; i++) {
        command_interfaces_[i].set_value(0);
    }
    exit(1); //Unclean safe exit
    //return controller_interface::return_type::ERROR;
  }


  // --- Integrate position ---
  p_d_ += p_d_dot_ * dt;

  // --- Integrate orientation (exponential map on SO(3)) ---
  // omega_d_ is in base frame. Convert to body frame of desired orientation.
  Vector3d omega_body = q_d_quat_.toRotationMatrix().transpose() * omega_d_;
  double angle = omega_body.norm() * dt;
  if (angle > 1e-10) {
    Vector3d axis = omega_body.normalized();
    Eigen::Quaterniond dq_rot(Eigen::AngleAxisd(angle, axis));
    q_d_quat_ = q_d_quat_ * dq_rot;
    q_d_quat_.normalize();
  }

  // --- Compute impedance torques ---
  Vector7d tau_cmd = impedance_ctrl();

  for (int i = 0; i < num_joints; ++i) {
    command_interfaces_[i].set_value(tau_cmd(i));
  }
  return controller_interface::return_type::OK;
}


CustomController::Vector7d CustomController::impedance_ctrl(){
//Note: compute J_cached_, its minimum singular value sigma_min_, and sigma_min_gradient before calling this function

  //Dynamics terms and Jacobians
  Matrix7x7d M = dynamics_->massMatrix(q_);
  Vector7d C = dynamics_->coriolis(q_, dq_filtered_);
  Matrix6x7d& J = J_cached_; //J, his minimum singular value, and the manipulability gradient have been already computed!! //#dynamics_->worldJacobian(q_);
  Vector6d J_dot_dq = dynamics_->worldBiasAcceleration(q_, dq_filtered_);
  
  
    // --- Adaptive regularization of Lambda (Nakamura-Hanafusa 1986) ---
  double lambda_sq = 1e-8;
  //sigma_min has been already computed
  if (sigma_min_ < sigma_min_threshold_) {
    double ratio = sigma_min_ / sigma_min_threshold_;
    lambda_sq += (1.0 - ratio * ratio) * lambda_max_ * lambda_max_;
  }

  Matrix6x6d JMinvJt = J * M.ldlt().solve(J.transpose());
  JMinvJt.diagonal().array() += lambda_sq;
  Matrix6x6d Lambda = JMinvJt.inverse();
  Matrix7x6d J_bar = M.ldlt().solve(J.transpose()) * Lambda;

  //End-effector twist
  Vector6d v_e = J*dq_filtered_;
  Vector3d p_e_dot = v_e.block<3,1>(0,0);
  Vector3d omega_e = v_e.block<3,1>(3,0);

  //End-effector pose
  Matrix4x4d T_e = dynamics_->forwardKinematics(q_);
  Vector3d p_e = T_e.block<3,1>(0,3);
  //Eigen::Quaterniond q_e(T_e.block<3,3>(0,0));

  //Orientation error
  // q_d_quat_.normalize();
  // q_e.normalize();
  // Matrix3x3d q_d_skew;
  // q_d_skew << 0, -q_d_quat_.z(), q_d_quat_.y(),
  //             q_d_quat_.z(), 0, -q_d_quat_.x(),
  //             -q_d_quat_.y(), q_d_quat_.x(), 0;
  // Vector3d e_o = q_d_quat_.w() * q_e.vec() - q_e.w()*q_d_quat_.vec() + q_d_skew * q_e.vec();

  // Orientation error (Ott 2008, eq. 3.18 - frame invariant)
  Matrix3x3d R_e = T_e.block<3, 3>(0, 0);
  Matrix3x3d R_d = q_d_quat_.toRotationMatrix();
  Vector3d e_o = 0.5 * (R_e.col(0).cross(R_d.col(0)) +
                         R_e.col(1).cross(R_d.col(1)) +
                         R_e.col(2).cross(R_d.col(2)));

  //Stiffness and damping
  Matrix3x3d K_p = k_gains_.block<3,1>(0,0).asDiagonal();
  Matrix3x3d K_o = k_gains_.block<3,1>(3,0).asDiagonal();
  Matrix3x3d D_p = d_gains_.block<3,1>(0,0).asDiagonal();
  Matrix3x3d D_o = d_gains_.block<3,1>(3,0).asDiagonal();

  //Reference accelerations (Nakanishi, 2008)
  Vector3d p_r_ddot = D_p * (p_d_dot_ - p_e_dot)  + K_p * (p_d_ - p_e);
  Vector3d omega_r_dot = D_o * (omega_d_ - omega_e) + K_o * e_o;
  Vector6d x_r_ddot;
  x_r_ddot << p_r_ddot, omega_r_dot;

// --- Singularity repulsive acceleration in task space (Dimeas 2016) with damping ---
  // Uses ∂σ_min/∂x in task space for the repulsive direction
  /*
  if (sigma_min_ < sigma_min_threshold_) {
    // Gradient of σ_min in task space: ∂σ_min/∂x = J_bar^T * ∂σ_min/∂q
    Vector6d grad_sigma_task = J_bar.transpose() * sigma_min_grad_cached_;
    double grad_norm = grad_sigma_task.norm();
    if (grad_norm > 1e-10) {
      Vector6d grad_dir = grad_sigma_task / grad_norm;

      // Repulsive strength (Dimeas 2016): k * (1/σ - 1/σ_thresh), saturated
      double strength = 1.0 / sigma_min_ - 1.0 / sigma_min_threshold_;
      double max_strength = manip_task_max_accel_ / manip_task_gain_;
      strength = std::min(strength, max_strength);

      // Repulsive acceleration
      Vector6d F_manip = manip_task_gain_ * strength * grad_dir;

      // Damping in repulsive direction to prevent oscillation
      double v_along_grad = v_e.dot(grad_dir);
      double d_manip = 2.0 * std::sqrt(manip_task_gain_ * strength);
      F_manip -= d_manip * v_along_grad * grad_dir;

      x_r_ddot += F_manip;  // comment this line to disable repulsive force
    }
  }
*/
  // Nullspace torques: damping + σ_min gradient (Dimeas 2017) + spring to home (Nakanishi 2008) + joint limit repulsion
  Matrix7x7d D_null = d_null_gains_.asDiagonal();
  Vector7d q_home = Eigen::Map<const Vector7d>(q_home_);
  Vector7d tau_jl = Vector7d::Zero();
  for (int i = 0; i < 7; ++i) {
      double q_bar = 0.5 * (q_max_[i] + q_min_[i]);
      double range = q_max_[i] - q_min_[i];
      tau_jl(i) = (q_(i) - q_bar) / (range * range);
  }
  Vector7d tau_null = -D_null * dq_filtered_ + null_sigma_gain_ * sigma_min_grad_cached_ - null_stiffness_gain_ * (q_ - q_home) - null_joint_limit_gain_ * tau_jl;
                      
  //Torques computation (eq 58 Nakanishi, 2008, from Khatib 1987)
  //return M*J_bar*(x_r_ddot - J_dot_dq) + C + (Matrix7x7d::Identity() - J.transpose() * J_bar.transpose()) * tau_null;
  //Gravity is not compensated because it is compensated in robot firmware
  
    // Publish pose error (position + orientation quaternion error) for monitoring
  // if (pose_error_pub_) {
  //   geometry_msgs::msg::TwistStamped msg;
  //   msg.header.stamp = get_node()->get_clock()->now();
  //   msg.twist.linear.x = p_d_(0) - p_e(0);
  //   msg.twist.linear.y = p_d_(1) - p_e(1);
  //   msg.twist.linear.z = p_d_(2) - p_e(2);
  //   msg.twist.angular.x = e_o(0);
  //   msg.twist.angular.y = e_o(1);
  //   msg.twist.angular.z = e_o(2);
  //   pose_error_pub_->publish(msg);
  // }

  // // Publish twist error (desired - actual velocity) for monitoring
  // if (twist_error_pub_) {
  //   geometry_msgs::msg::TwistStamped msg;
  //   msg.header.stamp = get_node()->get_clock()->now();
  //   msg.twist.linear.x = p_d_dot_(0) - p_e_dot(0);
  //   msg.twist.linear.y = p_d_dot_(1) - p_e_dot(1);
  //   msg.twist.linear.z = p_d_dot_(2) - p_e_dot(2);
  //   msg.twist.angular.x = omega_d_(0) - omega_e(0);
  //   msg.twist.angular.y = omega_d_(1) - omega_e(1);
  //   msg.twist.angular.z = omega_d_(2) - omega_e(2);
  //   twist_error_pub_->publish(msg);
  // }

  // if (tf_broadcaster_) {
  //   geometry_msgs::msg::TransformStamped t;
  //   t.header.stamp = get_node()->get_clock()->now();
  //   //t.header.frame_id = arm_prefix_ + robot_type_ + "_link0";
  //   t.header.frame_id = "base";
  //   t.child_frame_id = arm_prefix_ + "fk_pose";
  //   t.transform.translation.x = p_e(0);
  //   t.transform.translation.y = p_e(1);
  //   t.transform.translation.z = p_e(2);
  //   t.transform.rotation.x = q_e.x();
  //   t.transform.rotation.y = q_e.y();
  //   t.transform.rotation.z = q_e.z();
  //   t.transform.rotation.w = q_e.w();
  //   tf_broadcaster_->sendTransform(t);
  // }

  if (ee_pose_pub_) {
    geometry_msgs::msg::PoseStamped msg;
    Eigen::Quaterniond q_e_quat(T_e.block<3,3>(0,0));
    msg.header.stamp = get_node()->get_clock()->now();
    msg.pose.position.x = p_e(0);
    msg.pose.position.y = p_e(1);
    msg.pose.position.z = p_e(2);
    msg.pose.orientation.w = q_e_quat.w();
    msg.pose.orientation.x = q_e_quat.x();
    msg.pose.orientation.y = q_e_quat.y();
    msg.pose.orientation.z = q_e_quat.z();
    ee_pose_pub_->publish(msg);
  }

  if(gripper_status_pub_){
    std_msgs::msg::Bool msg;
    msg.data = gripper_is_closed_.load();
    gripper_status_pub_->publish(msg);
  }
  
  return J.transpose() * Lambda * (x_r_ddot - J_dot_dq) + C + (Matrix7x7d::Identity() - J.transpose() * J_bar.transpose()) * tau_null;
  

}


CallbackReturn CustomController::on_init() {
  try {
    auto_declare<std::string>("robot_type", "");
    auto_declare<std::string>("arm_prefix", "");
    auto_declare<std::vector<double>>("k_gains", {});
    auto_declare<std::vector<double>>("d_gains", {});
    auto_declare<std::vector<double>>("d_null_gains", {});
    auto_declare<double>("null_stiffness_gain", 10.0);
    auto_declare<double>("null_sigma_gain", 5.0);
    auto_declare<double>("grasp_width", 0.0);
    auto_declare<double>("grasp_speed", 0.1);
    auto_declare<double>("grasp_force", 40.0);
    auto_declare<double>("grasp_epsilon_inner", 0.005);
    auto_declare<double>("grasp_epsilon_outer", 0.08);
    auto_declare<double>("open_width", 0.08);
    auto_declare<double>("open_speed", 0.1);
    auto_declare<double>("twist_timeout", 0.5);
    auto_declare<double>("sigma_min_threshold", 0.05);
    auto_declare<double>("sigma_min_critical", 0.02);
    auto_declare<double>("lambda_max", 0.2);
    auto_declare<double>("manip_task_gain", 5.0);
    auto_declare<double>("manip_task_max_accel", 50.0);
    auto_declare<double>("null_joint_limit_gain", 20.0);
    // Collision behavior thresholds
    auto_declare<std::vector<double>>("lower_torque_thresholds_accel",
        {30.0, 30.0, 30.0, 25.0, 25.0, 25.0, 20.0});
    auto_declare<std::vector<double>>("upper_torque_thresholds_accel",
        {50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 40.0});
    auto_declare<std::vector<double>>("lower_torque_thresholds_nominal",
        {30.0, 30.0, 30.0, 25.0, 25.0, 25.0, 20.0});
    auto_declare<std::vector<double>>("upper_torque_thresholds_nominal",
        {50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 40.0});
    auto_declare<std::vector<double>>("lower_force_thresholds_accel",
        {20.0, 20.0, 20.0, 25.0, 25.0, 25.0});
    auto_declare<std::vector<double>>("upper_force_thresholds_accel",
        {50.0, 50.0, 50.0, 50.0, 50.0, 50.0});
    auto_declare<std::vector<double>>("lower_force_thresholds_nominal",
        {20.0, 20.0, 20.0, 25.0, 25.0, 25.0});
    auto_declare<std::vector<double>>("upper_force_thresholds_nominal",
        {50.0, 50.0, 50.0, 50.0, 50.0, 50.0});
    //auto_declare<bool>("gazebo", false);
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn CustomController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  robot_type_ = get_node()->get_parameter("robot_type").as_string();
  arm_prefix_ = get_node()->get_parameter("arm_prefix").as_string();
  arm_prefix_ = arm_prefix_.empty() ? "" : arm_prefix_ + "_";
  //gazebo_ = get_node()->get_parameter("gazebo").as_bool();

  auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  auto d_gains = get_node()->get_parameter("d_gains").as_double_array();
  auto d_null_gains = get_node()->get_parameter("d_null_gains").as_double_array();
  auto null_stiffness_gain = get_node()->get_parameter("null_stiffness_gain").as_double();
  null_sigma_gain_ = get_node()->get_parameter("null_sigma_gain").as_double();
  grasp_width_ = get_node()->get_parameter("grasp_width").as_double();
  grasp_speed_ = get_node()->get_parameter("grasp_speed").as_double();
  grasp_force_ = get_node()->get_parameter("grasp_force").as_double();
  grasp_epsilon_inner_ = get_node()->get_parameter("grasp_epsilon_inner").as_double();
  grasp_epsilon_outer_ = get_node()->get_parameter("grasp_epsilon_outer").as_double();
  open_width_ = get_node()->get_parameter("open_width").as_double();
  open_speed_ = get_node()->get_parameter("open_speed").as_double();
  twist_timeout_ = get_node()->get_parameter("twist_timeout").as_double();
  sigma_min_threshold_ = get_node()->get_parameter("sigma_min_threshold").as_double();
  sigma_min_critical_ = get_node()->get_parameter("sigma_min_critical").as_double();
  lambda_max_ = get_node()->get_parameter("lambda_max").as_double();
  manip_task_gain_ = get_node()->get_parameter("manip_task_gain").as_double();
  manip_task_max_accel_ = get_node()->get_parameter("manip_task_max_accel").as_double();
  null_joint_limit_gain_ = get_node()->get_parameter("null_joint_limit_gain").as_double();
  
  // --- Read collision behavior thresholds from config ---
  lower_torque_thresholds_accel_   = get_node()->get_parameter("lower_torque_thresholds_accel").as_double_array();
  upper_torque_thresholds_accel_   = get_node()->get_parameter("upper_torque_thresholds_accel").as_double_array();
  lower_torque_thresholds_nominal_ = get_node()->get_parameter("lower_torque_thresholds_nominal").as_double_array();
  upper_torque_thresholds_nominal_ = get_node()->get_parameter("upper_torque_thresholds_nominal").as_double_array();
  lower_force_thresholds_accel_    = get_node()->get_parameter("lower_force_thresholds_accel").as_double_array();
  upper_force_thresholds_accel_    = get_node()->get_parameter("upper_force_thresholds_accel").as_double_array();
  lower_force_thresholds_nominal_  = get_node()->get_parameter("lower_force_thresholds_nominal").as_double_array();
  upper_force_thresholds_nominal_  = get_node()->get_parameter("upper_force_thresholds_nominal").as_double_array();

  // --- Call SetFullCollisionBehavior service ---
  {
    auto collision_client = get_node()->create_client<franka_msgs::srv::SetFullCollisionBehavior>(
        "service_server/set_full_collision_behavior");

    auto request = std::make_shared<franka_msgs::srv::SetFullCollisionBehavior::Request>();

    // Copy torque thresholds (7 joints)
    for (size_t i = 0; i < 7; ++i) {
      request->lower_torque_thresholds_acceleration[i] = lower_torque_thresholds_accel_[i];
      request->upper_torque_thresholds_acceleration[i] = upper_torque_thresholds_accel_[i];
      request->lower_torque_thresholds_nominal[i]      = lower_torque_thresholds_nominal_[i];
      request->upper_torque_thresholds_nominal[i]      = upper_torque_thresholds_nominal_[i];
    }
    // Copy force thresholds (6 Cartesian components)
    for (size_t i = 0; i < 6; ++i) {
      request->lower_force_thresholds_acceleration[i] = lower_force_thresholds_accel_[i];
      request->upper_force_thresholds_acceleration[i] = upper_force_thresholds_accel_[i];
      request->lower_force_thresholds_nominal[i]      = lower_force_thresholds_nominal_[i];
      request->upper_force_thresholds_nominal[i]      = upper_force_thresholds_nominal_[i];
    }

    auto future_result = collision_client->async_send_request(request);
    if (future_result.wait_for(std::chrono::seconds(3)) == std::future_status::timeout) {
      RCLCPP_WARN(get_node()->get_logger(),
          "set_full_collision_behavior service not available. "
          "Skipping collision behavior configuration.");
    } else {
      auto response = future_result.get();
      if (!response->success) {
        RCLCPP_FATAL(get_node()->get_logger(),
            "Failed to set collision behavior: %s", response->error.c_str());
        return CallbackReturn::ERROR;
      }
      RCLCPP_INFO(get_node()->get_logger(), "Collision behavior set successfully.");
    }
  }
  
  if (k_gains.empty() || k_gains.size() != static_cast<uint>(6)) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_gains must have %d elements", 6);
    return CallbackReturn::FAILURE;
  }
  if (d_gains.empty() || d_gains.size() != static_cast<uint>(6)) {
    RCLCPP_FATAL(get_node()->get_logger(), "d_gains must have %d elements", 6);
    return CallbackReturn::FAILURE;
  }
  if (d_null_gains.empty() || d_null_gains.size() != static_cast<uint>(num_joints)) {
    RCLCPP_FATAL(get_node()->get_logger(), "d_null_gains must have %d elements", num_joints);
    return CallbackReturn::FAILURE;
  }

  for (int i = 0; i < 6; ++i) {
    k_gains_(i) = k_gains.at(i);
    d_gains_(i) = d_gains.at(i);
  }
  for (int i = 0; i < num_joints; ++i) {
    d_null_gains_(i) = d_null_gains.at(i);
  }
  null_stiffness_gain_ = null_stiffness_gain;
  dq_filtered_.setZero();

  // Get URDF from robot_state_publisher
  auto parameters_client =
      std::make_shared<rclcpp::AsyncParametersClient>(get_node(), "robot_state_publisher");
  parameters_client->wait_for_service();
  auto future = parameters_client->get_parameters({"robot_description"});
  auto result = future.get();
  if (!result.empty()) {
    robot_description_ = result[0].value_to_string();
  } else {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to get robot_description parameter.");
  }

  // Build dynamics model from URDF
  if (!robot_description_.empty()) {
    try {
dynamics_ = std::make_unique<Dynamics>(robot_description_, robot_type_, arm_prefix_);       RCLCPP_INFO(get_node()->get_logger(),
                  "Dynamics model built: %d joints, ee frame id=%lu",
                  dynamics_->njoints(), dynamics_->eeFrameId());
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_node()->get_logger(), "Failed to build dynamics model: %s", e.what());

    }
  }

  // Twist subscriber
  twist_sub_ = get_node()->create_subscription<geometry_msgs::msg::Twist>(
      "ee_twist_command", 1,
      [this](const geometry_msgs::msg::Twist::SharedPtr msg) { twistCallback(msg); });

  //Error publisher
  // pose_error_pub_ = get_node()->create_publisher<geometry_msgs::msg::TwistStamped>(
  //     "pose_error", 1);
  // twist_error_pub_ = get_node()->create_publisher<geometry_msgs::msg::TwistStamped>(
  //     "twist_error", 1);

  //Pose and gripper status publisher
  ee_pose_pub_ = get_node()->create_publisher<geometry_msgs::msg::PoseStamped>(
      "ee_pose", 1);

  gripper_status_pub_ = get_node()->create_publisher<std_msgs::msg::Bool>(
      "gripper_status", 1);


  // Desired pose tf broadcaster
  tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(get_node());

  // Gripper service
  gripper_service_ = get_node()->create_service<std_srvs::srv::SetBool>(
      "gripper_close_command",
      [this](const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
             std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
        gripperServiceCallback(request, response);
      });

  // Gripper action clients - aligned with gripper_example_controller
  std::string ns = get_node()->get_namespace();
  if (ns == "/") ns = "";

  gripper_grasp_client_ = rclcpp_action::create_client<franka_msgs::action::Grasp>(
      get_node(), ns + "/franka_gripper/grasp");
  gripper_move_client_ = rclcpp_action::create_client<franka_msgs::action::Move>(
      get_node(), ns + "/franka_gripper/move");

  assignMoveGoalOptionsCallbacks();
  assignGraspGoalOptionsCallbacks();


  return CallbackReturn::SUCCESS;
}

CallbackReturn CustomController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  updateJointStates();
  dq_filtered_.setZero();
  Matrix4x4d T0 = dynamics_->forwardKinematics(q_);
  q_d_quat_ = Eigen::Quaterniond(T0.block<3,3>(0,0));
  q_d_quat_.normalize();
  p_d_ = T0.block<3,1>(0,3);
  p_d_dot_.setZero();
  omega_d_.setZero();

  // Initialize twist buffer with zero twist
  {
    std::lock_guard<std::mutex> lock(twist_mutex_);
    twist_cmd_ = geometry_msgs::msg::Twist();
  }
  last_twist_time_ = std::chrono::steady_clock::now();
  twist_blocked_ = false;


  return CallbackReturn::SUCCESS;
}

void CustomController::updateJointStates() {
  for (int i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);
    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");
    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}




void CustomController::twistCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
  if(!first_twist_received_){first_twist_received_ = true;}
  std::lock_guard<std::mutex> lock(twist_mutex_);
  twist_cmd_ = *msg;
  last_twist_time_ = std::chrono::steady_clock::now();
}

void CustomController::gripperServiceCallback(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response) {

  if (gripper_action_in_progress_.load()) {
    response->success = false;
    response->message = "Gripper action already in progress";
    return;
  }

  // Check if gripper action servers are available
  bool grasp_available = gripper_grasp_client_ &&
      gripper_grasp_client_->action_server_is_ready();
  bool move_available = gripper_move_client_ &&
      gripper_move_client_->action_server_is_ready();

  if (!grasp_available || !move_available) {
    response->success = false;
    response->message = "Gripper action server not available (not connected or simulation mode)";
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 5000,
                          "Gripper action server not available - is franka_gripper_node running?");
    return;
  }

  if (request->data && !gripper_is_closed_.load()) {
    graspGripper();
    response->success = true;
    response->message = "Grasp command sent";
  } else if (!request->data && gripper_is_closed_.load()) {
    openGripper();
    response->success = true;
    response->message = "Open command sent";
  } else {
    response->success = true;
    response->message = gripper_is_closed_.load() ? "Gripper already closed" : "Gripper already open";
  }
}

void CustomController::assignMoveGoalOptionsCallbacks() {
  move_goal_options_.goal_response_callback =
      [this](const std::shared_ptr<rclcpp_action::ClientGoalHandle<franka_msgs::action::Move>>&
                 goal_handle) {
        if (!goal_handle) {
          RCLCPP_ERROR(get_node()->get_logger(), "Move Goal (open gripper) NOT accepted.");
        } else {
          RCLCPP_INFO(get_node()->get_logger(), "Move Goal accepted.");
        }
      };

  move_goal_options_.result_callback =
      [this](const rclcpp_action::ClientGoalHandle<franka_msgs::action::Move>::WrappedResult& result) {
        gripper_action_in_progress_.store(false);
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
          gripper_is_closed_.store(false);
          RCLCPP_INFO(get_node()->get_logger(), "Gripper opened.");
        } else {
          RCLCPP_WARN(get_node()->get_logger(), "Gripper open failed.");
        }
      };
}

void CustomController::assignGraspGoalOptionsCallbacks() {
  grasp_goal_options_.goal_response_callback =
      [this](const std::shared_ptr<rclcpp_action::ClientGoalHandle<franka_msgs::action::Grasp>>&
                 goal_handle) {
        if (!goal_handle) {
          RCLCPP_ERROR(get_node()->get_logger(), "Grasp Goal NOT accepted.");
        } else {
          RCLCPP_INFO(get_node()->get_logger(), "Grasp Goal accepted.");
        }
      };

  grasp_goal_options_.result_callback =
      [this](const rclcpp_action::ClientGoalHandle<franka_msgs::action::Grasp>::WrappedResult& result) {
        gripper_action_in_progress_.store(false);
        gripper_is_closed_.store(true);
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
          RCLCPP_INFO(get_node()->get_logger(), "Gripper grasp succeeded.");
        } else {
          RCLCPP_INFO(get_node()->get_logger(), "Gripper grasp completed (object may differ from expected width).");
        }
      };
}

void CustomController::openGripper() {
  if (!gripper_move_client_) {
    RCLCPP_WARN(get_node()->get_logger(), "Gripper move client not initialized.");
    return;
  }

  gripper_action_in_progress_.store(true);

  franka_msgs::action::Move::Goal goal;
  goal.width = open_width_;
  goal.speed = open_speed_;

  RCLCPP_INFO(get_node()->get_logger(), "Opening gripper - submitting Move Goal.");
  gripper_move_client_->async_send_goal(goal, move_goal_options_);
}

void CustomController::graspGripper() {
  if (!gripper_grasp_client_) {
    RCLCPP_WARN(get_node()->get_logger(), "Gripper grasp client not initialized.");
    return;
  }

  gripper_action_in_progress_.store(true);

  franka_msgs::action::Grasp::Goal goal;
  goal.width = grasp_width_;
  goal.speed = grasp_speed_;
  goal.force = grasp_force_;
  goal.epsilon.inner = grasp_epsilon_inner_;
  goal.epsilon.outer = grasp_epsilon_outer_;

  RCLCPP_INFO(get_node()->get_logger(), "Closing gripper - submitting Grasp Goal.");
  gripper_grasp_client_->async_send_goal(goal, grasp_goal_options_);
}

}  // namespace franka_example_controllers

#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CustomController,
                       controller_interface::ControllerInterface)

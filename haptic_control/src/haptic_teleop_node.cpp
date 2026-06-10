#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/bool.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Dense>

#include "raptor_api_interfaces/msg/out_virtuose_pose.hpp"
#include "raptor_api_interfaces/msg/out_virtuose_speed.hpp"
#include "raptor_api_interfaces/msg/out_virtuose_status.hpp"

#include <std_srvs/srv/set_bool.hpp>

using OutVirtuosePose   = raptor_api_interfaces::msg::OutVirtuosePose;
using OutVirtuoseSpeed  = raptor_api_interfaces::msg::OutVirtuoseSpeed;
using OutVirtuoseStatus = raptor_api_interfaces::msg::OutVirtuoseStatus;

class HapticTeleopNode : public rclcpp::Node
{
public:
  HapticTeleopNode()
  : Node("haptic_teleop_node"),
    tf_broadcaster_(this),
    home_initialized_(false),
    b_left_active_(false),
    b_right_prev_(false),
    gripper_closed_(false)
  {
    // --- Parameters ---
    this->declare_parameter("linear_scale",  1.0);
    this->declare_parameter("angular_scale", 1.0);

    // base_frame_: frame del Franka, usato come frame_id nei topic PoseStamped
    this->declare_parameter("base_frame",         std::string("fr3_link0"));

    // haptic_base_frame_: parent della TF dinamica haptic_ee, per RViz
    // MODIFICA 1: aggiunto parametro separato per il parent della TF
    this->declare_parameter("haptic_base_frame",  std::string("haptic_base"));

    this->declare_parameter("ee_frame",      std::string("haptic_ee"));
    this->declare_parameter("pose_mode",     std::string("both"));

    // MODIFICA 2: haptic_rotation_rpy → haptic_rotation_matrix (flat row-major 3x3)
    // Motivo: RPY non può esprimere swap di assi (det=-1), la matrice è più generale
    // Default: identità
    this->declare_parameter("haptic_rotation_matrix",
      std::vector<double>{1,0,0, 0,1,0, 0,0,1});

    linear_scale_       = this->get_parameter("linear_scale").as_double();
    angular_scale_      = this->get_parameter("angular_scale").as_double();
    base_frame_         = this->get_parameter("base_frame").as_string();
    haptic_base_frame_  = this->get_parameter("haptic_base_frame").as_string();
    ee_frame_           = this->get_parameter("ee_frame").as_string();
    pose_mode_          = this->get_parameter("pose_mode").as_string();

    // Costruzione R_h2f_ dalla matrice flat (row-major)
    auto m = this->get_parameter("haptic_rotation_matrix").as_double_array();
    R_h2f_ << m[0], m[1], m[2],
              m[3], m[4], m[5],
              m[6], m[7], m[8];

    // quat_h2f_ usato solo per ruotare l'orientamento nei topic PoseStamped
    quat_h2f_ = Eigen::Quaterniond(R_h2f_);
    quat_h2f_.normalize();

    RCLCPP_INFO(this->get_logger(),
      "R_h2f:\n"
      "  [%.3f  %.3f  %.3f]\n"
      "  [%.3f  %.3f  %.3f]\n"
      "  [%.3f  %.3f  %.3f]",
      R_h2f_(0,0), R_h2f_(0,1), R_h2f_(0,2),
      R_h2f_(1,0), R_h2f_(1,1), R_h2f_(1,2),
      R_h2f_(2,0), R_h2f_(2,1), R_h2f_(2,2));

    // --- Publishers ---
    if (pose_mode_ == "absolute" || pose_mode_ == "both") {
      abs_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "haptic_control/target_pose_absolute", 10);
    }
    if (pose_mode_ == "relative" || pose_mode_ == "both") {
      rel_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "haptic_control/target_pose_relative", 10);
    }
    vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
      "ee_twist_command", 10);
    gripper_client_ = this->create_client<std_srvs::srv::SetBool>("gripper_close_command");
    

    // --- Subscribers ---
    pose_sub_ = this->create_subscription<OutVirtuosePose>(
      "out_virtuose_pose", 1,
      std::bind(&HapticTeleopNode::poseCallback, this, std::placeholders::_1));
    speed_sub_ = this->create_subscription<OutVirtuoseSpeed>(
      "out_virtuose_speed", 1,
      std::bind(&HapticTeleopNode::speedCallback, this, std::placeholders::_1));
    status_sub_ = this->create_subscription<OutVirtuoseStatus>(
      "out_virtuose_status", 1,
      std::bind(&HapticTeleopNode::statusCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "HapticTeleopNode ready.");
    RCLCPP_INFO(this->get_logger(), "  base_frame        : %s", base_frame_.c_str());
    RCLCPP_INFO(this->get_logger(), "  haptic_base_frame : %s", haptic_base_frame_.c_str());
    RCLCPP_INFO(this->get_logger(), "  ee_frame          : %s", ee_frame_.c_str());
    RCLCPP_INFO(this->get_logger(), "  pose_mode         : %s", pose_mode_.c_str());
    RCLCPP_INFO(this->get_logger(), "  linear_scale      : %.3f", linear_scale_);
    RCLCPP_INFO(this->get_logger(), "  angular_scale     : %.3f", angular_scale_);
  }

private:

  // ---------------------------------------------------------------
  // statusCallback — invariato
  // ---------------------------------------------------------------
  void statusCallback(const OutVirtuoseStatus::SharedPtr msg)
  {
    constexpr uint32_t B_LEFT_MASK  = 6;
    constexpr uint32_t B_RIGHT_MASK = 8;

    bool was_active = b_left_active_;
    b_left_active_ = (msg->buttons == B_LEFT_MASK);

    if (b_left_active_ && !was_active) {
      RCLCPP_INFO(this->get_logger(), "[B_LEFT] Active - Franka paused!");
    } else if (!b_left_active_ && was_active) {
      home_initialized_ = false;
      RCLCPP_INFO(this->get_logger(), "[B_LEFT] Not active - Home pose restored!");
    }

    bool b_right_current = (msg->buttons & B_RIGHT_MASK) != 0;
    if (b_right_current && !b_right_prev_) {
      gripper_closed_ = !gripper_closed_;
      sendGripperCommand(gripper_closed_);
      RCLCPP_INFO(this->get_logger(), "[B_RIGHT] Gripper %s",
                  gripper_closed_ ? "CLOSED" : "OPEN");
    }
    b_right_prev_ = b_right_current;
  }

  // ---------------------------------------------------------------
  // poseCallback
  // ---------------------------------------------------------------
  void poseCallback(const OutVirtuosePose::SharedPtr msg)
  {
    if (b_left_active_) return;

    Eigen::Vector3d pos(
      msg->virtuose_pose.translation.x,
      msg->virtuose_pose.translation.y,
      msg->virtuose_pose.translation.z);
    Eigen::Quaterniond quat(
      msg->virtuose_pose.rotation.w,
      msg->virtuose_pose.rotation.x,
      msg->virtuose_pose.rotation.y,
      msg->virtuose_pose.rotation.z);
    quat.normalize();

    if (!home_initialized_) {
      home_pos_  = pos;
      home_quat_ = quat;
      home_initialized_ = true;
      RCLCPP_INFO(this->get_logger(), "Home pose initialized: [%.3f, %.3f, %.3f]",
        pos.x(), pos.y(), pos.z());
    }

    // MODIFICA 3: TF pubblica pose RAW relativa a haptic_base_frame_ (non ruotata)
    // Motivo: haptic_ee è fisicamente figlio del base dell'haptic device, non del Franka.
    // R_h2f_ non viene applicata qui — serve solo per velocity e PoseStamped topic.
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp    = msg->header.stamp;
    tf_msg.header.frame_id = haptic_base_frame_;  
    tf_msg.child_frame_id  = ee_frame_;           
    tf_msg.transform.translation.x = pos.x();
    tf_msg.transform.translation.y = pos.y();
    tf_msg.transform.translation.z = pos.z();
    tf_msg.transform.rotation.x = quat.x();
    tf_msg.transform.rotation.y = quat.y();
    tf_msg.transform.rotation.z = quat.z();
    tf_msg.transform.rotation.w = quat.w();
    tf_broadcaster_.sendTransform(tf_msg);

    // Pose ruotata in Franka frame — per i topic di controllo
    Eigen::Vector3d    pos_franka  = R_h2f_ * pos;
    Eigen::Quaterniond quat_franka = quat_h2f_ * quat;
    quat_franka.normalize();

    // PoseStamped absolute — frame_id = base_frame_ (fr3_link0)
    if (pose_mode_ == "absolute" || pose_mode_ == "both") {
      geometry_msgs::msg::PoseStamped pose_msg;
      pose_msg.header.stamp    = msg->header.stamp;
      pose_msg.header.frame_id = base_frame_;
      pose_msg.pose.position.x    = pos_franka.x();
      pose_msg.pose.position.y    = pos_franka.y();
      pose_msg.pose.position.z    = pos_franka.z();
      pose_msg.pose.orientation.x = quat_franka.x();
      pose_msg.pose.orientation.y = quat_franka.y();
      pose_msg.pose.orientation.z = quat_franka.z();
      pose_msg.pose.orientation.w = quat_franka.w();
      abs_pose_pub_->publish(pose_msg);
    }

    // PoseStamped relative — frame_id = base_frame_ (fr3_link0)
    if (pose_mode_ == "relative" || pose_mode_ == "both") {
      Eigen::Vector3d    delta_pos  = R_h2f_ * (pos - home_pos_);
      Eigen::Quaterniond delta_quat = (quat_h2f_ * home_quat_).inverse() * quat_franka;
      delta_quat.normalize();

      geometry_msgs::msg::PoseStamped rel_pose_msg;
      rel_pose_msg.header.stamp    = msg->header.stamp;
      rel_pose_msg.header.frame_id = base_frame_;
      rel_pose_msg.pose.position.x    = delta_pos.x();
      rel_pose_msg.pose.position.y    = delta_pos.y();
      rel_pose_msg.pose.position.z    = delta_pos.z();
      rel_pose_msg.pose.orientation.x = delta_quat.x();
      rel_pose_msg.pose.orientation.y = delta_quat.y();
      rel_pose_msg.pose.orientation.z = delta_quat.z();
      rel_pose_msg.pose.orientation.w = delta_quat.w();
      rel_pose_pub_->publish(rel_pose_msg);
    }
  }

  // ---------------------------------------------------------------
  // speedCallback — R_h2f_ applicata alla velocity (invariato nella logica)
  // ---------------------------------------------------------------
  void speedCallback(const OutVirtuoseSpeed::SharedPtr msg)
  {
    geometry_msgs::msg::Twist twist;

    if (!b_left_active_) {
      Eigen::Vector3d v_lin(
        msg->virtuose_speed.linear.x,
        msg->virtuose_speed.linear.y,
        msg->virtuose_speed.linear.z);
      Eigen::Vector3d v_ang(
        msg->virtuose_speed.angular.x,
        msg->virtuose_speed.angular.y,
        msg->virtuose_speed.angular.z);

      Eigen::Vector3d v_lin_franka = R_h2f_ * v_lin;
      Eigen::Vector3d v_ang_franka = R_h2f_ * v_ang;

      twist.linear.x  = v_lin_franka.x() * linear_scale_;
      twist.linear.y  = v_lin_franka.y() * linear_scale_;
      twist.linear.z  = v_lin_franka.z() * linear_scale_;
      twist.angular.x = v_ang_franka.x() * angular_scale_;
      twist.angular.y = v_ang_franka.y() * angular_scale_;
      twist.angular.z = v_ang_franka.z() * angular_scale_;
    }

    vel_pub_->publish(twist);
  }



  void sendGripperCommand(bool close){
  if (!gripper_client_->wait_for_service(std::chrono::milliseconds(200))) {
    RCLCPP_WARN(this->get_logger(), "Gripper service not available");
    return;
  }

  auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
  request->data = close;

  gripper_client_->async_send_request(request,
    [this, close](rclcpp::Client<std_srvs::srv::SetBool>::SharedFuture future) {
      auto response = future.get();
      if (response->success) {
        RCLCPP_INFO(this->get_logger(), "Gripper %s: %s",
          close ? "CLOSE" : "OPEN", response->message.c_str());
      } else {
        RCLCPP_WARN(this->get_logger(), "Gripper command failed: %s",
          response->message.c_str());
      }
    });
  }

  // --- Publishers ---
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr abs_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr rel_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr       vel_pub_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr gripper_client_;

  // --- Subscribers ---
  rclcpp::Subscription<OutVirtuosePose>::SharedPtr   pose_sub_;
  rclcpp::Subscription<OutVirtuoseSpeed>::SharedPtr  speed_sub_;
  rclcpp::Subscription<OutVirtuoseStatus>::SharedPtr status_sub_;

  // --- TF broadcaster ---
  tf2_ros::TransformBroadcaster tf_broadcaster_;

  // --- Frame transform ---
  Eigen::Matrix3d    R_h2f_;     // Haption → Franka, usata per velocity e PoseStamped
  Eigen::Quaterniond quat_h2f_;  // stessa rotazione come quaternione

  // --- Home pose ---
  Eigen::Vector3d    home_pos_;
  Eigen::Quaterniond home_quat_;
  bool               home_initialized_;

  // --- Button states ---
  bool b_left_active_;
  bool b_right_prev_;
  bool gripper_closed_;

  // --- Parameters ---
  double      linear_scale_;
  double      angular_scale_;
  std::string base_frame_;         // fr3_link0: frame_id nei topic PoseStamped
  std::string haptic_base_frame_;  // haptic_left/right_base: parent della TF per RViz
  std::string ee_frame_;
  std::string pose_mode_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HapticTeleopNode>());
  rclcpp::shutdown();
  return 0;
}
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

#include <string>

#include <Eigen/Eigen>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

namespace franka_example_controllers {

/**
 * Robot dynamics and kinematics computed directly via Pinocchio from a
 * URDF string.
 *
 * Unlike franka::Model / franka::RobotModel this class does NOT conflict
 * with the FrankaHardwareInterface plugin when loaded in the same
 * ros2_control_node process.
 */
class Dynamics {
 public:
  using Vector7d   = Eigen::Matrix<double, 7, 1>;
  using Matrix7d   = Eigen::Matrix<double, 7, 7>;
  using Matrix6x7d = Eigen::Matrix<double, 6, 7>;
  using Vector6d   = Eigen::Matrix<double, 6, 1>;
  using Matrix4d   = Eigen::Matrix4d;

  /**
   * Build the Pinocchio model from a URDF string.
   * @param urdf        Full URDF XML string.
   * @param robot_type  Robot type prefix used to find the EE frame
   *                    (e.g. "fr3" → looks for frame "fr3_link8").
   * @param arm_prefix  Optional arm prefix for multi-arm setups
   *                    (e.g. "left_" → looks for frame "left_fr3_link8").
   *                    Pass empty string for single-arm setups.
   * @throws std::runtime_error if the EE frame is not found in the URDF.
   */
  Dynamics(const std::string& urdf,
           const std::string& robot_type,
           const std::string& arm_prefix = "");

  // ---- Dynamics -------------------------------------------------------

  /// Gravity torque vector g(q)  [7x1]
  Vector7d gravity(const Vector7d& q) const;

  /// Joint-space mass (inertia) matrix M(q)  [7x7]
  Matrix7d massMatrix(const Vector7d& q) const;

  /// Coriolis + centrifugal vector C(q,dq)*dq  [7x1]
  Vector7d coriolis(const Vector7d& q, const Vector7d& dq) const;

  /// Coriolis + centrifugal matrix C(q,dq)  [7x7]
  /// Such that C(q,dq)*dq gives the Coriolis torque vector.
  Matrix7d coriolisMatrix(const Vector7d& q, const Vector7d& dq) const;

  // ---- Jacobians ------------------------------------------------------

  /// Geometric Jacobian in world (base) frame  [6x7]
  /// rows 0-2: linear,  rows 3-5: angular
  Matrix6x7d worldJacobian(const Vector7d& q) const;

  /// Body Jacobian in local end-effector frame  [6x7]
  Matrix6x7d bodyJacobian(const Vector7d& q) const;

  /// Bias acceleration dJ/dt * dq in world (base) frame  [6x1]
  /// This is the term that appears in: xddot = J*qddot + dJ*dq
  /// Computed analytically by Pinocchio (no finite differences).
  Vector6d worldBiasAcceleration(const Vector7d& q, const Vector7d& dq) const;

  /// Bias acceleration dJ/dt * dq in local end-effector frame  [6x1]
  Vector6d bodyBiasAcceleration(const Vector7d& q, const Vector7d& dq) const;

  /// Analytic Jacobian — ZYX RPY convention (Siciliano)  [6x7]
  /// Maps dq → [dx, dy, dz, droll, dpitch, dyaw]^T.
  /// Falls back to the geometric (world) Jacobian near the
  /// representation singularity (pitch ≈ ±π/2).
  Matrix6x7d analyticJacobian(const Vector7d& q) const;

  // ---- Kinematics -----------------------------------------------------

  /// Forward kinematics for the end-effector.
  /// @return 4x4 homogeneous transformation matrix (base → EE).
  Matrix4d forwardKinematics(const Vector7d& q) const;

  /// Forward kinematics for an arbitrary frame.
  /// @param frame_id  Pinocchio frame index.
  /// @return 4x4 homogeneous transformation matrix (base → frame).
  Matrix4d forwardKinematics(const Vector7d& q, pinocchio::FrameIndex frame_id) const;

  /// Numerical inverse kinematics (damped least-squares).
  Vector7d inverseKinematics(const Matrix4d& target,
                             const Vector7d& q_init,
                             int max_iter = 1000,
                             double eps = 1e-6,
                             double damping = 1e-3,
                             bool* success = nullptr) const;

  // ---- Manipulability -------------------------------------------------

  /// Yoshikawa manipulability index w = sqrt(det(J*J^T))
  /// @param J  Pre-computed 6x7 Jacobian (e.g. from worldJacobian()).
  /// @return Scalar manipulability index (≥ 0, zero at singularities).
  static double manipulability(const Matrix6x7d& J);

  /// Gradient of the Yoshikawa manipulability index ∂w/∂q  [7x1]
  ///
  /// Uses the analytical formula:
  ///   ∂w/∂q_i = w * tr( (J*J^T)^{-1} * (∂J/∂q_i) * J^T )
  ///
  /// where ∂J/∂q_i is approximated by central finite differences.
  ///
  /// @param q      Current joint configuration.
  /// @param J      Pre-computed 6x7 Jacobian at q (must be consistent).
  /// @param delta  Perturbation step for finite differences (default 1e-6).
  /// @return 7x1 gradient vector pointing towards higher manipulability.
  Vector7d manipulabilityGradient(const Vector7d& q, const Matrix6x7d& J,
                                  double delta = 1e-6) const;
/// Gradient of the minimum singular value ∂σ_min/∂q  [7x1]
  ///
  /// Uses the analytical formula (Lancaster 1964, Maciejewski 1989):
  ///   ∂σ_min/∂q_i = u_min^T * (∂J/∂q_i) * v_min
  ///
  /// where ∂J/∂q_i is computed analytically from joint axes and positions
  /// (Siciliano, Robotics ch. 3), and u_min, v_min are the left/right
  /// singular vectors corresponding to σ_min from the SVD of J.
  ///
  /// @param q      Current joint configuration.
  /// @param J      Pre-computed 6x7 Jacobian at q.
  /// @param u_min  Left singular vector (6x1) corresponding to σ_min.
  /// @param v_min  Right singular vector (7x1) corresponding to σ_min.
  /// @return 7x1 gradient vector pointing towards higher σ_min.
  Vector7d sigmaMinGradient(const Vector7d& q, const Matrix6x7d& J,
                            const Eigen::Matrix<double, 6, 1>& u_min,
                            const Eigen::Matrix<double, 7, 1>& v_min) const;
  // ---- Accessors ------------------------------------------------------

  int njoints() const { return pinocchio_model_.njoints; }
  pinocchio::FrameIndex eeFrameId() const { return ee_frame_id_; }

 private:
  static constexpr int kNumJoints = 7;

  mutable pinocchio::Model pinocchio_model_;
  mutable pinocchio::Data  pinocchio_data_;
  mutable pinocchio::Data  pinocchio_data_gravity_;
  pinocchio::FrameIndex    ee_frame_id_{0};

  Eigen::VectorXd pinQ(const Vector7d& q) const;
  Eigen::VectorXd pinV(const Vector7d& v) const;
};

}  // namespace franka_example_controllers
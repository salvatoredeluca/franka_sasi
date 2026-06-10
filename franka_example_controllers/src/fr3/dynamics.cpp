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

#include <franka_example_controllers/fr3/dynamics.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

// Pinocchio includes
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/urdf.hpp>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace franka_example_controllers {

// ---- Construction -----------------------------------------------------

Dynamics::Dynamics(const std::string& urdf,
                   const std::string& robot_type,
                   const std::string& arm_prefix) {
  pinocchio::urdf::buildModelFromXML(urdf, pinocchio_model_);
  pinocchio_data_         = pinocchio::Data(pinocchio_model_);
  pinocchio_data_gravity_ = pinocchio::Data(pinocchio_model_);

  // Build EE frame name: e.g. "left_fr3_link8" or "fr3_link8"
  
  std::string tcp_link = arm_prefix.empty()
      ? robot_type + "_hand_tcp"
      : arm_prefix + robot_type + "_hand_tcp";

  std::string link_8 = arm_prefix.empty()
      ? robot_type + "_link8"
      : arm_prefix + robot_type + "_link8";

  //Check if tcp frame exists. If yes, select it
  if (pinocchio_model_.existFrame(tcp_link)) {
    ee_frame_id_ = pinocchio_model_.getFrameId(tcp_link);
  } 
  //Otherwise, select link 8 frame
  else {
    if (pinocchio_model_.existFrame(link_8)) {
      ee_frame_id_ = pinocchio_model_.getFrameId(link_8);
    } 
    // If neither link 8 or hand tcp frame exists, throw the exception
    else {

      throw std::runtime_error(
          "Dynamics: EE frame '" + link_8 + "' or '" + tcp_link + "' not found in URDF. "
          "Check robot_type ('" + robot_type + "') and arm_prefix ('" + arm_prefix + "').");
    }
  }
}

// ---- Helpers ----------------------------------------------------------

Eigen::VectorXd Dynamics::pinQ(const Vector7d& q) const {
  Eigen::VectorXd q_pin = Eigen::VectorXd::Zero(pinocchio_model_.nq);
  q_pin.head(kNumJoints) = q;
  return q_pin;
}

Eigen::VectorXd Dynamics::pinV(const Vector7d& v) const {
  Eigen::VectorXd v_pin = Eigen::VectorXd::Zero(pinocchio_model_.nv);
  v_pin.head(kNumJoints) = v;
  return v_pin;
}

// ---- Dynamics ---------------------------------------------------------

Dynamics::Vector7d Dynamics::gravity(const Vector7d& q) const {
  pinocchio_model_.gravity.linear(Eigen::Vector3d(0.0, 0.0, -9.81));
  pinocchio::computeGeneralizedGravity(pinocchio_model_, pinocchio_data_gravity_, pinQ(q));
  return pinocchio_data_gravity_.g.head(kNumJoints);
}

Dynamics::Matrix7d Dynamics::massMatrix(const Vector7d& q) const {
  pinocchio::crba(pinocchio_model_, pinocchio_data_, pinQ(q));
  pinocchio_data_.M.triangularView<Eigen::StrictlyLower>() =
      pinocchio_data_.M.transpose().triangularView<Eigen::StrictlyLower>();
  return pinocchio_data_.M.topLeftCorner(kNumJoints, kNumJoints);
}

Dynamics::Vector7d Dynamics::coriolis(const Vector7d& q, const Vector7d& dq) const {
  auto q_pin = pinQ(q);
  auto v_pin = pinV(dq);
  Eigen::VectorXd a_zero = Eigen::VectorXd::Zero(pinocchio_model_.nv);

  pinocchio_model_.gravity.linear(Eigen::Vector3d(0.0, 0.0, -9.81));

  pinocchio::rnea(pinocchio_model_, pinocchio_data_, q_pin, v_pin, a_zero);
  pinocchio::computeGeneralizedGravity(pinocchio_model_, pinocchio_data_gravity_, q_pin);

  return (pinocchio_data_.tau - pinocchio_data_gravity_.g).head(kNumJoints);
}

Dynamics::Matrix7d Dynamics::coriolisMatrix(const Vector7d& q, const Vector7d& dq) const {
  auto q_pin = pinQ(q);
  auto v_pin = pinV(dq);

  pinocchio_model_.gravity.linear(Eigen::Vector3d(0.0, 0.0, -9.81));

  pinocchio::computeCoriolisMatrix(pinocchio_model_, pinocchio_data_, q_pin, v_pin);
  return pinocchio_data_.C.topLeftCorner(kNumJoints, kNumJoints);
}

// ---- Jacobians --------------------------------------------------------

Dynamics::Matrix6x7d Dynamics::worldJacobian(const Vector7d& q) const {
  auto q_pin = pinQ(q);
  pinocchio::computeJointJacobians(pinocchio_model_, pinocchio_data_, q_pin);
  pinocchio::updateFramePlacement(pinocchio_model_, pinocchio_data_, ee_frame_id_);

  Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, pinocchio_model_.nv);
  J.setZero();
  pinocchio::getFrameJacobian(pinocchio_model_, pinocchio_data_,
                              ee_frame_id_, pinocchio::LOCAL_WORLD_ALIGNED, J);
  return J.leftCols(kNumJoints);
}

Dynamics::Matrix6x7d Dynamics::bodyJacobian(const Vector7d& q) const {
  auto q_pin = pinQ(q);
  pinocchio::computeJointJacobians(pinocchio_model_, pinocchio_data_, q_pin);
  pinocchio::updateFramePlacement(pinocchio_model_, pinocchio_data_, ee_frame_id_);

  Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, pinocchio_model_.nv);
  J.setZero();
  pinocchio::getFrameJacobian(pinocchio_model_, pinocchio_data_,
                              ee_frame_id_, pinocchio::LOCAL, J);
  return J.leftCols(kNumJoints);
}

Dynamics::Vector6d Dynamics::worldBiasAcceleration(const Vector7d& q,
                                                    const Vector7d& dq) const {
  auto q_pin = pinQ(q);
  auto v_pin = pinV(dq);
  Eigen::VectorXd a_zero = Eigen::VectorXd::Zero(pinocchio_model_.nv);

  pinocchio::forwardKinematics(pinocchio_model_, pinocchio_data_, q_pin, v_pin, a_zero);
  pinocchio::updateFramePlacement(pinocchio_model_, pinocchio_data_, ee_frame_id_);

  return pinocchio::getFrameClassicalAcceleration(
             pinocchio_model_, pinocchio_data_, ee_frame_id_,
             pinocchio::LOCAL_WORLD_ALIGNED)
      .toVector();
}

Dynamics::Vector6d Dynamics::bodyBiasAcceleration(const Vector7d& q,
                                                   const Vector7d& dq) const {
  auto q_pin = pinQ(q);
  auto v_pin = pinV(dq);
  Eigen::VectorXd a_zero = Eigen::VectorXd::Zero(pinocchio_model_.nv);

  pinocchio::forwardKinematics(pinocchio_model_, pinocchio_data_, q_pin, v_pin, a_zero);
  pinocchio::updateFramePlacement(pinocchio_model_, pinocchio_data_, ee_frame_id_);

  return pinocchio::getFrameClassicalAcceleration(
             pinocchio_model_, pinocchio_data_, ee_frame_id_,
             pinocchio::LOCAL)
      .toVector();
}

Dynamics::Matrix6x7d Dynamics::analyticJacobian(const Vector7d& q) const {
  Matrix6x7d J_world = worldJacobian(q);

  Eigen::Matrix3d R = pinocchio_data_.oMf[ee_frame_id_].rotation();
  Eigen::Vector3d ypr = R.eulerAngles(2, 1, 0);  // [yaw, pitch, roll]
  double pitch = ypr(1);
  double roll  = ypr(2);

  double sr = std::sin(roll);
  double cr = std::cos(roll);
  double cp = std::cos(pitch);

  if (std::abs(cp) < 1e-10) {
    return J_world;  // singularity fallback
  }

  double sp = std::sin(pitch);
  double tp = sp / cp;

  // T^{-1} for ZYX convention (Siciliano eq. 3.46)
  Eigen::Matrix3d T_inv;
  T_inv << 1.0,  sr * tp,  cr * tp,
           0.0,  cr,      -sr,
           0.0,  sr / cp,  cr / cp;

  Matrix6x7d J_a;
  J_a.topRows(3)    = J_world.topRows(3);
  J_a.bottomRows(3) = T_inv * J_world.bottomRows(3);
  return J_a;
}

// ---- Kinematics -------------------------------------------------------

Dynamics::Matrix4d Dynamics::forwardKinematics(const Vector7d& q) const {
  return forwardKinematics(q, ee_frame_id_);
}

Dynamics::Matrix4d Dynamics::forwardKinematics(const Vector7d& q,
                                               pinocchio::FrameIndex frame_id) const {
  auto q_pin = pinQ(q);
  pinocchio::forwardKinematics(pinocchio_model_, pinocchio_data_, q_pin);
  pinocchio::updateFramePlacement(pinocchio_model_, pinocchio_data_, frame_id);
  return pinocchio_data_.oMf[frame_id].toHomogeneousMatrix();
}

Dynamics::Vector7d Dynamics::inverseKinematics(const Matrix4d& target,
                                               const Vector7d& q_init,
                                               int max_iter,
                                               double eps,
                                               double damping,
                                               bool* success) const {
  Eigen::Matrix3d R_des = target.block<3, 3>(0, 0);
  Eigen::Vector3d p_des = target.block<3, 1>(0, 3);
  pinocchio::SE3 oMdes(R_des, p_des);

  Vector7d q_current = q_init;

  for (int i = 0; i < max_iter; ++i) {
    auto q_pin = pinQ(q_current);
    pinocchio::forwardKinematics(pinocchio_model_, pinocchio_data_, q_pin);
    pinocchio::updateFramePlacement(pinocchio_model_, pinocchio_data_, ee_frame_id_);
    const pinocchio::SE3& oMcur = pinocchio_data_.oMf[ee_frame_id_];

    pinocchio::SE3 err_SE3 = oMcur.actInv(oMdes);
    Eigen::Matrix<double, 6, 1> err = pinocchio::log6(err_SE3).toVector();

    if (err.norm() < eps) {
      if (success) *success = true;
      return q_current;
    }

    pinocchio::computeJointJacobians(pinocchio_model_, pinocchio_data_, q_pin);
    Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, pinocchio_model_.nv);
    J.setZero();
    pinocchio::getFrameJacobian(pinocchio_model_, pinocchio_data_,
                                ee_frame_id_, pinocchio::LOCAL, J);
    Eigen::Matrix<double, 6, 7> J7 = J.leftCols(kNumJoints);

    Eigen::Matrix<double, 6, 6> JJt = J7 * J7.transpose();
    JJt.diagonal().array() += damping * damping;
    Vector7d dq = J7.transpose() * JJt.ldlt().solve(err);

    q_current += dq;
  }

  if (success) *success = false;
  return q_current;
}

// ---- Manipulability ---------------------------------------------------

double Dynamics::manipulability(const Matrix6x7d& J) {
  Eigen::Matrix<double, 6, 6> JJt = J * J.transpose();
  double det_val = JJt.determinant();
  return (det_val > 0.0) ? std::sqrt(det_val) : 0.0;
}

Dynamics::Vector7d Dynamics::manipulabilityGradient(const Vector7d& q,
                                                     const Matrix6x7d& J,
                                                     double delta) const {
  Eigen::Matrix<double, 6, 6> JJt = J * J.transpose();
  double det_val = JJt.determinant();
  double w = (det_val > 0.0) ? std::sqrt(det_val) : 0.0;

  Vector7d grad = Vector7d::Zero();

  if (w < 1e-10) {
    return grad;
  }

  Eigen::Matrix<double, 6, 6> JJt_inv = JJt.inverse();

  for (int i = 0; i < kNumJoints; ++i) {
    Vector7d q_plus  = q;
    Vector7d q_minus = q;
    q_plus(i)  += delta;
    q_minus(i) -= delta;

    Matrix6x7d J_plus  = worldJacobian(q_plus);
    Matrix6x7d J_minus = worldJacobian(q_minus);

    Matrix6x7d dJ_dqi = (J_plus - J_minus) / (2.0 * delta);

    Eigen::Matrix<double, 6, 6> product = JJt_inv * dJ_dqi * J.transpose();
    grad(i) = w * product.trace();
  }

  return grad;
}

Dynamics::Vector7d Dynamics::sigmaMinGradient(
    const Vector7d& q, const Matrix6x7d& J,
    const Eigen::Matrix<double, 6, 1>& u_min,
    const Eigen::Matrix<double, 7, 1>& v_min) const {

  // We need joint axes z_i and joint positions p_i in the base frame.
  // These are available from Pinocchio after forwardKinematics has been called.
  // Note: worldJacobian() already calls computeJointJacobians + updateFramePlacement,
  // so pinocchio_data_ contains the updated joint placements.

  // EE position in base frame
  Eigen::Vector3d p_ee = pinocchio_data_.oMf[ee_frame_id_].translation();

  Vector7d grad = Vector7d::Zero();

  for (int i = 0; i < kNumJoints; ++i) {
    // Joint i+1 in Pinocchio (joint 0 is universe)
    int joint_id = i + 1;
    Eigen::Vector3d z_i = pinocchio_data_.oMi[joint_id].rotation().col(2);  // joint axis in base frame
    Eigen::Vector3d p_i = pinocchio_data_.oMi[joint_id].translation();      // joint origin in base frame

    // Build ∂J/∂q_i (6x7 matrix, only columns j >= i are nonzero)
    Eigen::Matrix<double, 6, 7> dJ_dqi = Eigen::Matrix<double, 6, 7>::Zero();

    for (int j = i; j < kNumJoints; ++j) {
      int joint_j = j + 1;
      Eigen::Vector3d z_j = pinocchio_data_.oMi[joint_j].rotation().col(2);
      Eigen::Vector3d p_j = pinocchio_data_.oMi[joint_j].translation();

      // Angular part: ∂J_ang_j/∂q_i = z_i × z_j
      Eigen::Vector3d dJ_ang = z_i.cross(z_j);

      // Linear part: ∂J_lin_j/∂q_i = z_i × (z_j × (p_ee - p_j)) + z_j × (z_i × (p_ee - p_i))
      //            - z_j × (z_i × (p_j - p_i))
      // Simplified: = z_i × J_lin_j + z_j × (z_i × (p_j - p_i))  ... but let's use the direct form
      // ∂J_lin_j/∂q_i = z_i × (J_lin_j) where J_lin_j = z_j × (p_ee - p_j)
      // This is only exact for the contribution from rotating frame i.
      // The full derivative for revolute joints (Siciliano):
      //   ∂/∂q_i [z_j × (p_ee - p_j)] = (z_i × z_j) × (p_ee - p_j) + z_j × (z_i × (p_ee - p_j))
      //                                  for i <= j (both z_j and p_j depend on q_i)
      //   but p_ee also depends on q_i: ∂p_ee/∂q_i = z_i × (p_ee - p_i)
      //   and ∂p_j/∂q_i = z_i × (p_j - p_i) for i <= j
      //   and ∂z_j/∂q_i = z_i × z_j for i <= j
      //
      // Full formula:
      //   ∂J_lin_j/∂q_i = (z_i × z_j) × (p_ee - p_j) + z_j × (z_i × (p_ee - p_i) - z_i × (p_j - p_i))
      //                  = (z_i × z_j) × (p_ee - p_j) + z_j × (z_i × (p_ee - p_j))

      Eigen::Vector3d p_ee_minus_pj = p_ee - p_j;
      Eigen::Vector3d dJ_lin = (z_i.cross(z_j)).cross(p_ee_minus_pj)
                              + z_j.cross(z_i.cross(p_ee_minus_pj));

      dJ_dqi(0, j) = dJ_lin(0);
      dJ_dqi(1, j) = dJ_lin(1);
      dJ_dqi(2, j) = dJ_lin(2);
      dJ_dqi(3, j) = dJ_ang(0);
      dJ_dqi(4, j) = dJ_ang(1);
      dJ_dqi(5, j) = dJ_ang(2);
    }

    // ∂σ_min/∂q_i = u_min^T * (∂J/∂q_i) * v_min  (Lancaster 1964)
    grad(i) = u_min.transpose() * dJ_dqi * v_min;
  }

  return grad;
}

}  // namespace franka_example_controllers
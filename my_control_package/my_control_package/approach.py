import pinocchio as pin
from pathlib import Path
from sys import argv
import sys
import numpy as np
import xacro
import os
import math
import rclpy
import casadi
from pinocchio import casadi as cpin
from scipy.optimize import minimize
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, CameraInfo, Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class CollisionTaskNode(Node):
    def __init__(self):
        super().__init__('collision_task_node')

        # LEFT ARM (Camera - Ora è lui il principale/visuale)
        doc_left = self.urdf_file('left')
        left_urdf_string = doc_left.toxml()
        self.left_model = pin.buildModelFromXML(left_urdf_string)
        self.left_data = self.left_model.createData() 
        ee_frame_name_left = 'left_fr3_camera_link'
        self.left_frame_id = self.left_model.getFrameId(ee_frame_name_left)
        
        # --- Nuovo frame per il target di distanza del braccio destro ---
        target_frame_name_left = 'left_fr3_hand_tcp'
        if self.left_model.existFrame(target_frame_name_left):
            self.left_target_frame_id = self.left_model.getFrameId(target_frame_name_left)
        else:
            self.get_logger().warn(f"Frame {target_frame_name_left} not found, falling back to {ee_frame_name_left}")
            self.left_target_frame_id = self.left_frame_id

        self.left_collision_model = pin.buildGeomFromUrdfString(
            self.left_model, 
            left_urdf_string, 
            pin.GeometryType.COLLISION
        )
        self.left_collision_data = self.left_collision_model.createData()
        
        # RIGHT ARM (Tool)
        doc_right = self.urdf_file('right')
        right_urdf_string = doc_right.toxml()
        self.right_model = pin.buildModelFromXML(right_urdf_string)
        self.right_data = self.right_model.createData() 
        ee_frame_name_right = 'right_fr3_link8' 
        self.right_frame_id = self.right_model.getFrameId(ee_frame_name_right) 

        self.right_collision_model = pin.buildGeomFromUrdfString(
            self.right_model, 
            right_urdf_string, 
            pin.GeometryType.COLLISION
        )
        self.right_collision_data = self.right_collision_model.createData()

      

        self.add_sphere_frames(self.left_model, self.left_collision_model)
        self.add_sphere_frames(self.right_model,  self.right_collision_model)

        # ── LOG 2: pose delle geometrie a configurazione neutra ──────────────────
        q0_left = pin.neutral(self.left_model)
        q0_right  = pin.neutral(self.right_model)

        pin.forwardKinematics(self.left_model, self.left_data, q0_left)
        pin.forwardKinematics(self.right_model,  self.right_data,  q0_right)

      

        self.setupCasadi()

        # Expected names
        self.pinocchio_left_expected_names = [self.left_model.names[i] for i in range(1, self.left_model.njoints)]
        self.pinocchio_right_expected_names  = [self.right_model.names[i]  for i in range(1, self.right_model.njoints)]

        self.q_left    = None
        self.qdot_left = None
        self.q_right     = None
        self.qdot_right  = None

        self.Xt = None
        self.Yt = None
        self.Xa = None
        self.Ya = None
        self.Xb = None
        self.Yb = None

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.subscription_left_joints = self.create_subscription(
            JointState, '/left/joint_states', self.left_listener_callback, 10)
        self.subscription_right_joints = self.create_subscription(
            JointState, '/right/joint_states',  self.right_listener_callback,  10)

        self.left_publisher = self.create_publisher(Twist, 'left/ee_twist_command', 1)
        self.right_publisher  = self.create_publisher(Twist, 'right/ee_twist_command',  1)

        self.start_time = self.get_clock().now()
        self.timer = self.create_timer(0.2, self.on_timer)

    def cost_function(self, qdot, qd_dot):
        return np.linalg.norm(qdot - qd_dot)**2
    
    def add_sphere_frames(self, model, collision_model):
        for geom_id, geom_obj in enumerate(collision_model.geometryObjects):
            parent_joint_id = geom_obj.parentJoint
            placement       = geom_obj.placement
            parent_frame_id = model.getFrameId(model.names[parent_joint_id])
            frame = pin.Frame(
                f"sphere_{geom_id}",
                parent_joint_id,
                parent_frame_id,
                placement,
                pin.FrameType.BODY
            )
            model.addFrame(frame)

    def setupCasadi(self):
        self.left_cmodel = cpin.Model(self.left_model)
        self.left_cdata  = self.left_cmodel.createData()
        self.right_cmodel  = cpin.Model(self.right_model)
        self.right_cdata   = self.right_cmodel.createData()

        self.qL = casadi.SX.sym("qL", self.left_model.nq)
        self.qR = casadi.SX.sym("qR", self.right_model.nq)

        cpin.forwardKinematics(self.left_cmodel, self.left_cdata, self.qL)
        cpin.forwardKinematics(self.right_cmodel,  self.right_cdata,  self.qR)
        cpin.updateFramePlacements(self.left_cmodel, self.left_cdata)
        cpin.updateFramePlacements(self.right_cmodel,  self.right_cdata)
        cpin.computeJointJacobians(self.left_cmodel, self.left_cdata, self.qL)
        cpin.computeJointJacobians(self.right_cmodel,  self.right_cdata,  self.qR)

        self._build_collision_cost()

    def _build_collision_cost(self):
        alpha_c = 8.0
        beta_c  = 1.0
        rho_c   = 0.0016

        alpha_J = 10.0
        beta_J  = 1.0
        rho_J   = 0.005

        Q_LIMITS_MAX = [ 2.9007,  1.8361,  2.9007, -0.1169,  2.8763,  4.6216,  3.0508]
        Q_LIMITS_MIN = [-2.9007, -1.8361, -2.9007, -3.0770, -2.8763,  0.4398, -3.0508]

        alpha_S = 18
        beta_S  = 1.0
        rho_S   = 0.00

        # ── LOG 3: verifica pose simboliche a q=0 dentro CasADi ─────────────────
        q0 = np.zeros(self.left_model.nq)
        F_test_left = casadi.Function(
            "test_left",
            [self.qL],
            [self.left_cdata.oMi[1].translation,
             self.left_cdata.oMi[2].translation,
             self.left_cdata.oMi[3].translation]
        )
        res = F_test_left(q0)
        self.get_logger().info("=== CasADi LEFT oMi translations @ q=0 ===")
        for i, v in enumerate(res):
            self.get_logger().info(f"  joint {i+1}: {np.array(v).flatten()}")

        # ── H1: barriera collisioni ──────────────────────────────────────────────
        H1    = casadi.SX(0)
        d_min = casadi.SX(1e6)

        d_list   = []
        pair_names = []

        for l_id, l_geom in enumerate(self.left_collision_model.geometryObjects):
            l_joint_pose = self.left_cdata.oMi[l_geom.parentJoint]
            l_t_local    = casadi.SX(l_geom.placement.translation)
            l_pos        = casadi.mtimes(l_joint_pose.rotation, l_t_local) \
                           + l_joint_pose.translation
            try:
                l_radius = l_geom.geometry.radius
            except Exception:
                l_radius = 0.1

            for r_id, r_geom in enumerate(self.right_collision_model.geometryObjects):
                r_joint_pose = self.right_cdata.oMi[r_geom.parentJoint]
                r_t_local    = casadi.SX(r_geom.placement.translation)
                r_pos        = casadi.mtimes(r_joint_pose.rotation, r_t_local) \
                               + r_joint_pose.translation
                try:
                    r_radius = r_geom.geometry.radius
                except Exception:
                    r_radius = 0.1

                diff = l_pos - r_pos
                dist = casadi.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2 + 1e-8)
                d    = dist - (l_radius + r_radius)

                d_min = casadi.fmin(d_min, d)
                d_list.append(d)
                pair_names.append(f"{l_geom.name} <-> {r_geom.name}")

                d_safe = casadi.fmax(d, 0.01)
                H1 = H1 + casadi.exp(-alpha_c * d) *d**(-beta_c)

        # ── H2: barriera limiti giunti (ora su LEFT) ─────────────────────────────
        H2 = casadi.SX(0)
        for i in range(self.left_model.nq):
            d_upper = -self.qL[i] + Q_LIMITS_MAX[i]
            d_lower =  self.qL[i] - Q_LIMITS_MIN[i]
            H2 = H2 + casadi.exp(-alpha_J * d_upper) * d_upper**(-beta_J) \
                    + casadi.exp(-alpha_J * d_lower) * d_lower**(-beta_J)

        # ── H3: barriera singolarità (ora su LEFT) ───────────────────────────────
        J_ee    = cpin.getFrameJacobian(
            self.left_cmodel, self.left_cdata,
            self.left_cmodel.getFrameId('left_fr3_camera_link'),
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        JJT     = casadi.mtimes(J_ee, J_ee.T)
        det_JJT = casadi.det(JJT)
        H3      = rho_S * casadi.exp(-alpha_S * det_JJT) * det_JJT**(-beta_S)

        H_tot = rho_c * H1 + rho_J * H2 + H3
        DH    = casadi.jacobian(H_tot, self.qL).T # DH calcolato rispetto a qL

        self.F_cost = casadi.Function(
            "F_cost", [self.qL, self.qR], [DH, H_tot, d_min],
            ["qL", "qR"], ["DH", "H_tot", "d_min"]
        )

    def get_box_top_face_vertices(self, p_box, l, h):
        x, y, z = p_box
        return [
            np.array([x + l/2,  y + l/2,  z - h/2]),  # v0
            np.array([x - l/2,  y + l/2,  z - h/2]),  # v1
            np.array([x - l/2,  y - l/2,  z - h/2]),  # v2
            np.array([x + l/2,  y - l/2,  z - h/2]),  # v3
        ]
    
    def evaluate_single_segment_constraint(self, Rwc, pt_3, Jc, Jt,
                                        pa_world, pb_world):
        tb = 10
        xt, yt, zt = pt_3
        Lt  = compute_interaction_matrix(xt, yt, zt)
        Pct = Lt @ Jc
        pt  = np.array([xt/zt, yt/zt])

        Rwc_inv = np.linalg.inv(Rwc)

        pa_camera = Rwc_inv @ pa_world
        xa, ya, za = pa_camera
        La  = compute_interaction_matrix(xa, ya, za)
        Pca = La @ Jc
        pa  = np.array([xa/za, ya/za])

        pb_camera = Rwc_inv @ pb_world
        xb, yb, zb = pb_camera
        Lb  = compute_interaction_matrix(xb, yb, zb)
        Pcb = Lb @ Jc
        pb  = np.array([xb/zb, yb/zb])

        # Controlla che il segmento sia effettivamente più vicino
        # alla camera del tool 
        z_segment_avg = (za + zb) / 2
        if z_segment_avg >= zt:
            return None, None 
        alpha = (pb - pa).T @ (pb - pa)
        beta  = 2 * (pa - pt).T @ (pb - pa)
        gamma = (pa - pt).T @ (pa - pt)

        pb_dot = Pcb @ self.qdot_left # Jc lavora con qdot_left (Camera)
        pa_dot = Pca @ self.qdot_left
        pt_dot = Pct @ self.qdot_left

        alpha_prime = tb * (pb - pa).T @ (pb_dot - pa_dot)
        beta_prime  = tb * ((pa - pt).T @ (pb_dot - pa_dot) +
                            (pb - pa).T @ (pa_dot - pt_dot))
        gamma_prime = tb * (pa - pt).T @ (pa_dot - pt_dot)

        A = alpha + alpha_prime
        B = beta  + beta_prime
        C = gamma + gamma_prime

        sbar = self.evaluate_sbar(A, B, C)
        g    = -(A*sbar**2 + B*sbar + C)

        term1 = ((pb - pa).T @ (Pcb - Pca)) * (sbar**2)
        term2 = ((pb - pa).T @ (Pca - Pct) +
                (pa - pt).T  @ (Pcb - Pca)) * sbar
        term3 = (pa - pt).T @ (Pca - Pct)
        Ec    = tb * (term1 + term2 + term3)

        Ptt = Lt @ np.block([[Rwc.T,           np.zeros((3,3))],
                            [np.zeros((3,3)), np.zeros((3,3))]]) @ Jt
        Et  = -tb * ((pb - pa) * sbar + (pa - pt)).T @ Ptt

        E = np.concatenate((Ec, Et))
        return E, g
    
    def evaluate_E_g(self, Rwc, pt_3, Jc, Jt):
        
        p_box = np.array([-0.36, -0.27, 0.57])
        l = 0.04
        h = 0.15

        vertices = self.get_box_top_face_vertices(p_box, l, h)

       
        segments = [(vertices[i], vertices[(i+1) % 4]) for i in range(4)]

        constraints = []
        for pa_world, pb_world in segments:
            E, g = self.evaluate_single_segment_constraint(
                Rwc, pt_3, Jc, Jt, pa_world, pb_world)
            if E is not None:
                constraints.append((E, g))

        return constraints

    def evaluate_sbar(self, A, B, C):
        if A > 1e-5:
            sbar = np.clip(-B / (2*A), 0.0, 1.0)
        else:
            sbar = 0.0 if C <= A + B + C else 1.0
        return sbar

    def on_timer(self):
        if self.q_left is None or self.qdot_left is None \
                or self.q_right is None or self.qdot_right is None:
            self.get_logger().info('Waiting data...')
            return

        twist_msg_left  = Twist()
        twist_msg_right = Twist()
        
        world_frame  = 'world'
        # Inverto i frame TF letti dal listener (Camera a Sinistra, TCP Tool a Destra)
        link8_frame  = 'right_fr3_hand_tcp'
        camera_frame = 'left_fr3_camera_link_optical'
     
        try:
            Tcl = self.tf_buffer.lookup_transform(camera_frame, link8_frame,  rclpy.time.Time())
            Twc = self.tf_buffer.lookup_transform(world_frame,  camera_frame, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(f'Could not transform: {ex}')
            return
    
        x, y, z = (Tcl.transform.translation.x,
                   Tcl.transform.translation.y,
                   Tcl.transform.translation.z)
        pt_3 = np.array([x, y, z])

        self.X = x / z
        self.Y = y / z
        L = compute_interaction_matrix(x, y, z)
        
        Rwc = quaternion_to_rotation_matrix(Twc.transform.rotation)
        Rwc_augmented = np.block([[Rwc.T,            np.zeros((3,3))],
                                   [np.zeros((3,3)), Rwc.T          ]])
        
        e = np.array([self.X, self.Y])

        pin.computeJointJacobians(self.left_model, self.left_data, self.q_left)
        pin.updateFramePlacements(self.left_model, self.left_data)    
        J_left = pin.getFrameJacobian(self.left_model, self.left_data, 
                                       self.left_frame_id, 
                                       pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
        pin.computeJointJacobians(self.right_model, self.right_data, self.q_right)
        pin.updateFramePlacements(self.right_model, self.right_data)    
        J_right = pin.getFrameJacobian(self.right_model, self.right_data, 
                                      self.right_frame_id, 
                                      pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
        #Non so perché si muove troppo anche dopo aver raggiunto lo zero
        if np.linalg.norm(e) < 1e-3:
            left_twist_command = np.zeros(6)
        else:
            left_twist_command = -Rwc_augmented @ np.linalg.pinv(L) @ e
            
        LJl     = L @ Rwc_augmented @ J_left
        LJl_dag = np.linalg.pinv(LJl)

        result    = self.F_cost(self.q_left, self.q_right)
        DH_num    = np.array(result[0])
        H_val     = float(result[1])
        d_min_val = float(result[2])

        self.get_logger().info(
            f'H={H_val:.4f}  d_min={d_min_val:.4f} m  DH_norm={np.linalg.norm(DH_num):.4f}'
        )

        ql_nullspace = ((np.eye(7, dtype=np.int_) - LJl_dag @ LJl) @ DH_num).flatten()
        ql_dot_des   = 2.0* np.linalg.pinv(J_left) @ left_twist_command - 0.15*ql_nullspace

        
        # ─── LOGICA PER IL BRACCIO DESTRO (Approccio al TCP sinistro) ───────────
        # Estrai la posizione del tcp sinistro nel mondo
        p_left_target_world = self.left_data.oMf[self.left_target_frame_id].translation
        # Estrai la posizione del tcp/tool destro nel mondo
        p_right_world = self.right_data.oMf[self.right_frame_id].translation
        
        # Vettore che va dal braccio destro al TCP del braccio sinistro
        direction = p_left_target_world - p_right_world
        dist = np.linalg.norm(direction)

        t = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        
        target_dist = 0.80+0.26*math.sin(2*t) # Mantenersi a 40 cm di distanza
        Kp_pos = 0.1     # Guadagno proporzionale per la velocità lineare
        
        if dist > 1e-4:
            dir_norm = direction / dist
            v_linear = Kp_pos * (dist - target_dist) * dir_norm
        else:
            v_linear = np.zeros(3)

        # Crea il comando di twist spaziale per il braccio destro
        right_twist_command = np.array([
            v_linear[0],
            v_linear[1], 
            v_linear[2],
            0.0,  
            0.0,  
            0.0   
        ])
        # ─── FINE NUOVA LOGICA ───────────────────────────────────────────────────

        qr_dot_des = np.linalg.pinv(J_right) @ right_twist_command

        # Concatena mettendo la left per prima, coerenza con CasADi qL, qR
        qdot_current = np.concatenate((self.qdot_left, self.qdot_right))
        qd_desired   = np.concatenate((ql_dot_des, qr_dot_des))

        constraints = self.evaluate_E_g(Rwc, pt_3, J_left, J_right)

        scipy_constraints = []
        for E_i, g_i in constraints:
            scipy_constraints.append({
                'type': 'ineq',
                'fun':  lambda q, E=E_i, g=g_i: E @ q - g,
                'jac':  lambda q, E=E_i: E   
            })

        risultato = minimize(
            fun=self.cost_function,
            x0=qdot_current,
            args=(qd_desired,),
            method='SLSQP',
            # constraints=scipy_constraints           
        )

        if risultato.success:
            q_dot_ottimo = risultato.x
        else:
            self.get_logger().warn('Scipy failed')
            q_dot_ottimo = np.zeros(self.left_model.nv + self.right_model.nv)

        nv_left           = self.left_model.nv
        ql_dot_ottimo      = q_dot_ottimo[:nv_left]
        qr_dot_ottimo      = q_dot_ottimo[nv_left:]
        left_twist_command_opt = J_left @ ql_dot_ottimo
        right_twist_command_opt  = J_right  @ qr_dot_ottimo

        twist_msg_left.linear.x  = float(left_twist_command_opt[0])
        twist_msg_left.linear.y  = float(left_twist_command_opt[1])
        twist_msg_left.linear.z  = float(left_twist_command_opt[2])
        twist_msg_left.angular.x = float(left_twist_command_opt[3])
        twist_msg_left.angular.y = float(left_twist_command_opt[4])
        twist_msg_left.angular.z = float(left_twist_command_opt[5])

        twist_msg_right.linear.x  = float(right_twist_command_opt[0])
        twist_msg_right.linear.y  = float(right_twist_command_opt[1])
        twist_msg_right.linear.z  = float(right_twist_command_opt[2])
        twist_msg_right.angular.x = float(right_twist_command_opt[3])
        twist_msg_right.angular.y = float(right_twist_command_opt[4])
        twist_msg_right.angular.z = float(right_twist_command_opt[5])

        self.get_logger().info(f'x={x:.4f} y={y:.4f} z={z:.4f}')
        self.left_publisher.publish(twist_msg_left)
        self.right_publisher.publish(twist_msg_right)

    def left_listener_callback(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        name_to_vel = dict(zip(msg.name, msg.velocity))
        self.q_left    = np.array([name_to_pos.get(n, 0.0) for n in self.pinocchio_left_expected_names])
        self.qdot_left = np.array([name_to_vel.get(n, 0.0) for n in self.pinocchio_left_expected_names])

    def right_listener_callback(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        name_to_vel = dict(zip(msg.name, msg.velocity))
        self.q_right    = np.array([name_to_pos.get(n, 0.0) for n in self.pinocchio_right_expected_names])
        self.qdot_right = np.array([name_to_vel.get(n, 0.0) for n in self.pinocchio_right_expected_names])

    def urdf_file(self, prefix):
        pkg_path   = get_package_share_directory('franka_gazebo_bringup')
        xacro_file = os.path.join(pkg_path, 'urdf', 'franka_arm.gazebo.xacro')

        if prefix == "right":
            xyz='-0.343 0.416 0'
            rpy='0 0 -1.570796'
        else:
            xyz='-0.338 -0.8 0'
            rpy='0 0 -1.570796'
            
        doc = xacro.process_file(xacro_file, mappings={
            'arm_prefix': prefix, 'robot_type': 'fr3',
            'hand': 'false', 'ros2_control': 'true',
            'gazebo': 'true', 'ee_id': 'franka_hand',
            'gazebo_effort': 'true', 'xyz': xyz,
            'rpy': rpy, 'robot_namespace': prefix
        })
        return doc

def compute_interaction_matrix(x, y, z):
    xn, yn = x/z, y/z
    L = np.array([
        [-1/z,    0,   xn/z,    xn*yn,      -(1+xn**2),  yn],
        [   0, -1/z,   yn/z,  1+yn**2,        -xn*yn,   -xn]
    ])
    return L

def quaternion_to_rotation_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1-2*(y**2+z**2),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w),   1-2*(x**2+z**2), 2*(y*z-x*w)],
        [  2*(x*z-y*w),     2*(y*z+x*w), 1-2*(x**2+y**2)]
    ])

def main(args=None):
    rclpy.init(args=args)
    node = CollisionTaskNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
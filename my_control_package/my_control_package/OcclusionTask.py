import pinocchio as pin
from pathlib import Path
from sys import argv
import numpy as np
import xacro
import os
import math
import rclpy
from scipy.optimize import minimize
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, CameraInfo, Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class OcclusionTaskNode(Node):
    def __init__(self):
        super().__init__('occlusion_task_node')

        #  Pinocchio models and data (for jacobians) 
        
        # RIGHT ARM (Camera)
        doc_right = self.urdf_file('right')
        self.right_model = pin.buildModelFromXML(doc_right.toxml())
        self.right_data = self.right_model.createData() 
        ee_frame_name_right = 'right_fr3_camera_link'
        self.right_frame_id = self.right_model.getFrameId(ee_frame_name_right)

        # LEFT ARM (TCP Teleoperato)
        doc_left = self.urdf_file('left')
        self.left_model = pin.buildModelFromXML(doc_left.toxml())
        self.left_data = self.left_model.createData() 
        ee_frame_name_left = 'left_fr3_link8' 
        self.left_frame_id = self.left_model.getFrameId(ee_frame_name_left) 

        # Expected names
        self.pinocchio_right_expected_names = [self.right_model.names[i] for i in range(1, self.right_model.njoints)]
        self.pinocchio_left_expected_names = [self.left_model.names[i] for i in range(1, self.left_model.njoints)]

        #Current joint positions and velocities
        self.q_right = None
        self.qdot_right = None
        self.q_left = None
        self.qdot_left = None

        # Normalized coordinates TCP (left robot)
        self.Xt = None
        self.Yt = None
        
        # Normalized coordinates edges of the box
        self.Xa = None
        self.Ya = None
        self.Xb = None
        self.Yb = None

        # TF subscription
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Joint states subscriptions
        self.subscription_right_joints = self.create_subscription(
            JointState,
            '/right/joint_states',
            self.right_listener_callback,
            10)
            
        self.subscription_left_joints = self.create_subscription(
            JointState,
            '/left/joint_states',
            self.left_listener_callback,
            10)

        # Publishers
        self.right_publisher = self.create_publisher(Twist, 'right/ee_twist_command', 1)
        self.left_publisher = self.create_publisher(Twist, 'left/ee_twist_command', 1)


        self.start_time = self.get_clock().now()
        self.timer = self.create_timer(0.2, self.on_timer)

    def cost_function(self,qdot, qd_dot):
        
        
        return np.linalg.norm(qdot-qd_dot)**2
    


    

    #Should impose constraints in the form E * q_dot >= g    
    def evaluate_E_g(self,Rwc,pt_3,Jc,Jt):

        tb=5

        xt=pt_3[0]
        yt=pt_3[1]
        zt=pt_3[2]
        Lt=compute_interaction_matrix(xt,yt,zt)
        Pct= Lt @ Jc

        pt=np.array([xt/zt,yt/zt])
        

        x_a = -0.34
        y_a = -0.25
        z_a = 0.645
        pa_world = np.array([x_a, y_a, z_a])

        # Punto B: Angolo in alto, avanti, a destra
        x_b = -0.34
        y_b = -0.29
        z_b = 0.645
        pb_world = np.array([x_b, y_b, z_b])

        pa_camera=np.linalg.inv(Rwc) @ pa_world
        xa=pa_camera[0]
        ya=pa_camera[1]
        za=pa_camera[2]
        La=compute_interaction_matrix(xa,ya,za)
        Pca= La @ Jc
        pa=np.array([xa/za,ya/za])

        pb_camera=np.linalg.inv(Rwc) @ pb_world
        xb=pb_camera[0]
        yb=pb_camera[1]
        zb=pb_camera[2]
        Lb=compute_interaction_matrix(xb,yb,zb)
        Pcb= Lb @ Jc
        pb=np.array([xb/zb,yb/zb])







        alpha = (pb - pa).T @ (pb - pa)
        beta = 2 * (pa - pt).T @ (pb - pa)
        gamma = (pa - pt).T @ (pa - pt)

        pb_dot=Pcb @ self.qdot_right
        pa_dot=Pca @ self.qdot_right
        pt_dot=Pct @ self.qdot_right

       
        alpha_prime = tb * (pb - pa).T @ (pb_dot - pa_dot)
        beta_prime = tb * ((pa - pt).T @ (pb_dot - pa_dot) + (pb - pa).T @ (pa_dot - pt_dot))
        gamma_prime = tb * (pa - pt).T @ (pa_dot - pt_dot)

        A=alpha+alpha_prime
        B=beta+beta_prime
        C=gamma+gamma_prime



        sbar=self.evaluate_sbar(A,B,C)

        g=-(A*sbar**2+B*sbar+C)

        term1 = ((pb - pa).T @ (Pcb - Pca)) * (sbar**2)
        term2 = ((pb - pa).T @ (Pca - Pct) + (pa - pt).T @ (Pcb - Pca)) * sbar
        term3 = (pa - pt).T @ (Pca - Pct)
        
        Ec = tb * (term1 + term2 + term3)

        Ptt=Lt @ np.block([[Rwc.T,           np.zeros((3,3))],
                                  [np.zeros((3,3)),np.zeros((3,3))  ]]) @ Jt
        
        Et = -tb * ((pb- pa) * sbar + (pa - pt)).T @ Ptt


        E=np.concatenate((Ec,Et))
        
    
        return E , g
    


    def evaluate_sbar(self,A,B,C):

        

        if A > 1e-5:
            s_vertice = -B / (2 * A)
    
            sbar = np.clip(s_vertice, 0.0, 1.0) 
        else:
          
            val_0 = C                 
            val_1 = A + B + C        
            
            if val_0 <= val_1:
                sbar = 0.0
            else:
                sbar = 1.0


        return sbar
    
   
   
    
    def on_timer(self):
       
        if self.q_right is None or self.qdot_right is None or self.q_left is None or self.qdot_left is None:
            self.get_logger().info('Waiting data...')
            return

        twist_msg_right = Twist()
        twist_msg_left = Twist()
        
        world_frame = 'world'
        link8_frame = 'left_fr3_hand_tcp'
        camera_frame = 'right_fr3_camera_link_optical'
     
        try:
            Tcl = self.tf_buffer.lookup_transform(camera_frame, link8_frame, rclpy.time.Time())
            Twc = self.tf_buffer.lookup_transform(world_frame, camera_frame, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(f'Could not transform {link8_frame} to {camera_frame}: {ex}')
            return
    
        x = Tcl.transform.translation.x
        y = Tcl.transform.translation.y
        z = Tcl.transform.translation.z
        pt_3 = np.array([x,y,z])

        self.X = (x / z) 
        self.Y = (y / z) 
        L = compute_interaction_matrix(x, y, z)
        
        Rwc = quaternion_to_rotation_matrix(Twc.transform.rotation)
        Rwc_augmented = np.block([[Rwc.T,           np.zeros((3,3))],
                                  [np.zeros((3,3)), Rwc.T          ]])
        
        #Image Space Error
        e = np.array([self.X, self.Y])  

        #Compute right jacobian
        pin.computeJointJacobians(self.right_model, self.right_data, self.q_right)
        pin.updateFramePlacements(self.right_model, self.right_data)    
        J_right = pin.getFrameJacobian(self.right_model, self.right_data, 
                                       self.right_frame_id, 
                                       pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
        #Compute left jacobian
        pin.computeJointJacobians(self.left_model, self.left_data, self.q_left)
        pin.updateFramePlacements(self.left_model, self.left_data)    
        J_left = pin.getFrameJacobian(self.left_model, self.left_data, 
                                    self.left_frame_id, 
                                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
        
        right_twist_command = -2 * Rwc_augmented @ np.linalg.pinv(L) @ e
        qr_dot_des = np.linalg.pinv(J_right) @ right_twist_command

        t = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        left_twist_command = np.array([
            0.19 * math.cos(t), 
            0.19 * math.sin(t),                
            0.0,                
            0.0,                
            0.0,                
            0.0                 
        ])
        ql_dot_des = np.linalg.pinv(J_left) @ left_twist_command

        qdot_current = np.concatenate((self.qdot_right, self.qdot_left))
        qd_desired = np.concatenate((qr_dot_des, ql_dot_des))
      
        

        [E,g]=self.evaluate_E_g(Rwc,pt_3,J_right,J_left)

      

        constraints_occlusion = [{
            'type': 'ineq',
            'fun':  lambda qdot: E @ qdot - g,
            'jac':  lambda qdot: E,
        }]

        
        
        risultato = minimize(
            fun=self.cost_function,             
            x0=qdot_current,               
            args=(qd_desired,), 
            method='SLSQP',
            # constraints=constraints_occlusion            
        )

        if risultato.success:
            q_dot_ottimo = risultato.x
        else:
            self.get_logger().warn('Scipy failed')
            q_dot_ottimo = np.zeros(self.right_model.nv + self.left_model.nv)

        nv_right = self.right_model.nv
        qr_dot_ottimo = q_dot_ottimo[:nv_right]
        ql_dot_ottimo = q_dot_ottimo[nv_right:]

        right_twist_command_opt = J_right @ qr_dot_ottimo
        left_twist_command_opt = J_left @ ql_dot_ottimo

        twist_msg_right.linear.x = float(right_twist_command_opt[0])
        twist_msg_right.linear.y = float(right_twist_command_opt[1])
        twist_msg_right.linear.z = float(right_twist_command_opt[2])
        twist_msg_right.angular.x = float(right_twist_command_opt[3])
        twist_msg_right.angular.y = float(right_twist_command_opt[4])
        twist_msg_right.angular.z = float(right_twist_command_opt[5])

        twist_msg_left.linear.x = float(left_twist_command_opt[0])
        twist_msg_left.linear.y = float(left_twist_command_opt[1])
        twist_msg_left.linear.z = float(left_twist_command_opt[2])
        twist_msg_left.angular.x = float(left_twist_command_opt[3])
        twist_msg_left.angular.y = float(left_twist_command_opt[4])
        twist_msg_left.angular.z = float(left_twist_command_opt[5])

        self.get_logger().info(f'x={x:.4f} y={y:.4f} z={z:.4f}')

        self.right_publisher.publish(twist_msg_right)
        self.left_publisher.publish(twist_msg_left)


   
    def right_listener_callback(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        name_to_vel = dict(zip(msg.name, msg.velocity))
        q_list = []
        qdot_list = []
        for name in self.pinocchio_right_expected_names:
            pos = name_to_pos.get(name, 0.0)
            vel = name_to_vel.get(name, 0.0)
            q_list.append(pos)
            qdot_list.append(vel)
            
        self.q_right = np.array(q_list) 
        self.qdot_right = np.array(qdot_list)

    def left_listener_callback(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        name_to_vel = dict(zip(msg.name, msg.velocity))
        q_list = []
        qdot_list = []
        for name in self.pinocchio_left_expected_names: # Corretto il nome della lista
            pos = name_to_pos.get(name, 0.0)
            vel = name_to_vel.get(name, 0.0)
            q_list.append(pos)
            qdot_list.append(vel)
            
        self.q_left = np.array(q_list) 
        self.qdot_left = np.array(qdot_list)


   
    def urdf_file(self, prefix):
        pkg_path = get_package_share_directory('franka_gazebo_bringup')
        xacro_file = os.path.join(pkg_path, 'urdf', 'franka_arm.gazebo.xacro')

        doc = xacro.process_file(xacro_file, mappings={'arm_prefix': prefix, 'robot_type': 'fr3', 
                                                       'hand': 'false', 'ros2_control': 'true', 
                                                       'gazebo': 'true', 'ee_id': 'franka_hand', 
                                                       'gazebo_effort': 'true', 'xyz': '0 0 0', 
                                                       'rpy': '0 0 0', 'robot_namespace': prefix})
        return doc
       
def compute_interaction_matrix(x, y, z):
    xn = x / z   
    yn = y / z   

    L = np.array([
        [-1/z,   0,    xn/z,   xn*yn,       -(1 + xn**2),  yn ],
        [  0,   -1/z,  yn/z,  (1 + yn**2),   -xn*yn,       -xn]
    ])
    return L   

def quaternion_to_rotation_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
        [    2*(x*y + z*w),     1 - 2*(x**2 + z**2),   2*(y*z - x*w)],
        [    2*(x*z - y*w),         2*(y*z + x*w),   1 - 2*(x**2 + y**2)]
    ])
    return R

def main(args=None):
    rclpy.init(args=args)
    node = OcclusionTaskNode()
    rclpy.spin(node) 
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
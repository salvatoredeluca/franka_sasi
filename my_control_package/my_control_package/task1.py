import pinocchio as pin
from pathlib import Path
from sys import argv
import numpy as np
import xacro
import os
import rclpy
from rclpy.node import Node
from rclpy.wait_for_message import wait_for_message
from geometry_msgs.msg import Twist

from sensor_msgs.msg import JointState, CameraInfo, Image

from cv_bridge import CvBridge
import numpy as np

from ament_index_python.packages import get_package_share_directory

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener



class TaskNode(Node):
    def __init__(self):
        super().__init__('task_node')

        self.create_pin_model()

        self.pinocchio_expected_names = [self.model.names[i] for i in range(1, self.model.njoints)]

        self.q=None


        self.X=None
        self.Y=None


        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.subscription = self.create_subscription(
            JointState,
            '/right/joint_states',
            self.listener_callback,
            10)


        self.publisher = self.create_publisher(Twist, 'right/ee_twist_command', 1)
       
        self.timer = self.create_timer(0.2, self.on_timer)

   

    def on_timer(self):

        twist_msg=Twist()
        
        world_frame='world'
        link8_frame = 'left_fr3_hand_tcp'
        camera_frame='right_fr3_camera_link_optical'
     
        try:
            Tcl= self.tf_buffer.lookup_transform(camera_frame,link8_frame,rclpy.time.Time())
            Twc=self.tf_buffer.lookup_transform(world_frame,camera_frame,rclpy.time.Time())
            
            
        except TransformException as ex:
            self.get_logger().info(
            f'Could not transform {link8_frame} to {camera_frame}: {ex}')
            return
    
        x=Tcl.transform.translation.x
        y=Tcl.transform.translation.y
        z=Tcl.transform.translation.z

        self.X = (x / z) 
        self.Y = (y / z) 
        L=compute_interaction_matrix(x,y,z)

        
        Rwc=quaternion_to_rotation_matrix(Twc.transform.rotation)

        Rwc_augmented = np.block([[Rwc.T,           np.zeros((3,3))],
                                [np.zeros((3,3)), Rwc.T          ]])
        
        

        e = np.array([self.X, self.Y])  

        
        # twist_command=-2*Rwc_augmented @ np.linalg.pinv(L) @ e
        # anutwist_command=-2*Rwc_augmented @ np.linalg.pinv(L) @ e
        Lomega=L[:, [3,4,5]]

        omega_command=-4.5*Rwc@np.linalg.pinv(Lomega) @ e

        twist_msg.linear.x=float(0)
        twist_msg.linear.y=float(0)
        twist_msg.linear.z=float(0)
        twist_msg.angular.x=float(omega_command[0])
        twist_msg.angular.y=float(omega_command[1])
        twist_msg.angular.z=float(omega_command[2])

        self.get_logger().info(f'x={x:.4f} y={y:.4f} z={z:.4f}')

        self.publisher.publish(twist_msg)





    
    def listener_callback(self,msg):

        name_to_pos = dict(zip(msg.name, msg.position))
        q_list = []
        for name in self.pinocchio_expected_names:
            
            valore = name_to_pos.get(name, 0.0)
            q_list.append(valore)
            
        self.q = np.array(q_list) 




        

    def create_pin_model(self):
        pkg_path = get_package_share_directory('franka_gazebo_bringup')
        xacro_file = os.path.join(pkg_path, 'urdf', 'franka_arm.gazebo.xacro')

        doc_left = xacro.process_file(xacro_file, mappings={'arm_prefix': 'right', 'robot_type': 'fr3', 
                                                            'hand': 'false', 'ros2_control': 'true', 
                                                            'gazebo': 'true', 'ee_id': 'franka_hand', 
                                                            'gazebo_effort': 'true', 'xyz': '0 0 0', 
                                                            'rpy': '0 0 0', 'robot_namespace': 'right'})
        self.model = pin.buildModelFromXML(doc_left.toxml())
        self.data = self.model.createData()
        ee_frame_name = 'right_fr3_camera_link'
        self.frame_id = self.model.getFrameId(ee_frame_name)

        
    
      

def compute_interaction_matrix( x, y, z):
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
    node = TaskNode()
    rclpy.spin(node) 
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()  

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
import math

class RobotControllerNode(Node):
    def __init__(self):
        super().__init__('my_robot_controller')

        self.publisher_ = self.create_publisher(Twist, '/left/ee_twist_command', 1)

        self.start_time = self.get_clock().now()
       

        self.timer = self.create_timer(0.1, self.send_command)

    def send_command(self):
        twist_msg = Twist()
        t = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        
        # twist_msg.linear.z = 0.12*math.sin(t)
        twist_msg.linear.x = 0.12*math.cos(t)
        #self.get_logger().info(f'Movimento verso l alto - elapsed: {elapsed:.1f}s')
       

        self.publisher_.publish(twist_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RobotControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
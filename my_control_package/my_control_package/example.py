import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
 
class RobotControllerNode(Node):
    def __init__(self):
        super().__init__('my_robot_controller')
        
        self.publisher_ = self.create_publisher(Twist, '/left/ee_twist_command', 1)
        
        self.timer = self.create_timer(0.2, self.send_command)

    def send_command(self):
        msg = Twist()
        msg.linear.x=0.1
      
 
           
        self.publisher_.publish(msg)
        

def main(args=None):
    rclpy.init(args=args)
    node = RobotControllerNode()
    rclpy.spin(node) 
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
import rclpy
from franka_msgs.msg import Edges
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import  CameraInfo, Image

class EdgesDetectionNode(Node):
    def __init__(self):
        super().__init__('edges_detection_node')

        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None

        self.create_subscription(
            Image,
            '/right_camera/image',
            self.rgb_callback,
            10
        )

        self.create_subscription(
            Image,
            '/right_camera/depth_image',
            self.depth_callback,
            10
        )

       
        # self.create_subscription(
        #     CameraInfo,
        #     '/right_camera/camera_info',
        #     self.camera_info_callback,
        #     10
        # )

        self.edges_publisher = self.create_publisher(Edges, '/edges', 1)
        self.timer = self.create_timer(0.2, self.on_timer)

     
    def rgb_callback(self, msg: Image):
       
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg: Image):
       
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')


    def on_timer(self):
        if self.rgb_image is None or self.depth_image is None:
            self.get_logger().warn('Waiting for images...', throttle_duration_sec=2.0)
            return

    
def main(args=None):
    rclpy.init(args=args)
    node = EdgesDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
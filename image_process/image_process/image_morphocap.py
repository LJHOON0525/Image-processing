# image_process/image_morphocap.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageMorphoCap(Node):
    def __init__(self):
        super().__init__('image_morphocap')  
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)
        self.cap = cv2.VideoCapture(2)  # 2번 카메라 사용
        self.bridge = CvBridge()
        self.timer = self.create_timer(1/30, self.publish_frame)  # 30 FPS

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to capture image")
        else:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.publisher_.publish(msg)

def main(args=None):  
    rclpy.init(args=args)
    node = ImageMorphoCap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

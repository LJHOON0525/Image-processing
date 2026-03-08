# image_process/image_capture.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageCaptureNode(Node):
    def __init__(self):
        super().__init__('image_capture')
        self.publisher = self.create_publisher(Image, 'image_raw', 10)
        self.cap = cv2.VideoCapture(2)  # 비디오 2번
        self.cv_bridge = CvBridge()
        self.timer = self.create_timer(1/30, self.capture_image)

    def capture_image(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to capture image')
        else:
            image_msg = self.cv_bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.publisher.publish(image_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ImageCaptureNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

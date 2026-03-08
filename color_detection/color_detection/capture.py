# image_process/image_capture.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageCaptureNode(Node):
    def __init__(self):
        super().__init__('image_capture')
        self.publisher_1 = self.create_publisher(Image, 'image_raw_1', 10)
        self.publisher_2 = self.create_publisher(Image, 'image_raw_2', 10)

        self.cap1 = cv2.VideoCapture(2)  # 첫 번째 카메라
        self.cap2 = cv2.VideoCapture(4)  # 두 번째 카메라

        self.cv_bridge = CvBridge()
        self.timer = self.create_timer(1/30, self.capture_image)

    def capture_image(self):
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()

        if ret1:
            image_msg1 = self.cv_bridge.cv2_to_imgmsg(frame1, encoding="bgr8")
            self.publisher_1.publish(image_msg1)
        else:
            self.get_logger().warn('Failed to capture image from camera 2')

        if ret2:
            image_msg2 = self.cv_bridge.cv2_to_imgmsg(frame2, encoding="bgr8")
            self.publisher_2.publish(image_msg2)
        else:
            self.get_logger().warn('Failed to capture image from camera 4')

def main(args=None):
    rclpy.init(args=args)
    node = ImageCaptureNode()
    rclpy.spin(node)
    
    node.cap1.release()  # 자원 해제
    node.cap2.release()
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

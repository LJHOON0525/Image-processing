# image_process/image_roisub.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageRoiSub(Node):
    def __init__(self):
        super().__init__('image_roisub')
        self.subscription = self.create_subscription(Image, 'image_raw', self.process_frame, 10)
        self.bridge = CvBridge()

    def process_frame(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # ROI 설정 (x=100, y=100, w=300, h=300)
        x, y, w, h = 100, 100, 300, 300
        roi = frame[y:y+h, x:x+w]  # ROI 추출

        cv2.imshow('ROI Frame', roi)
        cv2.imshow('Oringnal',frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ImageRoiSub()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

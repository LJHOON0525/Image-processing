# image_process/image_processing.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageProcessingNode(Node):
    def __init__(self):
        super().__init__('image_processing')
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.process_image,
            10
        )
        self.cv_bridge = CvBridge()

    def process_image(self, msg):
        frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # 이미지 크기 변경 (1280 X 720) = 해상도 절약
        img_resized = cv2.resize(frame, (640,480))

        # 이미지 처리 코드
        blurred_frame = cv2.GaussianBlur(frame, (3, 3), 0)
        hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
        
        # 예시: 파란색 검출
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        result_blue = cv2.bitwise_and(frame, frame, mask=mask_blue)

        cv2.imshow('Original', frame)
        cv2.imshow("Blue Detection", result_blue)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

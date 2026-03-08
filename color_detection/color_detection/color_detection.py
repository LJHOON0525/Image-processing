import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ColorDetectionNode(Node):
    def __init__(self):
        super().__init__('color_detection_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, '/image_raw_1', self.image_callback, 10)
        self.publisher = self.create_publisher(Image, '/image_mask', 10)

    def image_callback(self, msg):
        # ROS 2 이미지 메시지를 OpenCV 형식으로 변환
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #파란색(B) 범위
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        result_blue = cv2.bitwise_and(frame, frame, mask=mask_blue)

        #초록색(G) 범위
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        result_green = cv2.bitwise_and(frame, frame, mask=mask_green)

        #빨간색(R) 범위 (두 개의 범위 필요)
        lower_red1 = np.array([0, 150, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 150, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = mask_red1 | mask_red2
        result_red = cv2.bitwise_and(frame, frame, mask=mask_red)

        # OpenCV 창으로 출력
        cv2.imshow('Original', frame)
        cv2.imshow('Blue Detection', result_blue)
        cv2.imshow('Green Detection', result_green)
        cv2.imshow('Red Detection', result_red)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ColorDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

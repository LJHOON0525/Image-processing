import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageMorphoDilate(Node):
    def __init__(self):
        super().__init__('morpho_dilate')
        self.subscription = self.create_subscription(
            Image,
            'image_raw', 
            self.image_callback,
            10
        )
        self.publisher_ = self.create_publisher(Image, 'image_dilated', 10)
        self.bridge = CvBridge()

        # 5x5 사각형 커널 생성
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def image_callback(self, msg):
        try:
            # ROS 이미지 메시지를 OpenCV 이미지로 변환
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 팽창 연산 적용
            dilated = cv2.dilate(gray, self.kernel, iterations=1)

            # 결과를 다시 퍼블리시
            msg_out = self.bridge.cv2_to_imgmsg(dilated, encoding="mono8")
            self.publisher_.publish(msg_out)

            # 디버깅용 OpenCV 창 표시
            cv2.imshow("Original", frame)
            cv2.imshow("Dilated Image", dilated)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ImageMorphoDilate()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

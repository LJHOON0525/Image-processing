import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
import cv2

class MultiImagePublisherGrapefruit(Node):
    def __init__(self):
        super().__init__('multi_image_publisher_grapefruit')

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.bridge = CvBridge()

        # 원본 Image 구독자
        
        self.sub_yolo_detection = self.create_subscription(Image, '/yolo_detection_image', self.yolo_detection_callback, qos) 
        self.sub_yellow_line = self.create_subscription(Image, '/yellow_line_image', self.yellow_line_callback, qos) 
        self.sub_red_flag_image = self.create_subscription(Image, '/red_flag_image', self.red_flag_image_callback, qos) 

        # 압축 CompressedImage 퍼블리셔

        self.pub_yolo_detection = self.create_publisher(CompressedImage, '/yolo_detection_image/compressed', qos)
        self.pub_yellow_line = self.create_publisher(CompressedImage, '/yellow_line_image/compressed', qos)
        self.pub_red_flag_image = self.create_publisher(CompressedImage, '/red_flag_image/compressed', qos)

    def _publish_compressed(self, msg: Image, publisher, topic_name: str):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # OpenCV JPEG 압축 (품질 10, 30 정도가 보통 적당함)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
            success, encoded_image = cv2.imencode('.jpg', cv_image, encode_param)

            if not success:
                self.get_logger().error(f"JPEG encoding failed at {topic_name}")
                return

            compressed = CompressedImage()
            compressed.header = msg.header
            compressed.format = "jpeg"
            compressed.data = encoded_image.tobytes()
            publisher.publish(compressed)

        except Exception as e:
            self.get_logger().error(f"Compression error at {topic_name}: {e}")

    # 개별 콜백들
   
    def yolo_detection_callback(self, msg):
        self._publish_compressed(msg, self.pub_yolo_detection, "/yolo_detection_image/compressed")

    def yellow_line_callback(self, msg):
        self._publish_compressed(msg, self.pub_yellow_line, "/yellow_line_image/compressed")

    def red_flag_image_callback(self, msg):
        self._publish_compressed(msg, self.pub_red_flag_image, "/red_flag_image/compressed")


def main(args=None):
    rclpy.init(args=args)
    node = MultiImagePublisherGrapefruit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


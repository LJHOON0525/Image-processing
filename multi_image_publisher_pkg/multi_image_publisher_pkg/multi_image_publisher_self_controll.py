import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
import cv2

class MultiImagePublisherSelfControll(Node):
    def __init__(self):
        super().__init__('multi_image_publisher_self_controll')

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.bridge = CvBridge()

        # 원본 Image 구독자
        self.sub_img_raw = self.create_subscription(Image, '/img_raw', self.img_raw_callback, qos) #원격 웹캠
        self.sub_img_data = self.create_subscription(Image, '/img_data', self.img_data_callback, qos) #봄,여름,가을,겨울,원격 주행


        # 압축 CompressedImage 퍼블리셔
        self.pub_img_raw = self.create_publisher(CompressedImage, '/img_raw/compressed', qos)
        self.pub_img_data = self.create_publisher(CompressedImage, '/img_data/compressed', qos)


    def _publish_compressed(self, msg: Image, publisher, topic_name: str):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # OpenCV로 직접 압축 (품질 10 설정, 30쯤이 딱 좋긴함)
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

    def img_data_callback(self, msg):
        self._publish_compressed(msg, self.pub_img_data, "/img_data/compressed")

        
    def img_raw_callback(self, msg):
        self._publish_compressed(msg, self.pub_img_raw, "/img_raw/compressed")


def main(args=None):
    rclpy.init(args=args)
    node = MultiImagePublisherSelfControll()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
import cv2


class MultiImagePublisher4(Node):
    def __init__(self):
        super().__init__('multi_image_publisher_4')

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.bridge = CvBridge()

        # ----- 원본 Image 구독자 -----

        self.sub_survivor_detection = self.create_subscription(
            Image, '/camera/survivor_detection', self.survivor_detection_callback, qos
        )

        self.sub_box_image = self.create_subscription(
            Image, '/img_raw', self.box_image_callback, qos
        )

        # ----- 압축 CompressedImage 퍼블리셔 -----

        self.pub_survivor_detection = self.create_publisher(
            CompressedImage, '/camera/survivor_detection/compressed', qos
        )

        self.pub_box_image = self.create_publisher(
            CompressedImage, '/img_raw/compressed', qos
        )

    def _publish_compressed(self, msg: Image, publisher, topic_name: str):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # OpenCV JPEG 압축 (품질 20)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
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

    # ----- 각 콜백 -----


    def survivor_detection_callback(self, msg):
        self._publish_compressed(msg, self.pub_survivor_detection,
                                 "/camera/survivor_detection/compressed")



    def box_image_callback(self, msg):
        self._publish_compressed(msg, self.pub_box_image,
                                 "/img_raw/compressed")


def main(args=None):
    rclpy.init(args=args)
    node = MultiImagePublisher4()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


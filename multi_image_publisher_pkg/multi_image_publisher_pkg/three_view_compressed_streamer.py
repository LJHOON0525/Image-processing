import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
import cv2
import numpy as np

class ThreeViewCompressedStreamer(Node):
    def __init__(self):
        super().__init__('three_view_compressed_streamer')

        self.bridge = CvBridge()

        # 구독할 3개의 토픽
        self.topics = ['/img_data/compressed', '/yolo_detection_image/compressed', '/side_camera/compressed']
        self.frames = {t: None for t in self.topics}

        qos_profile = QoSProfile(depth=1)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT

        # 구독자 생성
        for topic in self.topics:
            self.create_subscription(CompressedImage, topic,
                                     lambda msg, t=topic: self.image_callback(msg, t),
                                     qos_profile)

        # 퍼블리셔 (멀티뷰 합성)
        self.pub_multiview = self.create_publisher(CompressedImage, '/multiview/compressed', qos_profile)

        # 송출 주기
        self.fps = 30
        self.create_timer(1.0 / self.fps, self.publish_multiview)

    def image_callback(self, msg, topic):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.frames[topic] = cv2.resize(frame, (426, 240))  # 타일 크기
        except Exception as e:
            self.get_logger().error(f"{topic} decoding error: {e}")

    def publish_multiview(self):
        blank = np.zeros((240, 426, 3), dtype=np.uint8)
        imgs = [self.frames[t] if self.frames[t] is not None else blank for t in self.topics]

        # 2x2 그리드: 마지막 칸은 빈칸
        row1 = np.hstack([imgs[0], imgs[1]])
        row2 = np.hstack([imgs[2], blank])
        multiview = np.vstack([row1, row2])

        # 다시 JPEG로 압축하여 퍼블리시
        try:
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(multiview, dst_format='jpg')
            self.pub_multiview.publish(compressed_msg)
        except Exception as e:
            self.get_logger().error(f"Multiview compression error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ThreeViewCompressedStreamer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

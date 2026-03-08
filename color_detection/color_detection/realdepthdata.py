import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge


class RealSenseDepthPublisher(Node):
    def __init__(self):
        super().__init__('realsense_depth_publisher')

        qos_profile = QoSProfile(depth=10)
        self.depth_publisher = self.create_publisher(Image, 'depth_data', qos_profile)

        # RealSense 카메라 초기화
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # 깊이 스트림 설정 (640x480, 30FPS)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # 파이프라인 시작
        self.pipeline.start(self.config)

        # CvBridge 초기화
        self.bridge = CvBridge()

        # 30FPS로 실행
        self.timer = self.create_timer(1 / 30.0, self.publish_depth_data)

    def publish_depth_data(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        
        if not depth_frame:
            self.get_logger().warn("No depth frame received!")
            return

        # 깊이 프레임을 NumPy 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())

        # 깊이 데이터를 ROS2 Image 메시지로 변환하여 퍼블리시
        depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1")
        self.depth_publisher.publish(depth_msg)

        self.get_logger().info("Published depth frame")


def main(args=None):
    rclpy.init(args=args)
    node = RealSenseDepthPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down RealSense Depth Publisher")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from std_msgs.msg import String

import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge
import time


class DepthCapture(Node):
    def __init__(self):
        super().__init__('image_catcher')  # 노드 이름을 일관되게 소문자로 변경
        qos_profile = QoSProfile(depth=10)

        # Depth & Color Frame Publisher 설정
        self.depth_frame_pub = self.create_publisher(Image, 'depth_data', qos_profile)
        self.color_frame_pub = self.create_publisher(Image, 'color_data', qos_profile)

        # RealSense 파이프라인 설정
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Depth & Color 스트림 활성화
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        depth_profile = self.pipeline.start(self.config)

        # Depth Sensor 설정
        depth_sensor = depth_profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Clipping Distance 설정 (1m)
        clipping_distance_in_meters = 1.0
        self.clipping_distance = clipping_distance_in_meters / self.depth_scale

        # Depth와 Color 프레임 정렬 (Alignment)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # CvBridge 초기화
        self.cv_bridge = CvBridge()

        # 30FPS로 Depth Capture 실행
        self.timer = self.create_timer(1 / 30, self.depth_cap)

    def depth_cap(self):
        start_time = time.time()

        frames = self.pipeline.wait_for_frames()

        # Depth & Color 프레임 정렬
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # 프레임 데이터 가져오기
        align_image = np.asarray(aligned_depth_frame.get_data())  # asanyarray → asarray로 변경
        color_image = np.asarray(color_frame.get_data())

        # 배경 제거
        grey_color = 153
        align_image_3d = np.dstack((align_image, align_image, align_image))  # Depth는 1채널, Color는 3채널
        bg_removed = np.where(
            (align_image_3d > self.clipping_distance) | (align_image_3d <= 0),
            grey_color,
            color_image
        )

        # 메시지 변환 후 ROS2 토픽으로 퍼블리싱
        self.depth_frame_pub.publish(self.cv_bridge.cv2_to_imgmsg(bg_removed, encoding="passthrough"))
        self.color_frame_pub.publish(self.cv_bridge.cv2_to_imgmsg(color_image, encoding="bgr8"))

        # 프레임 속도 출력
        end_time = time.time()
        frame_rate = 1 / (end_time - start_time)
        self.get_logger().info(f'Frame rate: {frame_rate:.2f} FPS')


def main(args=None):
    rclpy.init(args=args)
    node = DepthCapture()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

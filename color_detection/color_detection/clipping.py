import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
import numpy as np
import pyrealsense2 as rs
import cv2
from cv_bridge import CvBridge

class RealSenseDepthMask(Node):
    def __init__(self):
        super().__init__('realsense_depth_mask')

        # ROS2 퍼블리셔 설정
        qos_profile = QoSProfile(depth=10)
        self.masked_color_pub = self.create_publisher(Image, '/masked_color_image', qos_profile)
        self.mask_pub = self.create_publisher(Image, '/near_mask', qos_profile)
        
        # 타이머로 30FPS 실행
        self.pub_timer = self.create_timer(1 / 30.0, self.process_frames)

        # RealSense 파이프라인 초기화
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth 활성화
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color 활성화
        self.pipeline.start(self.config)

        # Depth Scale 얻기
        depth_profile = self.pipeline.get_active_profile()
        depth_sensor = depth_profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Clipping 거리 설정 (단위: 미터)
        clipping_distance_in_meters = 1  # 1m
        self.clipping_distance = clipping_distance_in_meters / self.depth_scale  # mm 단위로 변환

        # cv_bridge 설정
        self.bridge = CvBridge()

    def process_frames(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return

        # 깊이 및 컬러 프레임을 NumPy 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Clipping 거리를 기준으로 마스크 생성
        near_mask = np.where((depth_image > 0) & (depth_image < self.clipping_distance), 255, 0).astype(np.uint8)

        # 컬러 이미지에 마스크 적용
        masked_color = cv2.bitwise_and(color_image, color_image, mask=near_mask)

        # ROS2 퍼블리시
        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(near_mask, encoding="mono8"))
        self.masked_color_pub.publish(self.bridge.cv2_to_imgmsg(masked_color, encoding="bgr8"))

        # 디버깅용 출력
        cv2.imshow("Near Mask (within 1m)", near_mask)
        cv2.imshow("Masked Color Image", masked_color)
        cv2.waitKey(1)

    def destroy_node(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RealSenseDepthMask()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

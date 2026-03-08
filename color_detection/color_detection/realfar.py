import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
import cv2

class RealSenseDepthMask(Node):
    def __init__(self):
        super().__init__('realsense_depth_mask')

        # ROS2 퍼블리셔 설정
        self.bridge = CvBridge()
        self.masked_color_pub = self.create_publisher(Image, '/masked_color_image', 10)
        self.mask_pub = self.create_publisher(Image, '/far_mask', 10)  # 먼 마스크 토픽 변경

        # RealSense 파이프라인 초기화
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth 활성화
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color 활성화
        self.pipeline.start(self.config)

        # 타이머로 30FPS 실행
        self.timer = self.create_timer(1 / 30.0, self.process_frames)

        # 거리 설정 (2m 이상 객체 감지)
        self.far_distance = 2000  # 2m (단위: mm)

    def process_frames(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return

        # 깊이 및 컬러 프레임을 NumPy 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 먼 마스크 생성 (2m 이상의 객체만 남김)
        far_mask = np.where(depth_image > self.far_distance, 255, 0).astype(np.uint8)

        # 컬러 이미지에 마스크 적용
        masked_color = cv2.bitwise_and(color_image, color_image, mask=far_mask)

        # ROS2 퍼블리시
        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(far_mask, encoding="mono8"))
        self.masked_color_pub.publish(self.bridge.cv2_to_imgmsg(masked_color, encoding="bgr8"))

        # 디버깅용 출력
        cv2.imshow("Far Mask (beyond 2m)", far_mask)
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

if __name__ == '__main__':
    main()

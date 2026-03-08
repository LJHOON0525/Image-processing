import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
import cv2


class RealSensePublisher(Node):
    def __init__(self):
        super().__init__('realsense_publisher')

        # QoS 설정
        self.bridge = CvBridge()
        self.depth_pub = self.create_publisher(Image, '/depth_image', 10)
        self.color_pub = self.create_publisher(Image, '/color_image', 10)

        # RealSense 파이프라인 초기화
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth 스트림 활성화
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color 스트림 활성화
        self.pipeline.start(self.config)

        # Depth 정렬
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # 타이머로 15FPS 실행
        self.timer = self.create_timer(1/15, self.publish_frames)

    def publish_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        # 깊이 프레임 & 컬러 프레임 추출
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return

        # NumPy 배열 변환
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 깊이 데이터 시각화를 위해 컬러맵 적용
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # ROS2 토픽 퍼블리시
        self.depth_pub.publish(self.bridge.cv2_to_imgmsg(depth_colormap, encoding="bgr8"))
        self.color_pub.publish(self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8"))

        # 프레임 출력 (디버깅용)
        cv2.imshow("Depth Frame", depth_colormap)
        cv2.imshow("Color Frame", color_image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = RealSensePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.pipeline.stop()
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

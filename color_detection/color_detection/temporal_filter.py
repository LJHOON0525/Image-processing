import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
import cv2


class RealSenseTemporalFilter(Node):
    def __init__(self):
        super().__init__('realsense_temporal_filter')

        self.bridge = CvBridge()
        self.filtered_depth_pub = self.create_publisher(Image, '/filtered_depth', 10)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

        # 📌 Temporal Filter 설정
        self.temporal_filter = rs.temporal_filter()
        self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)  # 0~1 사이 값 (높을수록 부드러움)
        self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20)   # 깊이 변화 허용값

        self.timer = self.create_timer(1 / 30.0, self.process_frames)

    def process_frames(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            return

        # 📌 Temporal Filter 적용
        filtered_depth_frame = self.temporal_filter.process(depth_frame)

        # 🔍 Depth 이미지를 NumPy 배열로 변환
        filtered_depth_image = np.asanyarray(filtered_depth_frame.get_data())

        # 🔥 Depth 정규화 (0~255로 변환)
        normalized_depth = cv2.normalize(filtered_depth_image, None, 0, 255, cv2.NORM_MINMAX)
        normalized_depth = np.uint8(normalized_depth)

        # 🎨 컬러맵 적용
        colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)

        # 🛰 ROS2 퍼블리시
        self.filtered_depth_pub.publish(self.bridge.cv2_to_imgmsg(filtered_depth_image, encoding="mono16"))

        # 📺 OpenCV 시각화
        cv2.imshow("Filtered Depth Image (Temporal)", colored_depth)
        cv2.waitKey(1)

    def destroy_node(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RealSenseTemporalFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

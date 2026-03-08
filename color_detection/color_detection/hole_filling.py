import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
import cv2


class RealSenseHoleFilling(Node):
    def __init__(self):
        super().__init__('realsense_hole_filling')

        self.bridge = CvBridge()
        self.filtered_depth_pub = self.create_publisher(Image, '/filtered_depth', 10)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

        self.hole_filling_filter = rs.hole_filling_filter()

        self.timer = self.create_timer(1 / 30.0, self.process_frames)

    def process_frames(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            return

        filtered_depth_frame = self.hole_filling_filter.process(depth_frame)
        filtered_depth_image = np.asanyarray(filtered_depth_frame.get_data())

        #  Depth 이미지 정규화 (0~255)
        normalized_depth = cv2.normalize(filtered_depth_image, None, 0, 255, cv2.NORM_MINMAX)
        normalized_depth = np.uint8(normalized_depth)

        #  컬러맵 적용
        colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)

        self.filtered_depth_pub.publish(self.bridge.cv2_to_imgmsg(filtered_depth_image, encoding="mono16"))

        cv2.imshow("Filtered Depth Image", colored_depth)
        cv2.waitKey(1)

    def destroy_node(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RealSenseHoleFilling()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

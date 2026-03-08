import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
import cv2


class RealSenseRGBOnly(Node):
    def __init__(self):
        super().__init__('realsense_rgb_only')

        self.bridge = CvBridge()
        self.rgb_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

        self.pipeline.start(self.config)

        self.timer = self.create_timer(1/30.0, self.process_frames)

    def process_frames(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return

        # numpy 배열 변환
        color_image = np.asanyarray(color_frame.get_data())

        # ROS2 퍼블리시
        self.rgb_pub.publish(self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8"))

        # OpenCV 시각화
        #cv2.imshow("RGB Camera (1920x1080 @30FPS)", color_image)
        #cv2.waitKey(1)

    def destroy_node(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RealSenseRGBOnly()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

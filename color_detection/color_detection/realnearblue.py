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
        self.mask_pub = self.create_publisher(Image, '/near_mask', 10)

        # 파이프라인 
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

        depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.get_logger().info(f'Depth Scale: {self.depth_scale:.6f} meters per unit')

        # 1미터 이내 거리만 감지
        self.near_distance = 1.0  # 단위: meter

        # 타이머로 30FPS 처리
        self.timer = self.create_timer(1 / 30.0, self.process_frames)

    def process_frames(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_meters = depth_image * self.depth_scale

        near_mask = np.where((depth_meters > 0) & (depth_meters < self.near_distance), 255, 0).astype(np.uint8)

        blue_masked_color = np.zeros_like(color_image)
        blue_masked_color[near_mask == 255] = (255, 0, 0)  #  파란색
        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(near_mask, encoding="mono8"))
        self.masked_color_pub.publish(self.bridge.cv2_to_imgmsg(blue_masked_color, encoding="bgr8"))

        cv2.imshow("Original",color_image)
        #cv2.imshow("Near Mask (within 1m)", near_mask)
        cv2.imshow("Blue Masked Color Image", blue_masked_color)
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

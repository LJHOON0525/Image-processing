import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np
import cv2

class D435iCrossNode(Node):
    def __init__(self):
        super().__init__('d435i_cross_node')

        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        config = rs.config()
        config.enable_stream(rs.stream.accel)
        config.enable_stream(rs.stream.gyro)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        self.timer = self.create_timer(0.01, self.timer_callback)

    def timer_callback(self):
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()

            if not color_frame:
                self.get_logger().warn("Color frame not found.")
                return

            color_image = np.asanyarray(color_frame.get_data())
            image_with_cross = self.apply_mask(color_image)

            cv2.imshow("Frame with Cross", image_with_cross)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().warn(f"Error during frame processing: {e}")

    def apply_mask(self, image):
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # 큰 십자가를 그리기 위한 마스크 생성
        center = (w // 2, h // 2)
        cross_size = 200

        cv2.line(mask, (center[0] - cross_size, center[1]), (center[0] + cross_size, center[1]), 255, 2)
        cv2.line(mask, (center[0], center[1] - cross_size), (center[0], center[1] + cross_size), 255, 2)

        # bitwise 연산으로 마스크와 원본 이미지를 결합하여 십자가 외 부분을 검정색으로 처리
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        return masked_image

def main(args=None):
    rclpy.init(args=args)
    node = D435iCrossNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

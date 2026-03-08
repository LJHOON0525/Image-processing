import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge


class Lanedecttest(Node):
    def __init__(self):
        super().__init__('spring_color_checker')

        self.img_size_x = 848
        self.img_size_y = 480
        self.U_detection_threshold = 140

        self.cvbrid = CvBridge()

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.img_size_x, self.img_size_y, rs.format.bgr8, 15)
        self.config.enable_stream(rs.stream.depth, self.img_size_x, self.img_size_y, rs.format.z16, 15)

        profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.hole_filling_filter = rs.hole_filling_filter()

        self.capture_timer = self.create_timer(1/15, self.image_capture)
        self.process_timer = self.create_timer(1/15, self.image_processing)

        self.center_x = int(self.img_size_x / 2)
        self.color_img = np.zeros((self.img_size_y, self.img_size_x, 3), dtype=np.uint8)

    def image_capture(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()

        self.color_img = np.asanyarray(color_frame.get_data())

    def yuv_detection(self, img):
        gaussian = cv2.GaussianBlur(img, (3, 3), 1)
        yuv_img = cv2.cvtColor(gaussian, cv2.COLOR_BGR2YUV)
        _, U_img, _ = cv2.split(yuv_img)

        ret, U_img_treated = cv2.threshold(U_img, self.U_detection_threshold, 255, cv2.THRESH_BINARY)

        if ret:
            contours, _ = cv2.findContours(U_img_treated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_area = 0
            max_contour = None
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    max_contour = contour

            if max_contour is not None:
                x, y, w, h = cv2.boundingRect(max_contour)
                left = x
                right = x + w
                center = int((left + right) / 2)
                return left, center, right, max_contour
        return None, None, None, None

    def image_processing(self):
        x1 = int(self.img_size_x * 0.05)
        x2 = int(self.img_size_x * 0.95)
        y1 = int(self.img_size_y * 0.7)
        y2 = int(self.img_size_y * 0.9)

        roi = self.color_img[y1:y2, x1:x2]

        left, center, right, contour = self.yuv_detection(roi)

        if center is not None:
            # Low-pass filtering
            alpha = 0.2
            target_center_x = x1 + center
            self.center_x = int((1 - alpha) * self.center_x + alpha * target_center_x)

            angle_offset = right - left
            self.get_logger().info(f"Angle Offset: {angle_offset}")

            # 시각화
            # 원본 이미지에 중심선
            cv2.line(self.color_img, (self.center_x, y1), (self.center_x, y2), (255, 0, 0), 4)

            # ROI 상에 좌/우 경계선
            cv2.line(roi, (left, 0), (left, y2 - y1), (0, 255, 255), 2)  # Left boundary - Yellow
            cv2.line(roi, (right, 0), (right, y2 - y1), (0, 255, 255), 2)  # Right boundary - Yellow

            # ROI 컨투어 시각화
            cv2.drawContours(roi, [contour], -1, (0, 0, 255), 2)

        # ROI 사각형 그리기
        cv2.rectangle(self.color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("ROI", roi)
        cv2.imshow("Color Image", self.color_img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = Lanedecttest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

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

        self.L_sum = 0
        self.R_sum = 0
        self.color_ROI = np.zeros((int(self.img_size_y * 0.2), int(self.img_size_x * 0.9), 3), dtype=np.uint8)

        self.center_x = int(self.img_size_x / 2)

    def image_capture(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()

        self.color_img = np.asanyarray(color_frame.get_data())

    def yuv_detection(self, img):
        y, x, c = img.shape

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
                mask = np.zeros_like(U_img_treated)
                cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)

                filtered = cv2.bitwise_and(img, img, mask=mask)
                cv2.imshow("Filtered U Channel", filtered)
                cv2.waitKey(1)

                histogram = np.sum(mask, axis=0)
                midpoint = int(self.img_size_x / 2)
                L_histo = histogram[:midpoint]
                R_histo = histogram[midpoint:]

                L_sum = int(np.sum(L_histo) / 255)
                R_sum = int(np.sum(R_histo) / 255) - y

                return L_sum, midpoint, R_sum
        return 1, 1, 1

    def image_processing(self):
        x1 = int(self.img_size_x * 0.05)
        x2 = int(self.img_size_x * 0.95)
        y1 = int(self.img_size_y * 0.7)
        y2 = int(self.img_size_y * 0.9)

        self.color_ROI = self.color_img[y1:y2, x1:x2]

        l_sum, midpoint, r_sum = self.yuv_detection(self.color_ROI)
        self.L_sum = l_sum
        self.R_sum = r_sum

        shift = int((r_sum - l_sum) * 0.01)  # scale 계수 조절 가능
        target_center_x = int(self.img_size_x / 2 + shift)
        target_center_x = np.clip(target_center_x, x1, x2)

        # low pass
        alpha = 0.2  #낮을 수록 현재 가중치 감소
        self.center_x = int((1 - alpha) * self.center_x + alpha * target_center_x)
        angle_offset = (r_sum - l_sum)  
        self.get_logger().info(f"Angle Offset: {angle_offset}")

        cv2.imshow("ROI", self.color_ROI)
        cv2.line(self.color_img, (self.center_x, y1), (self.center_x, y2), (255, 0, 0), 4)
        cv2.rectangle(self.color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(self.color_img, f'L : {self.L_sum}   R : {self.R_sum}', (20, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
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

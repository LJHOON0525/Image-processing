#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray, String
from sensor_msgs.msg import Image
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge


class YellowRatioCirculator(Node):
    def __init__(self):
        super().__init__('YellowRatioCirculator')

        # QoS 설정
        qos_profile = QoSProfile(depth=10)
        img_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1
        )

        # 퍼블리셔
        self.img_publisher = self.create_publisher(Image, 'img_data', img_qos_profile)
        self.center_publisher = self.create_publisher(Float32MultiArray, 'center_x', qos_profile)
        self.direction_publisher = self.create_publisher(String, 'tracking', qos_profile)
        self.direction_msg = String()

        # 타이머
        self.capture_timer = self.create_timer(1/15, self.image_capture)
        self.process_timer = self.create_timer(1/15, self.image_processing)
        self.track_timer = self.create_timer(1/15, self.track_tracking)

        # 이미지 크기
        self.img_size_x = 848
        self.img_size_y = 480

        # 차선 합 초기화
        self.L_sum = 0
        self.R_sum = 0
        self.center_x = None

        # RealSense 설정
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.img_size_x, self.img_size_y, rs.format.bgr8, 15)
        self.config.enable_stream(rs.stream.depth, self.img_size_x, self.img_size_y, rs.format.z16, 15)
        profile = self.pipeline.start(self.config)
        self.depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.hole_filling_filter = rs.hole_filling_filter()
        self.temporal_filter = rs.temporal_filter()
        self.spatial_filter = rs.spatial_filter()
        self.cvbridge = CvBridge()

        # 화이트밸런스 수동 설정
        device = profile.get_device()
        color_sensor = device.query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
        color_sensor.set_option(rs.option.white_balance, 6500)

    # ------------------ 이미지 캡처 ------------------
    def image_capture(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        self.color_frame = aligned_frames.get_color_frame()
        self.aligned_depth_frame = aligned_frames.get_depth_frame()
        depth_filtered = self.temporal_filter.process(self.aligned_depth_frame)
        depth_filtered = self.spatial_filter.process(depth_filtered)
        self.filled_depth_frame = self.hole_filling_filter.process(depth_filtered)
        self.depth_img = np.asanyarray(self.filled_depth_frame.get_data())
        self.color_img = np.asanyarray(self.color_frame.get_data())

    # ------------------ 노란색 검출 ------------------
    def yellow_detection(self, img):
        h, w, c = img.shape
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        denoised = cv2.bilateralFilter(blur, 9, 75, 75)
        hsv_img = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)

        # 노란색 범위
        lower_yellow = np.array([18, 130, 180])
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

        # Morphology로 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 컨투어 검출
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, None, 0, None, None

        # 면적 큰 순으로 최대 2개 차선
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        left_center = right_center = None
        L_sum = R_sum = 0

        for contour in contours:
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            center = int(x + w_rect / 2)
            if center < w / 2:
                left_center = center
                L_sum = int(np.sum(mask[:, x:x+w_rect]) / 255)
            else:
                right_center = center
                R_sum = int(np.sum(mask[:, x:x+w_rect]) / 255)

            # 시각화
            cv2.drawContours(img, [contour], -1, (0, 0, 255), 2)
            cv2.line(img, (x, 0), (x, h), (0, 255, 255), 1)
            cv2.line(img, (x + w_rect, 0), (x + w_rect, h), (0, 255, 255), 1)
            cv2.line(img, (center, int(h*0.7)), (center, int(h*0.9)), (255, 0, 0), 2)

        # 두 차선 모두 있을 때만 중심 계산
        if left_center is not None and right_center is not None:
            self.center_x = int((left_center + right_center) / 2)
        elif right_center is not None and left_center is None:
            offset = int(w * 0.1)  # ROI 폭 기준 10% 왼쪽으로 이동
            self.center_x = max(right_center - offset, 0)
        else:
            self.center_x = None  # 한쪽만 있으면 중심 계산하지 않음

        return L_sum, self.center_x, R_sum, left_center, right_center

    # ------------------ 이미지 처리 ------------------
    def image_processing(self):
        if not hasattr(self, 'color_img') or self.color_img is None:
            return

        x1_roi = int(self.img_size_x * 0.05)
        y1_roi = int(self.img_size_y * 0.7)
        x2_roi = int(self.img_size_x * 0.95)
        y2_roi = int(self.img_size_y * 0.9)
        roi = self.color_img[y1_roi:y2_roi, x1_roi:x2_roi]

        self.L_sum, center, self.R_sum, left_c, right_c = self.yellow_detection(roi)

        # 중심 퍼블리시
        if center is not None:
            center_msg = Float32MultiArray()
            center_msg.data = [float(center)]
            self.center_publisher.publish(center_msg)
            cv2.line(self.color_img, (x1_roi + center, y1_roi), (x1_roi + center, y2_roi), (255, 0, 0), 2)

        # ROI 시각화
        cv2.rectangle(self.color_img, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 255, 0), 2)
        cv2.putText(self.color_img, f'L : {self.L_sum}   R : {self.R_sum}', (20, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        cv2.imshow("Color Image", self.color_img)
        cv2.imshow("ROI", roi)
        cv2.waitKey(1)

    # ------------------ 방향 판단 ------------------
    def track_tracking(self):
        detect_sum = self.L_sum + self.R_sum
        if detect_sum < 100:
            self.direction_msg.data = "WEEK FRONT"
        elif abs(self.L_sum - self.R_sum) < detect_sum * 0.32:
            self.direction_msg.data = "FRONT"
        elif self.L_sum > self.R_sum:
            self.direction_msg.data = "LEFT"
        else:
            self.direction_msg.data = "RIGHT"

        self.direction_publisher.publish(self.direction_msg)


# ------------------ 메인 ------------------
def main(args=None):
    rclpy.init(args=args)
    node = YellowRatioCirculator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

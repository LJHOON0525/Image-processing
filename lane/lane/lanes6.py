#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge

class YellowLaneWebcam(Node):
    def __init__(self):
        super().__init__('YellowLaneWebcam')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 퍼블리셔 & 브리지
        self.center_publisher = self.create_publisher(Float32MultiArray, 'center_x', 10)
        self.img_publisher = self.create_publisher(Image, 'yellow_line_image', qos_profile)
        self.cvbridge = CvBridge()

        # 웹캠 설정
        self.cap = cv2.VideoCapture(2)
        self.img_size_x = 640
        self.img_size_y = 480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size_x)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size_y)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.color_img = None
        self.R_sum = 0
        self.timer = self.create_timer(1/30.0, self.timer_callback)

        # ------------------ 프레임 분할 (픽셀 좌표) ------------------
        self.left_min_x = 31
        self.left_max_x = 320
        self.right_min_x = 320
        self.right_max_x = 608

    # ------------------ 노란색 차선 검출 ------------------
    def yellow_detection(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([18, 130, 180])
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        M = cv2.moments(mask)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            return cx, int(M['m00'])
        else:
            return None, 0

    # ------------------ 슬라이딩 윈도우 ------------------
    def sliding_window_tracking(self, img, x1_roi, x2_roi, num_windows=3, window_height=48, side="left"):
        h, _, _ = img.shape
        distances = []
        current_center = None
        area_sum = 0

        for i in range(num_windows):
            y2_roi = h - i * window_height
            y1_roi = max(0, y2_roi - window_height)
            roi = img[y1_roi:y2_roi, x1_roi:x2_roi]

            center, area = self.yellow_detection(roi)
            area_sum += area  # ROI 면적 합계
            roi_center_x = (x2_roi - x1_roi) // 2

            if center is not None:
                current_center = center
                distance = center - roi_center_x
            else:
                distance = (current_center - roi_center_x) if current_center is not None else 0

            distances.append(distance)

            # ROI 표시
            if center is not None:
                cv2.rectangle(img, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 0, 255), 2)
                cv2.line(img, (x1_roi + center, y1_roi), (x1_roi + center, y2_roi), (255, 0, 0), 2)
                cv2.putText(img, f'D:{distance}', (x1_roi + center - 10, y1_roi + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                self.get_logger().info(f'{side.capitalize()} Window {i+1} Distance: {distance}')

        return distances, area_sum

    # ------------------ 이미지 처리 ------------------
    def image_processing(self):
        if self.color_img is None:
            return

        # 오른쪽/왼쪽 차선 추적
        distances_right, area_right = self.sliding_window_tracking(
            self.color_img, self.right_min_x, self.right_max_x, side="right")
        distances_left, area_left = self.sliding_window_tracking(
            self.color_img, self.left_min_x, self.left_max_x, side="left")

        # 전체 하단 ROI 박스 (초록)
        x1_roi = int(self.img_size_x * 0.05)
        x2_roi = int(self.img_size_x * 0.95)
        y1_roi = int(self.img_size_y * 0.7)
        y2_roi = int(self.img_size_y * 1.0)
        cv2.rectangle(self.color_img, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 255, 0), 2)

        # ROI 중앙 기준선 (핑크)
        center_line_x = (x1_roi + x2_roi) // 2
        cv2.line(self.color_img, (center_line_x, y1_roi), (center_line_x, y2_roi), (255, 0, 255), 2)

        # 동적 표시: 면적 차이가 크면 큰 쪽만 표시
        display_left = display_right = True
        area_diff_threshold = 2000
        if abs(area_right - area_left) > area_diff_threshold:
            if area_right > area_left:
                display_left = False
            else:
                display_right = False

        # 왼쪽/오른쪽 ROI 구분선 표시 (조건부)
        if display_left:
            cv2.line(self.color_img, (self.left_max_x, y1_roi), (self.left_max_x, y2_roi), (255, 0, 0), 2)
        if display_right:
            cv2.line(self.color_img, (self.right_min_x, y1_roi), (self.right_min_x, y2_roi), (255, 0, 0), 2)
        # 중앙선 기준 거리 계산
        distance_right_center = ((self.right_min_x + self.right_max_x)//2 + np.mean(distances_right)) - center_line_x
        distance_left_center = ((self.left_min_x + self.left_max_x)//2 + np.mean(distances_left)) - center_line_x

        # 로그 출력
        self.get_logger().info(f'Distance to Center - Right: {distance_right_center:.1f}, Left: {distance_left_center:.1f}')

        # Float32MultiArray 퍼블리시
        msg = Float32MultiArray()
        msg.data = [float(distance_right_center), float(distance_left_center)]
        self.center_publisher.publish(msg)

        # 이미지 퍼블리시
        try:
            self.img_publisher.publish(
                self.cvbridge.cv2_to_imgmsg(self.color_img, encoding='bgr8'))
        except Exception as e:
            self.get_logger().error(f"Image publish failed: {e}")

        cv2.imshow("Yellow Lane", self.color_img)
        cv2.waitKey(1)

    # ------------------ 타이머 콜백 ------------------
    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("카메라 프레임을 읽지 못했습니다.")
            return
        self.color_img = cv2.resize(frame, (self.img_size_x, self.img_size_y))
        self.image_processing()

def main(args=None):
    rclpy.init(args=args)
    node = YellowLaneWebcam()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

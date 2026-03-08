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
        self.center_publisher = self.create_publisher(
            Float32MultiArray, 'center_x', 10)
        self.img_publisher = self.create_publisher(
            Image, 'yellow_line_image', qos_profile)
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
    def sliding_window_tracking(self, img, side='right', num_windows=6, window_height=50):
        h, w, _ = img.shape
        centers = []

        if side == 'right':
            x1_roi = int(self.img_size_x * 0.5)
            x2_roi = int(self.img_size_x * 0.95)
        else:  # left
            x1_roi = int(self.img_size_x * 0.05)
            x2_roi = int(self.img_size_x * 0.5)

        current_center = None
        for i in range(num_windows):
            y2_roi = h - i * window_height
            y1_roi = max(0, y2_roi - window_height)
            roi = img[y1_roi:y2_roi, x1_roi:x2_roi]

            center, area = self.yellow_detection(roi)
            roi_center_x = (x2_roi - x1_roi) // 2

            if center is not None:
                centers.append(center)
                current_center = center

                # ROI 시각화
                cv2.rectangle(img, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 255, 255), 1)
                cv2.line(img, (x1_roi + center, y1_roi), (x1_roi + center, y2_roi), (255, 0, 0), 2)

                distance = (x1_roi + center) - (x1_roi + roi_center_x)
                cv2.putText(img, f'D:{distance}', (x1_roi, y1_roi + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                if current_center is not None:
                    centers.append(current_center)

        return centers

    # ------------------ 이미지 처리 ------------------
    def image_processing(self):
        if self.color_img is None:
            return

        # 먼저 오른쪽 차선
        centers = self.sliding_window_tracking(self.color_img, side='right')

        if len(centers) == 0:
            # 오른쪽 차선이 없으면 왼쪽 차선 시도
            centers = self.sliding_window_tracking(self.color_img, side='left')

        if len(centers) > 0:
            center = centers[0]
            self.R_sum = 1
            center_msg = Float32MultiArray()
            center_msg.data = [float(center)]
            self.center_publisher.publish(center_msg)
        else:
            center = None
            self.R_sum = 0

        # 전체 ROI 중앙선
        x1_roi = int(self.img_size_x * 0.05)
        x2_roi = int(self.img_size_x * 0.95)
        y1_roi = int(self.img_size_y * 0.7)
        y2_roi = int(self.img_size_y * 0.9)
        roi_center_x = (x2_roi - x1_roi) // 2

        cv2.line(self.color_img, (x1_roi + roi_center_x, y1_roi),
                 (x1_roi + roi_center_x, y2_roi), (255, 0, 255), 2)
        cv2.rectangle(self.color_img, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 255, 0), 2)
        cv2.putText(self.color_img, f'OK : {self.R_sum}', (20, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

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

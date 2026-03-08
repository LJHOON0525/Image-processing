#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray, String
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge


class YellowRightLaneWebcam(Node):
    def __init__(self):
        super().__init__('YellowRightLaneWebcam')

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
        self.cap = cv2.VideoCapture(2)  # USB 카메라 2번
        self.img_size_x = 640
        self.img_size_y = 480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size_x)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size_y)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # 상태 변수
        self.color_img = None
        self.center_x = None
        self.R_sum = 0

        # 타이머 (30 FPS)
        self.timer = self.create_timer(1/30.0, self.timer_callback)

    # ------------------ 노란색 차선 검출 ------------------
    def yellow_detection(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([90, 50, 70])
        upper_yellow = np.array([130, 255, 255])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        M = cv2.moments(mask)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            return cx, int(M['m00'])
        else:
            return None, 0

    # ------------------ 슬라이딩 윈도우 방식 ------------------
    def sliding_window_tracking(self, img, num_windows=6, window_height=50):
        h, w, _ = img.shape
        centers = []

        # 시작 ROI: 맨 아래 가로폭
        x1_roi = int(self.img_size_x * 0.7)   # 오른쪽 차선만 보도록 반 화면부터
        x2_roi = int(self.img_size_x * 0.95)

        current_center = None
        for i in range(num_windows):
            # 아래에서 위로 ROI 설정
            y2_roi = h - i * window_height
            y1_roi = max(0, y2_roi - window_height)

            roi = img[y1_roi:y2_roi, x1_roi:x2_roi]
            center, area = self.yellow_detection(roi)

            if center is not None:
                centers.append(center)
                current_center = center

                # 시각화: ROI 박스 & 중심선
                cv2.rectangle(img, (x1_roi, y1_roi),
                              (x2_roi, y2_roi), (0, 255, 0), 1)
                cv2.line(img, (x1_roi + center, y1_roi),
                         (x1_roi + center, y2_roi), (255, 0, 0), 2)
            else:
                if current_center is not None:
                    centers.append(current_center)

        return centers

    # ------------------ 메인 이미지 처리 ------------------
    def image_processing(self):
        if self.color_img is None:
            return

        centers = self.sliding_window_tracking(self.color_img)

        if len(centers) > 0:
            # 가장 아래 ROI 중심을 사용 (차량 기준)
            self.center_x = centers[0]
            self.R_sum = 1

            msg = Float32MultiArray()
            msg.data = [float(self.center_x)]
            self.center_publisher.publish(msg)
        else:
            self.center_x = None
            self.R_sum = 0

        cv2.putText(self.color_img, f'Centers: {len(centers)}',
                    (20, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        try:
            self.img_publisher.publish(
                self.cvbridge.cv2_to_imgmsg(self.color_img, encoding='bgr8'))
        except Exception as e:
            self.get_logger().error(f"Image publish failed: {e}")

        cv2.imshow("Yellow Right Lane", self.color_img)
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
    node = YellowRightLaneWebcam()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

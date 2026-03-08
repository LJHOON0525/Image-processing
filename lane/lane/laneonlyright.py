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
        self.img_size_x = 640
        self.img_size_y = 480

        # 검출 값 초기화
        self.R_sum = 0
        self.center_x = None

        # 웹캠 설정
        self.cap = cv2.VideoCapture(2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size_x)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size_y)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.cvbridge = CvBridge()
        self.color_img = None

    # ------------------ 이미지 캡처 ------------------
    def image_capture(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Webcam frame read failed")
            self.color_img = None
            return
        self.color_img = frame

    # ------------------ 노란색 검출 (오른쪽만) ------------------
    def yellow_detection(self, img):
        h, w, _ = img.shape
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        denoised = cv2.bilateralFilter(blur, 9, 75, 75)
        hsv_img = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([18, 130, 180])
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0

        # 가장 큰 오른쪽 차선만 선택
        right_center = None
        R_sum = 0
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            center = int(x + w_rect / 2)
            if center > w // 2:  # 오른쪽만 사용
                right_center = center
                R_sum = int(np.sum(mask[:, x:x+w_rect]) / 255)

                # 시각화
                cv2.drawContours(img, [contour], -1, (0, 0, 255), 2)
                cv2.line(img, (x, 0), (x, h), (0, 255, 255), 1)
                cv2.line(img, (x + w_rect, 0), (x + w_rect, h), (0, 255, 255), 1)
                cv2.line(img, (center, int(h*0.7)), (center, int(h*0.9)), (255, 0, 0), 2)
                break  # 첫 번째 오른쪽 차선만 사용

        if right_center is not None:
            # 오른쪽 차선 중심에 약간 왼쪽 offset을 줘서 주행 중심 추정
            offset = int(w * 0)###여기 값줘서 offeset 가능
            self.center_x = max(right_center - offset, 0)
        else:
            self.center_x = None

        return self.center_x, R_sum

    # ------------------ 이미지 처리 ------------------
    def image_processing(self):
        if self.color_img is None:
            return

        # ROI 설정 (하단 20%)
        x1_roi = int(self.img_size_x * 0.05)
        y1_roi = int(self.img_size_y * 0.7)
        x2_roi = int(self.img_size_x * 0.95)
        y2_roi = int(self.img_size_y * 0.9)
        roi = self.color_img[y1_roi:y2_roi, x1_roi:x2_roi]

        center, self.R_sum = self.yellow_detection(roi)

        if center is not None:
            center_msg = Float32MultiArray()
            center_msg.data = [float(center)]
            self.center_publisher.publish(center_msg)
            cv2.line(self.color_img, (x1_roi + center, y1_roi),
                     (x1_roi + center, y2_roi), (255, 0, 0), 2)
            
        roi_center_x = (x2_roi - x1_roi) // 2
        cv2.line(self.color_img, (x1_roi + roi_center_x, y1_roi),
                 (x1_roi + roi_center_x, y2_roi), (255, 0, 255), 2)

        cv2.rectangle(self.color_img, (x1_roi, y1_roi),
                      (x2_roi, y2_roi), (0, 255, 0), 2)
        cv2.putText(self.color_img, f'R : {self.R_sum}',
                    (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        try:
            self.img_publisher.publish(self.cvbridge.cv2_to_imgmsg(self.color_img, encoding='bgr8'))
        except Exception as e:
            self.get_logger().error(f"Image publish failed: {e}")

        cv2.imshow("Color Image", self.color_img)
        cv2.imshow("ROI", roi)
        cv2.waitKey(1)

    # ------------------ 방향 판단 (오른쪽 기준) ------------------
    def track_tracking(self):
        if self.R_sum < 100:
            self.direction_msg.data = "LOST"
        elif self.center_x is None:
            self.direction_msg.data = "SEARCHING"
        else:
            self.direction_msg.data = "FOLLOWING RIGHT"

        self.direction_publisher.publish(self.direction_msg)

    # ------------------ 종료 처리 ------------------
    def destroy_node(self):
        super().destroy_node()
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


# ------------------ 메인 ------------------
def main(args=None):
    rclpy.init(args=args)
    node = YellowRightLaneWebcam()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

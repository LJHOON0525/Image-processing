#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class RedFlagWebcam(Node):
    def __init__(self):
        super().__init__('RedFlagWebcam')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.red_flag_pub = self.create_publisher(String, 'red_flag_detect', 10)
        self.img_publisher = self.create_publisher(Image, 'red_flag_image', qos_profile)
        self.cvbridge = CvBridge()

        # 웹캠 설정
        self.cap = cv2.VideoCapture(2)
        self.img_size_x = 640
        self.img_size_y = 480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size_x)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size_y)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.timer = self.create_timer(1/30.0, self.timer_callback)

    # ------------------ 빨간색 감지 ------------------
    def red_detection(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 빨간색 HSV 범위
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        mask = cv2.medianBlur(mask, 5)
        mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=1)

        return mask

    # ------------------ 이미지 처리 ------------------
    def image_processing(self, frame):
        mask = self.red_detection(frame)

        # 컨투어 검출
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_detected = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10000:  # 면적 기준값
                red_detected = True
                cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 2)
           
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        if red_detected:
            self.get_logger().info('RED DETECTED')
        else:
            self.get_logger().info('NO RED')

        msg = String()
        msg.data = 'RED DETECTED' if red_detected else 'NO RED'
        self.red_flag_pub.publish(msg)
        

        try:
            self.img_publisher.publish(
                self.cvbridge.cv2_to_imgmsg(frame, encoding='bgr8'))
        except Exception as e:
            self.get_logger().error(f"Image publish failed: {e}")

        cv2.imshow("Red Flag Detection", frame)
        cv2.waitKey(1)

    # ------------------ 타이머 콜백 ------------------
    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("카메라 프레임을 읽지 못했습니다.")
            return
        frame_resized = cv2.resize(frame, (self.img_size_x, self.img_size_y))
        self.image_processing(frame_resized)


def main(args=None):
    rclpy.init(args=args)
    node = RedFlagWebcam()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

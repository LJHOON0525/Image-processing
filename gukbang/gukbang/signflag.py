import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray, String, Bool
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO


class BlueRatioCirculatorWebcam(Node):
    def __init__(self):
        super().__init__('testmove_webcam')

        qos_profile = QoSProfile(depth=10)
        img_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # pub
        self.img_publisher = self.create_publisher(Image, 'img_data', img_qos_profile)
        self.center_publisher = self.create_publisher(Float32MultiArray, 'center_x', qos_profile)
        self.direction_publisher = self.create_publisher(String, 'tracking', qos_profile)
        self.direction_msg = String()

        # sub
        self.red_flag = False
        self.green_flag = False
        self.red_sub = self.create_subscription(Bool, 'red_light_flag', self._on_red_flag, qos_profile)
        self.green_sub = self.create_subscription(Bool, 'green_light_flag', self._on_green_flag, qos_profile)

        self.tracking_enabled = True  # 웹캠 테스트니까 바로 True

        # time
        self.capture_timer = self.create_timer(1/15, self.image_capture)
        self.process_timer = self.create_timer(1/15, self.image_processing)
        self.pub_control = self.create_timer(1/15, self.track_tracking)

        # parameter
        self.img_size_x = 640
        self.img_size_y = 480
        self.center_x = int(self.img_size_x / 2)
        self.L_sum = 0
        self.R_sum = 0

        # Webcam setup
        self.cap = cv2.VideoCapture(2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size_x)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size_y)
        self.cap.set(cv2.CAP_PROP_FPS, 15)

        self.cvbridge = CvBridge()

        # YOLO 모델
        self.goodbox_model = YOLO("/home/ljh/goodbox-project/train_goodbox_highacc/weights/goodbox.pt")

    # -------------------- callback--------------------
    def _on_red_flag(self, msg: Bool):
        self.red_flag = msg.data

    def _on_green_flag(self, msg: Bool):
        self.green_flag = msg.data

    # -------------------- vision--------------------
    def image_capture(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("웹캠 프레임을 읽을 수 없습니다.")
            return
        self.color_img = frame.copy()

    def yuv_detection(self, img):
        y, x, c = img.shape
        gaussian = cv2.GaussianBlur(img, (3, 3), 1)
        denoised = cv2.bilateralFilter(gaussian, d=9, sigmaColor=75, sigmaSpace=75)
        hsv_img = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([30, 70, 70])
        upper_bound = np.array([130, 255, 255])
        U_mask = cv2.inRange(hsv_img, lower_bound, upper_bound)

        contours, _ = cv2.findContours(U_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        if max_contour is None:
            return 1, 1, 1

        max_contour_mask = np.zeros_like(U_mask)
        cv2.drawContours(max_contour_mask, [max_contour], -1, 255, thickness=cv2.FILLED)
        filtered = cv2.bitwise_and(img, img, mask=max_contour_mask)

        cv2.imshow("filter", filtered)
        cv2.waitKey(1)

        x_rect, y_rect, w, h = cv2.boundingRect(max_contour)
        left = x_rect
        right = x_rect + w
        center = int((left + right) / 2)

        roi_center_x = int(x / 2)
        angle_offset = center - roi_center_x
        alpha = 0.2
        target_center_x = int((self.img_size_x * 0.05)) + center
        self.center_x = int((1 - alpha) * self.center_x + alpha * target_center_x)

        cv2.line(img, (left, 0), (left, y), (0, 255, 255), 1)
        cv2.line(img, (right, 0), (right, y), (0, 255, 255), 1)
        cv2.line(self.color_img, (self.center_x, int(self.img_size_y * 0.7)), (self.center_x, int(self.img_size_y * 0.9)), (255, 0, 0), 2)
        cv2.drawContours(img, [max_contour], -1, (0, 0, 255), 2)

        try:
            self.img_publisher.publish(self.cvbridge.cv2_to_imgmsg(filtered, encoding='bgr8'))
        except Exception as e:
            self.get_logger().error(f"Image publishing failed: {e}")

        histogram = np.sum(max_contour_mask, axis=0)
        midpoint = int(self.img_size_x / 2)
        L_histo = histogram[:midpoint]
        R_histo = histogram[midpoint:]

        L_sum = int(np.sum(L_histo) / 255)
        R_sum = int(np.sum(R_histo) / 255) - y

        return L_sum, midpoint, R_sum

    def image_processing(self):
        if not hasattr(self, 'color_img') or self.color_img is None:
            return

        x1_roi = int(self.img_size_x * 0.05)
        y1_roi = int(self.img_size_y * 0.7)
        x2_roi = int(self.img_size_x * 0.95)
        y2_roi = int(self.img_size_y * 0.9)

        roi = self.color_img[y1_roi:y2_roi, x1_roi:x2_roi]
        self.L_sum, _, self.R_sum = self.yuv_detection(roi)

        # -------------------- goodbox YOLO --------------------
        goodbox_results = self.goodbox_model.predict(self.color_img, conf=0.4, verbose=False)
        for box in goodbox_results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            u = int((x1 + x2) / 2)
            v = int((y1 + y2) / 2)

            cv2.rectangle(self.color_img, (x1, y1), (x2, y2), (136, 175, 0), 2)
            cv2.circle(self.color_img, (u, v), 4, (0, 0, 255), -1)

        cv2.line(self.color_img, (int(self.img_size_x / 2), int(self.img_size_y * 0.7)),
                 (int(self.img_size_x / 2), int(self.img_size_y * 0.9)), (0, 0, 255), 2)
        cv2.putText(self.color_img, f'L : {self.L_sum}   R : {self.R_sum}', (20, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.rectangle(self.color_img, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 255, 0), 2)

        cv2.imshow("Color", self.color_img)
        cv2.imshow("ROI", roi)
        cv2.waitKey(1)

    # -------------------- 자동 주행 --------------------
    def track_tracking(self):
        if not self.tracking_enabled:
            self.direction_msg.data = "FINISH"
            self.direction_publisher.publish(self.direction_msg)
            return
        
        if self.red_flag and not self.green_flag:
            self.direction_msg.data = "STOP"
            self.direction_publisher.publish(self.direction_msg)
            return
        elif self.green_flag:
            self.direction_msg.data = "GO"
            self.direction_publisher.publish(self.direction_msg)
            return
        
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


def main(args=None):
    rclpy.init(args=args)
    node = BlueRatioCirculatorWebcam()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt')
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

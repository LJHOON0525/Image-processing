import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray, String
from sensor_msgs.msg import Image
import pyrealsense2 as rs
import numpy as np
import cv2
import math
from cv_bridge import CvBridge

class BlueRatioCirculator(Node):
    def __init__(self):
        super().__init__('BlueRatioCirculator')

        qos_profile = QoSProfile(depth=10)
        img_qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        # Publishers
        self.img_publisher = self.create_publisher(Image, 'img_data', img_qos_profile)
        self.center_publisher = self.create_publisher(Float32MultiArray, 'center_x', qos_profile)
        self.goodbox_pub = self.create_publisher(Float32MultiArray, 'goodbox_position', qos_profile)
        self.direction_publisher = self.create_publisher(String, 'tracking', qos_profile)
        self.direction_msg = String()

        # Timers
        self.capture_timer = self.create_timer(1/15, self.image_capture)
        self.process_timer = self.create_timer(1/15, self.image_processing)
        self.pub_control = self.create_timer(1/15, self.track_tracking)

        # OCR Subscriber
        self.ocr_subscriber = self.create_subscription(
            String,
            'ocr_text',
            self.ocr_callback,
            10
        )
        self.tracking_enabled = False  # START/FINISH 플래그

        # 기본 변수
        self.U_detection_threshold = 137
        self.img_size_x = 848
        self.img_size_y = 480
        self.depth_size_x = 848
        self.depth_size_y = 480
        self.max_speed = 10
        self.odrive_mode = 1.
        self.joy_status = False
        self.joy_stick_data = [0, 0]

        self.center_x = int(self.img_size_x / 2)
        self.L_sum = 0
        self.R_sum = 0

        # RealSense 초기화
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.img_size_x, self.img_size_y, rs.format.bgr8, 15)
        self.config.enable_stream(rs.stream.depth, self.depth_size_x, self.depth_size_y, rs.format.z16, 15)
        profile = self.pipeline.start(self.config)
        self.depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.hole_filling_filter = rs.hole_filling_filter()
        self.temporal_filter = rs.temporal_filter()
        self.spatial_filter = rs.spatial_filter()

        # CV Bridge
        self.cvbridge = CvBridge()

        # 화이트밸런스 수동 설정
        device = profile.get_device()
        color_sensor = device.query_sensors()[1]  # 보통 1번 컬러 센서
        color_sensor.set_option(rs.option.enable_auto_white_balance, 0) # 자동 켜기(0이면 끄기)
        color_sensor.set_option(rs.option.white_balance, 6500) # 수동 화이트밸런스 예시

    def encoder_clear(self, msg):
        self.encoder = msg.data
        self.get_logger().info(f"Encoder L={self.encoder[0]:.2f}, R={self.encoder[1]:.2f}")

    def ocr_callback(self, msg):
        if msg.data == "START":
            self.tracking_enabled = True
            self.get_logger().info("OCR START received: Tracking enabled")
        elif msg.data == "FINISH":
            self.tracking_enabled = False
            self.get_logger().info("OCR FINISH received: Tracking disabled")

    def image_capture(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        self.color_frame = aligned_frames.get_color_frame()
        self.aligned_depth_frame = aligned_frames.get_depth_frame()
        depth_filtered = self.temporal_filter.process(self.aligned_depth_frame)
        depth_filtered = self.spatial_filter.process(depth_filtered)
        self.filled_depth_frame = self.hole_filling_filter.process(depth_filtered)
        self.depth_intrinsics = self.aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        self.depth_img = np.asanyarray(self.filled_depth_frame.get_data())
        self.color_img = np.asanyarray(self.color_frame.get_data())

    def yuv_detection(self, img):
        y, x, c = img.shape
        gaussian = cv2.GaussianBlur(img, (3, 3), 1)
        denoised = cv2.bilateralFilter(gaussian, d=9, sigmaColor=75, sigmaSpace=75) ##변경 깨지는거 방지용
        yuv_img = cv2.cvtColor(denoised, cv2.COLOR_BGR2YUV)
        _, U_img, _ = cv2.split(yuv_img)

        U_lower = 135       
        U_upper = 155
        U_mask = cv2.inRange(U_img, U_lower, U_upper)

        contours, _ = cv2.findContours(U_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        if max_contour is None:
            return 1, 1, 1 # contour 없을 때 기본값 반환

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

        #self.get_logger().info(f"Angle Offset (center deviation): {angle_offset}")

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
        
        # ROI 좌표
        x1_roi = int(self.img_size_x * 0.05)
        y1_roi = int(self.img_size_y * 0.7)
        x2_roi = int(self.img_size_x * 0.95)
        y2_roi = int(self.img_size_y * 0.9)

        # ROI 지정
        roi = self.color_img[y1_roi:y2_roi, x1_roi:x2_roi]

        self.L_sum, _, self.R_sum = self.yuv_detection(roi)

        # 중앙선, 히스토그램 텍스트, ROI 사각형 표시
        cv2.line(self.color_img, (int(self.img_size_x / 2), int(self.img_size_y * 0.7)),
                 (int(self.img_size_x / 2), int(self.img_size_y * 0.9)), (0, 0, 255), 2)
        cv2.putText(self.color_img, f'L : {self.L_sum}   R : {self.R_sum}', (20, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        
        cv2.rectangle(self.color_img, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 255, 0), 2)

        # imshow
        cv2.imshow("Color Image", self.color_img)
        cv2.imshow("ROI", roi)
        cv2.waitKey(1)

        try:
            self.img_publisher.publish(self.cvbridge.cv2_to_imgmsg(self.color_img, encoding='bgr8'))
        except Exception as e:
            self.get_logger().error(f"Image publish error: {e}")

    def track_tracking(self):
        if not self.tracking_enabled:
            self.direction_msg.data = "Finished"
            self.direction_publisher.publish(self.direction_msg)
            self.get_logger().info("Tracking stopped")
            return

        detect_sum = self.L_sum + self.R_sum
        if detect_sum < 100:
            self.get_logger().info("WEEK FRONT")
            self.direction_msg.data = "WEEK FRONT"
        elif abs(self.L_sum - self.R_sum) < detect_sum * 0.32:
            self.get_logger().info("FRONT")
            self.direction_msg.data = "FRONT"
        elif self.L_sum > self.R_sum:
            self.get_logger().info("LEFT")
            self.direction_msg.data = "LEFT"
        else:
            self.get_logger().info("RIGHT")
            self.direction_msg.data = "RIGHT"

        self.direction_publisher.publish(self.direction_msg)
        

def main(args=None):
    rclpy.init(args=args)
    node = BlueRatioCirculator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
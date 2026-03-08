import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray, String
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import Bool
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO
import math

class BlueRatioCirculator(Node):
    def __init__(self):
        super().__init__('testmove')

        qos_profile = QoSProfile(depth=10)
        img_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # pub
        self.img_publisher = self.create_publisher(Image, 'img_data', img_qos_profile)
        self.center_publisher = self.create_publisher(Float32MultiArray, 'center_x', qos_profile)
        self.goodbox_coordinate_pub = self.create_publisher(Float32MultiArray, 'goodbox_3d', qos_profile) #이거는 3차원 데이터
        self.direction_publisher = self.create_publisher(String, 'tracking', qos_profile)
        self.direction_msg = String()

        # sub
        # self.joy_subscriber = self.create_subscription(Joy, 'joy', self.joy_msg_sampling, qos_profile)
        # self.imu_subscriber = self.create_subscription(Float32MultiArray, 'imu_data', self.imu_msg_sampling, qos_profile)

        #traffic 신호등
        self.red_flag = False
        self.green_flag = False
        self.red_sub = self.create_subscription(Bool, 'red_light_flag', self._on_red_flag, qos_profile)
        self.green_sub = self.create_subscription(Bool, 'green_light_flag', self._on_green_flag, qos_profile)
       

        # coordinate sub: 라이다에서 True 받으면 3D 좌표 계산
        self.coordinate_flag = False #이거 가 받으면 YOLO 실행
        self.coordinate_subscriber = self.create_subscription(Bool, 'coordinate', self.coordinate_callback, qos_profile)
        # OCR Subscriber
        self.ocr_subscriber = self.create_subscription(String,'ocr_text',self.ocr_callback,qos_profile)
        self.tracking_enabled = False  # START/FINISH 플래그

        # time
        self.capture_timer = self.create_timer(1/15, self.image_capture)
        self.process_timer = self.create_timer(1/15, self.image_processing)
        self.pub_control = self.create_timer(1/15, self.track_tracking)

        # parameter
        self.U_detection_threshold = 135
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

        # Realsense setup
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

        self.cvbridge = CvBridge()

        # YOLO 모델
        #self.traffic_model = YOLO("'/home/ljh/goodbox-project/train_goodbox_highacc/weights/traffic.pt'") 얘기해보고
        self.goodbox_model = YOLO("/home/ljh/goodbox-project/train_goodbox_highacc/weights/goodbox.pt")

        # self.finish_ROI = [
        #     [int(self.img_size_x * 0.45), int(self.img_size_y * 0.6)],
        #     [int(self.img_size_x * 0.55), int(self.img_size_y * 0.7)]
        # ]
        # self.chess_detection_flag = False

        
        self.encoder = [0., 0.]
        self.theta = 0.0
        self.robot_roll = 0

        # # 화이트밸런스 수동 설정
        # device = profile.get_device()
        # color_sensor = device.query_sensors()[1]
        # color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
        # color_sensor.set_option(rs.option.white_balance, 3100)#이거 변경하면 찾아야됨

    # -------------------- callback--------------------
    def coordinate_callback(self, msg):
        self.coordinate_flag = msg.data
        self.get_logger().info(f"Coordinate flag: {self.coordinate_flag}")

    def encoder_clear(self, msg):
        self.encoder = msg.data
        #self.get_logger().info(f"Encoder L={self.encoder[0]:.2f}, R={self.encoder[1]:.2f}")

    # def joy_msg_sampling(self, msg):
    #     axes = msg.axes
    #     self.joy_status = axes[2] != 1
    #     if self.joy_status:
    #         self.joy_stick_data = [axes[1], axes[4]]

    # def imu_msg_sampling(self, msg):
    #     if msg.data[0] <= 77.5:
    #         self.robot_roll = -1
    #     elif msg.data[0] >= 105:
    #         self.robot_roll = 1
    #     else:
    #         self.robot_roll = 0
    #     self.theta = msg.data[1]
    #     self.get_logger().info(f"IMU Theta: {self.theta:.2f}")

    def ocr_callback(self, msg):
        if msg.data == "START":
            self.tracking_enabled = True
            self.get_logger().info("OCR START received: Tracking enabled")
        elif msg.data == "FINISH":
            self.tracking_enabled = False
            self.get_logger().info("OCR FINISH received: Tracking disabled")


    def _in_roi(self, u, v, x1_roi, y1_roi, x2_roi, y2_roi):
        return (x1_roi <= u <= x2_roi) and (y1_roi <= v <= y2_roi)
    
    def _on_red_flag(self, msg: Bool):
        self.red_flag = msg.data

    def _on_green_flag(self, msg: Bool):
        self.green_flag = msg.data

    # -------------------- vision--------------------
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
        hsv_img = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([86, 40, 70])
        upper_bound = np.array([110, 255, 255])
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

        x1_roi = int(self.img_size_x * 0.05)
        y1_roi = int(self.img_size_y * 0.7)
        x2_roi = int(self.img_size_x * 0.95)
        y2_roi = int(self.img_size_y * 0.9)

        roi = self.color_img[y1_roi:y2_roi, x1_roi:x2_roi]

        self.L_sum, _, self.R_sum = self.yuv_detection(roi)



        # # 체스판 감지
        # results = self.chess_model.predict(self.color_img, conf=0.6, verbose=False, max_det=1)
        # if results and results[0].boxes.xywh.numel() > 0:
        #     box = results[0].boxes.xywh[0].detach().cpu().numpy().astype(int)
        #     x, y, w, h = box
        #     center_x, center_y = x, y
        #     if (self.finish_ROI[0][0] < center_x < self.finish_ROI[1][0] and
        #         self.finish_ROI[0][1] < center_y < self.finish_ROI[1][1]):
        #         self.chess_detection_flag = True
        #         self.get_logger().info("Finish line detected!")
        #     x1_box, y1_box = x - w//2, y - h//2
        #     x2_box, y2_box = x + w//2, y + h//2
        #     cv2.rectangle(self.color_img, (x1_box, y1_box), (x2_box, y2_box), (0, 0, 255), 2)

        # -------------------- goodbox YOLO --------------------
        if self.coordinate_flag or True:
            goodbox_results = self.goodbox_model.predict(self.color_img, conf=0.4, verbose=False)
            for box in goodbox_results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)

                if not self._in_roi(u, v, x1_roi, y1_roi, x2_roi, y2_roi):
                     cv2.rectangle(self.color_img, (x1, y1), (x2, y2), (128, 128, 128), 1)
                     cv2.circle(self.color_img, (u, v), 3, (128, 128, 128), -1)
                     continue
                
                 
                depth = self.aligned_depth_frame.get_distance(u, v)
                point_3d = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [u, v], depth)
                X, Y, Z = point_3d
                #self.get_logger().info(f"Detected goodbox at (u,v)=({u},{v}), 3D=({X:.2f}, {Y:.2f}, {Z:.2f})")

                msg_3d = Float32MultiArray()
                msg_3d.data = [X, Y, Z, 1.0] #쿼터니언
                self.goodbox_coordinate_pub.publish(msg_3d)

                cv2.rectangle(self.color_img, (x1, y1), (x2, y2), (136, 175, 0), 2)
                cv2.circle(self.color_img, (u, v), 4, (0, 0, 255), -1)
                cv2.putText(self.color_img,f"X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f} Q:{1.0}",  
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 225, 0), 2)

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
            self.get_logger().info("Tracking stopped")
            return
        
        if self.red_flag and not self.green_flag:
            self.direction_msg.data = "STOP"
            self.direction_publisher.publish(self.direction_msg)
            self.get_logger().info("RED -> STOP")
            return
        elif self.green_flag:
            self.direction_msg.data = "GO"
            self.direction_publisher.publish(self.direction_msg)
            self.get_logger().info("GREEN -> GO")
            return
        
        else:
            self.get_logger().info("자동 추적")
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
        node.get_logger().info('Keyboard Interrupt')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
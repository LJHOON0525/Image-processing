import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge


class SpringColorChecker(Node):
    def __init__(self):
        super().__init__('spring_color_checker')

        img_qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                     history=HistoryPolicy.KEEP_LAST,
                                     depth=1)
        qos_profile = QoSProfile(depth=10)

        self.control_publisher = self.create_publisher(
            Float32MultiArray, 'Odrive_control', qos_profile)
        self.img_publisher = self.create_publisher(
            Image, 'img_data', img_qos_profile)

        self.imu_subscriber = self.create_subscription(
            Float32MultiArray,
            'imu_data',
            self.imu_msg_sampling,
            QoSProfile(depth=2))

        self.capture_timer = self.create_timer(1/15, self.image_capture)
        self.process_timer = self.create_timer(1/15, self.image_processing)

        ### Parameters ###
        self.U_detection_threshold = 130
        self.img_size_x = 848
        self.img_size_y = 480
        self.depth_size_x = 848
        self.depth_size_y = 480
        self.max_speed = 15

        ### RealSense 설정 ###
        self.get_logger().info("Trying to access RealSense camera")

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.color, self.img_size_x, self.img_size_y, rs.format.bgr8, 15)
        self.config.enable_stream(rs.stream.depth, self.depth_size_x, self.depth_size_y, rs.format.z16, 15)

        profile = self.pipeline.start(self.config)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.clipping_distance = 1.0 / self.depth_scale

        self.align = rs.align(rs.stream.color)
        self.hole_filling_filter = rs.hole_filling_filter()

        self.get_logger().info("RealSense access complete")

        ### 변수 초기화 ###
        self.L_sum = 0
        self.R_sum = 0
        self.cvbrid = CvBridge()
        self.color_ROI = np.zeros((int(self.img_size_y * 0.2), int(self.img_size_x * 0.9), 3), dtype=np.uint8)

        self.get_logger().info("Initialization complete")

    def image_capture(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        filled_depth = self.hole_filling_filter.process(depth_frame)

        if not color_frame or not depth_frame:
            self.get_logger().warn("Invalid frame received from RealSense")
            return

        self.depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        self.depth_img = np.asanyarray(filled_depth.get_data())
        self.color_img = np.asanyarray(color_frame.get_data())

    def yuv_detection(self, img):
        try:
            gaussian = cv2.GaussianBlur(img, (3, 3), 1)
            yuv_img = cv2.cvtColor(gaussian, cv2.COLOR_BGR2YUV)
            _, U_img, _ = cv2.split(yuv_img)
            ret, U_thresh = cv2.threshold(U_img, self.U_detection_threshold, 255, cv2.THRESH_BINARY)

            if not ret:
                return 1, 1, 1

            contours, _ = cv2.findContours(U_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return 1, 1, 1

            max_contour = max(contours, key=cv2.contourArea)
            max_contour_mask = np.zeros_like(U_thresh)
            cv2.drawContours(max_contour_mask, [max_contour], -1, 255, thickness=cv2.FILLED)

            filtered = cv2.bitwise_and(img, img, mask=max_contour_mask)

            # 안전한 imshow
            self.safe_imshow("Filtered U Channel", filtered)

            histogram = np.sum(max_contour_mask, axis=0)
            midpoint = self.img_size_x // 2
            L_sum = int(np.sum(histogram[:midpoint]) / 255)
            R_sum = int(np.sum(histogram[midpoint:]) / 255)

            self.img_publisher.publish(self.cvbrid.cv2_to_imgmsg(filtered, encoding='bgr8'))

            return L_sum, midpoint, R_sum
        except Exception as e:
            self.get_logger().error(f"yuv_detection error: {e}")
            return 1, 1, 1

    def image_processing(self):
        try:
            self.color_ROI = self.color_img[int(self.img_size_y * 0.7):int(self.img_size_y * 0.9),
                                            int(self.img_size_x * 0.05):int(self.img_size_x * 0.95)]

            l_sum, midpoint, r_sum = self.yuv_detection(self.color_ROI)
            self.L_sum = l_sum
            self.R_sum = r_sum

            # 이미지 표시용 요소 추가
            cv2.line(self.color_img, (self.img_size_x // 2, int(self.img_size_y * 0.7)),
                     (self.img_size_x // 2, int(self.img_size_y * 0.9)), (0, 0, 255), 2)
            cv2.rectangle(self.color_img, (int(self.img_size_x * 0.05), int(self.img_size_y * 0.7)),
                          (int(self.img_size_x * 0.95), int(self.img_size_y * 0.9)), (255, 0, 0), 2)
            cv2.putText(self.color_img, f'L : {self.L_sum}   R : {self.R_sum}', (20, 20),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

            self.safe_imshow("Color Image", self.color_img)
            self.safe_imshow("ROI", self.color_ROI)

        except Exception as e:
            self.get_logger().error(f"image_processing error: {e}")

    def imu_msg_sampling(self, msg):
        # TODO: IMU 데이터 처리
        pass

    def safe_imshow(self, window_name, img):
        try:
            cv2.imshow(window_name, img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pass  # 'q' 누르면 창 닫기 등 추가 가능
        except cv2.error as e:
            self.get_logger().warn(f"cv2.imshow 실패: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SpringColorChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.pipeline.stop()  # RealSense 종료
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

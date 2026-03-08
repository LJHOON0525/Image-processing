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
        self.U_detection_threshold = 130  # 0~255
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

        depth_profile = self.pipeline.start(self.config)

        depth_sensor = depth_profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        clipping_distance_in_meters = 1  # 1 meter
        self.clipping_distance = clipping_distance_in_meters / self.depth_scale

        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.hole_filling_filter = rs.hole_filling_filter()
        self.get_logger().info("RealSense access complete")

        ### 변수 초기화 ###
        self.L_sum = 0
        self.R_sum = 0
        self.robot_roll = 0
        self.odrive_mode = 1.
        self.joy_status = False
        self.joy_stick_data = [0, 0]
        self.before_L_joy = 0.
        self.before_R_joy = 0.

        self.cvbrid = CvBridge()

        self.color_ROI = np.zeros((int(self.img_size_y * 0.2), int(self.img_size_x * 0.9), 3), dtype=np.uint8)

        self.get_logger().info("Initialization complete")

    def image_capture(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        self.aligned_depth_frame = aligned_frames.get_depth_frame()
        self.filled_depth_frame = self.hole_filling_filter.process(self.aligned_depth_frame)

        if not color_frame:
            self.get_logger().warn("No color frame received")
            return

        self.depth_intrinsics = self.aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        self.depth_img = np.asanyarray(self.filled_depth_frame.get_data())
        self.color_img = np.asanyarray(color_frame.get_data())

    def yuv_detection(self, img):
        y, x, c = img.shape

        gaussian = cv2.GaussianBlur(img, (3, 3), 1)
        yuv_img = cv2.cvtColor(gaussian, cv2.COLOR_BGR2YUV)
        Y_img, U_img, V_img = cv2.split(yuv_img)

        ret, U_img_treated = cv2.threshold(U_img, self.U_detection_threshold, 255, cv2.THRESH_BINARY)

        if ret:
            contours, _ = cv2.findContours(U_img_treated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            max_area = 0
            max_contour = None
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    max_contour = contour

            if max_contour is not None:
                max_contour_mask = np.zeros_like(U_img_treated)
                cv2.drawContours(max_contour_mask, [max_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

                filtered = cv2.bitwise_and(img, img, mask=max_contour_mask)
                cv2.imshow("Filtered U Channel", filtered)
                cv2.waitKey(1)

                histogram = np.sum(max_contour_mask, axis=0)
                midpoint = int(self.img_size_x / 2)
                L_histo = histogram[:midpoint]
                R_histo = histogram[midpoint:]

                L_sum = int(np.sum(L_histo) / 255)
                R_sum = int(np.sum(R_histo) / 255) - y

                self.img_publisher.publish(self.cvbrid.cv2_to_imgmsg(filtered))

                return L_sum, midpoint, R_sum
        return 1, 1, 1

    def image_processing(self):
        if self.color_img is None:
            self.get_logger().warn("No image data available for processing")
            return

        self.color_ROI = self.color_img[int(self.img_size_y * 0.7):int(self.img_size_y * 0.9),
                                        int(self.img_size_x * 0.05):int(self.img_size_x * 0.95)]

        l_sum, midpoint, r_sum = self.yuv_detection(self.color_ROI)
        self.L_sum = l_sum
        self.R_sum = r_sum

        overlay = self.color_img.copy()
        height, width, _ = overlay.shape
        
        box_top_left = (width // 20, int(height * 0.65))
        box_bottom_right = (width * 19 // 20, height - 20)
        cv2.rectangle(overlay, box_top_left, box_bottom_right, (0, 255, 0), 2)
        
        line_y = height - 80
        center_x = width // 2
        offset = 210
        left_x1 = center_x - offset - 100
        left_x2 = center_x - offset
        right_x1 = center_x + offset
        right_x2 = center_x + offset + 100
        cv2.line(overlay, (left_x1, line_y), (left_x2, line_y), (0, 255, 255), 4)
        cv2.line(overlay, (right_x1, line_y), (right_x2, line_y), (0, 255, 255), 4)

        cv2.line(overlay, (center_x, line_y - 20), (center_x, line_y + 20), (255, 255, 255), 2)
        cv2.line(overlay, (center_x - 20, line_y), (center_x + 20, line_y), (255, 255, 255), 2)

        cv2.addWeighted(overlay, 0.7, self.color_img, 0.3, 0, self.color_img)

        cv2.putText(self.color_img, f'L : {self.L_sum}   R : {self.R_sum}', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("ROI",self.color_ROI)

        cv2.imshow("Color Image", self.color_img)
        cv2.waitKey(1)

    def imu_msg_sampling(self, msg):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = SpringColorChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

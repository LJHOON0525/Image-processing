import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, Joy
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO
import math

# --- PID Controller Class ---
class PIDController:
    def __init__(self, kp, ki, kd, output_min, output_max):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        self.output_min = output_min
        self.output_max = output_max

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return np.clip(output, self.output_min, self.output_max)

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

# --- BlueRatioCirculator Node ---
class BlueRatioCirculator(Node):
    def __init__(self):
        super().__init__('blue_ratio_circulator')

        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('img_size_x', 848),
                ('img_size_y', 480),
                ('depth_size_x', 848),
                ('depth_size_y', 480),
                ('U_detection_threshold', 140),
                ('max_speed', 10.0),
                ('roi_y_min_ratio', 0.7),
                ('roi_y_max_ratio', 0.9),
                ('roi_x_min_ratio', 0.05),
                ('roi_x_max_ratio', 0.95),
                ('chess_model_path', '/home/ljh/goodbox-project/train_goodbox_highacc/weights/chess.pt'),
                ('goodbox_model_path', '/home/ljh/goodbox-project/train_goodbox_highacc/weights/goodbox.pt'),
                ('pid_kp', 0.005), # Adjusted Kp for smoother control
                ('pid_ki', 0.0),
                ('pid_kd', 0.0005), # Adjusted Kd
                ('imu_correction_decay', 0.9),
                ('imu_correction_gain', 0.01),
                ('max_imu_correction', 0.5),
                ('line_lost_speed_ratio', 0.3), # Speed when line is lost
                ('line_straight_speed_ratio', 0.5), # Speed when going straight
                ('line_turn_speed_ratio_slow', 0.2), # Speed for slow turn
                ('line_turn_speed_ratio_fast', 0.6), # Speed for fast turn
                ('line_diff_threshold_ratio', 0.1) # Threshold for straight mode
            ]
        )

        # Retrieve parameters
        self.img_size_x = self.get_parameter('img_size_x').value
        self.img_size_y = self.get_parameter('img_size_y').value
        self.depth_size_x = self.get_parameter('depth_size_x').value
        self.depth_size_y = self.get_parameter('depth_size_y').value
        self.U_detection_threshold = self.get_parameter('U_detection_threshold').value
        self.max_speed = self.get_parameter('max_speed').value
        self.roi_y_min_ratio = self.get_parameter('roi_y_min_ratio').value
        self.roi_y_max_ratio = self.get_parameter('roi_y_max_ratio').value
        self.roi_x_min_ratio = self.get_parameter('roi_x_min_ratio').value
        self.roi_x_max_ratio = self.get_parameter('roi_x_max_ratio').value
        self.chess_model_path = self.get_parameter('chess_model_path').value
        self.goodbox_model_path = self.get_parameter('goodbox_model_path').value
        self.pid_kp = self.get_parameter('pid_kp').value
        self.pid_ki = self.get_parameter('pid_ki').value
        self.pid_kd = self.get_parameter('pid_kd').value
        self.imu_correction_decay = self.get_parameter('imu_correction_decay').value
        self.imu_correction_gain = self.get_parameter('imu_correction_gain').value
        self.max_imu_correction = self.get_parameter('max_imu_correction').value
        self.line_lost_speed_ratio = self.get_parameter('line_lost_speed_ratio').value
        self.line_straight_speed_ratio = self.get_parameter('line_straight_speed_ratio').value
        self.line_turn_speed_ratio_slow = self.get_parameter('line_turn_speed_ratio_slow').value
        self.line_turn_speed_ratio_fast = self.get_parameter('line_turn_speed_ratio_fast').value
        self.line_diff_threshold_ratio = self.get_parameter('line_diff_threshold_ratio').value


        # QoS Profiles
        qos_profile_default = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        qos_profile_img = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5, # Increased depth for images to reduce loss
            durability=DurabilityPolicy.VOLATILE
        )

        # Publishers
        self.control_publisher = self.create_publisher(Float32MultiArray, 'Odrive_control', qos_profile_default)
        self.img_publisher = self.create_publisher(Image, 'img_data', qos_profile_img)
        self.center_publisher = self.create_publisher(Float32MultiArray, 'center_x', qos_profile_default) # Not currently used, consider removing or using

        # Subscribers
        self.joy_subscriber = self.create_subscription(Joy, 'joy', self.joy_msg_sampling, qos_profile_default)
        self.imu_subscriber = self.create_subscription(Float32MultiArray, 'imu_data', self.imu_msg_sampling, qos_profile_default)
        self.encoder_subscriber = self.create_subscription(Float32MultiArray, 'Odrive_encoder', self.encoder_clear, qos_profile_default)

        # Timers
        self.capture_timer = self.create_timer(1/15, self.image_capture) # 15 FPS capture
        self.process_timer = self.create_timer(1/15, self.image_processing) # 15 FPS processing
        self.pub_control = self.create_timer(1/15, self.track_tracking) # 15 FPS control loop

        # Control & State Variables
        self.odrive_mode = 1.0 # Assuming mode 1 is active control
        self.joy_status = False
        self.joy_stick_data = [0.0, 0.0] # [left_stick_y, right_stick_y]

        self.center_x_detected = int(self.img_size_x / 2) # Center of detected blue line
        self.L_sum = 0
        self.R_sum = 0

        self.chess_detection_flag = False
        self.finish_ROI = [[int(self.img_size_x * 0.45), int(self.img_size_y * 0.6)],
                           [int(self.img_size_x * 0.55), int(self.img_size_y * 0.7)]]

        self.encoder = [0.0, 0.0]
        self.theta = 0.0 # IMU Yaw
        self.robot_roll = 0 # -1: left tilt, 0: straight, 1: right tilt

        # IMU correction variables
        self.imu_correction_L = 0.0
        self.imu_correction_R = 0.0

        # PID controller for steering
        self.target_center_x = int(self.img_size_x * 0.5) # Ideal center of the line in the image
        self.pid_controller = PIDController(self.pid_kp, self.pid_ki, self.pid_kd, -self.max_speed, self.max_speed)
        self.prev_time_control = self.get_clock().now()

        # Camera & Vision Setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.cvbridge = CvBridge()
        self.color_frame = None
        self.aligned_depth_frame = None

        try:
            self.config.enable_stream(rs.stream.color, self.img_size_x, self.img_size_y, rs.format.bgr8, 15)
            self.config.enable_stream(rs.stream.depth, self.depth_size_x, self.depth_size_y, rs.format.z16, 15)
            profile = self.pipeline.start(self.config)
            self.depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = self.depth_sensor.get_depth_scale()
            self.align = rs.align(rs.stream.color)
            self.hole_filling_filter = rs.hole_filling_filter()
            self.temporal_filter = rs.temporal_filter()
            self.spatial_filter = rs.spatial_filter()

            # Enable auto white balance for color sensor
            color_sensor = profile.get_device().query_sensors()[1]
            color_sensor.set_option(rs.option.enable_auto_white_balance, 1)

            self.get_logger().info("RealSense camera initialized successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize RealSense camera: {e}")
            self.pipeline = None # Indicate failure

        # Load YOLO models
        try:
            self.chess_model = YOLO(self.chess_model_path)
            self.goodbox_model = YOLO(self.goodbox_model_path)
            self.get_logger().info("YOLO models loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO models: {e}")
            self.chess_model = None
            self.goodbox_model = None

        self.get_logger().info("Node initialized and ready.")

    def encoder_clear(self, msg):
        """Callback for encoder data. Currently just logs the values."""
        self.encoder = msg.data
        # self.get_logger().info(f"Encoder L={self.encoder[0]:.2f}, R={self.encoder[1]:.2f}") # Too verbose

    def image_capture(self):
        """Captures color and depth frames from RealSense and applies filters."""
        if self.pipeline is None:
            return

        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            self.color_frame = aligned_frames.get_color_frame()
            self.aligned_depth_frame = aligned_frames.get_depth_frame()

            if not self.color_frame or not self.aligned_depth_frame:
                self.get_logger().warn("Failed to get color or depth frame.")
                return

            # Apply depth filters
            depth_filtered = self.temporal_filter.process(self.aligned_depth_frame)
            depth_filtered = self.spatial_filter.process(depth_filtered)
            self.filled_depth_frame = self.hole_filling_filter.process(depth_filtered)
            
            # Convert to numpy arrays
            self.depth_img = np.asanyarray(self.filled_depth_frame.get_data())
            self.color_img = np.asanyarray(self.color_frame.get_data())
            
            self.depth_intrinsics = self.aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        except rs.error as e:
            self.get_logger().error(f"RealSense error during frame capture: {e}")
        except Exception as e:
            self.get_logger().error(f"Error during frame capture: {e}")


    def yuv_detection(self, img_roi):
        """
        Detects blue line in the given ROI using YUV color space.
        Returns L_sum, R_sum (blue pixel counts) and the x-coordinate of the line center.
        """
        if img_roi is None or img_roi.size == 0:
            return 0, 0, self.img_size_x // 2 # Return default values if ROI is empty

        h, w, _ = img_roi.shape

        # Apply Gaussian blur for noise reduction
        gaussian = cv2.GaussianBlur(img_roi, (5, 5), 0) # Increased kernel size for smoother blur
        yuv_img = cv2.cvtColor(gaussian, cv2.COLOR_BGR2YUV)
        _, U_img, _ = cv2.split(yuv_img)
        
        # Apply adaptive thresholding for U channel
        U_img_treated = cv2.adaptiveThreshold(U_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 11, 2) # Inverted for blue detection

        contours, _ = cv2.findContours(U_img_treated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        center_x_line = self.img_size_x // 2 # Default to center if no line detected
        L_sum_roi = 0
        R_sum_roi = 0
        
        # For visualization purposes, create a copy of the original color image
        display_img = self.color_img.copy() 

        if max_contour is not None and max_area > 50: # Minimum area threshold
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                center_x_line = int(M["m10"] / M["m00"]) + int(self.img_size_x * self.roi_x_min_ratio) # Adjust for ROI offset

            # Draw the contour on the original color image for full view display
            cv2.drawContours(display_img, [max_contour + np.array([int(self.img_size_x * self.roi_x_min_ratio), int(self.img_size_y * self.roi_y_min_ratio)])], -1, (0, 0, 255), 2)
            cv2.circle(display_img, (center_x_line, int(self.img_size_y * (self.roi_y_min_ratio + self.roi_y_max_ratio) / 2)), 5, (255, 0, 0), -1)

            # Calculate L_sum and R_sum from the ROI mask
            max_contour_mask = np.zeros_like(U_img_treated)
            cv2.drawContours(max_contour_mask, [max_contour], -1, 255, thickness=cv2.FILLED)
            
            # Calculate pixel sums for line tracking
            # Histogram across the width of the ROI
            histogram = np.sum(max_contour_mask, axis=0) 
            midpoint_roi = w // 2 # Midpoint of the ROI
            
            L_histo_roi = histogram[:midpoint_roi]
            R_histo_roi = histogram[midpoint_roi:]

            L_sum_roi = int(np.sum(L_histo_roi) / 255) # Normalize by 255 (pixel value)
            R_sum_roi = int(np.sum(R_histo_roi) / 255)

        else:
            self.get_logger().debug("No significant blue line contour found in ROI.")
        
        # Publish the processed image for visualization
        try:
            self.img_publisher.publish(self.cvbridge.cv2_to_imgmsg(display_img, encoding='bgr8'))
        except Exception as e:
            self.get_logger().error(f"Image publishing failed: {e}")

        self.center_x_detected = center_x_line # Store the detected center x
        self.L_sum = L_sum_roi
        self.R_sum = R_sum_roi
        return L_sum_roi, R_sum_roi, center_x_line


    def image_processing(self):
        """Processes the captured image for line and object detection."""
        if self.color_img is None or self.depth_img is None:
            return

        # Define ROI based on parameters
        x1_roi = int(self.img_size_x * self.roi_x_min_ratio)
        y1_roi = int(self.img_size_y * self.roi_y_min_ratio)
        x2_roi = int(self.img_size_x * self.roi_x_max_ratio)
        y2_roi = int(self.img_size_y * self.roi_y_max_ratio)

        roi = self.color_img[y1_roi:y2_roi, x1_roi:x2_roi]
        
        # Perform YUV detection on ROI
        self.yuv_detection(roi)

        # Chessboard detection
        if self.chess_model is not None:
            results = self.chess_model.predict(self.color_img, conf=0.6, verbose=False, max_det=1)
            if results and len(results[0].boxes.xywh) > 0:
                box = results[0].boxes.xywh[0].detach().cpu().numpy().astype(int)
                x_center_chess, y_center_chess, w_chess, h_chess = box
                
                # Check if chessboard center is within finish ROI
                if (self.finish_ROI[0][0] < x_center_chess < self.finish_ROI[1][0] and
                    self.finish_ROI[0][1] < y_center_chess < self.finish_ROI[1][1]):
                    self.chess_detection_flag = True
                    self.get_logger().info("Finish line (chessboard) detected!")

                x1_box, y1_box = x_center_chess - w_chess // 2, y_center_chess - h_chess // 2
                x2_box, y2_box = x_center_chess + w_chess // 2, y_center_chess + h_chess // 2
                cv2.rectangle(self.color_img, (x1_box, y1_box), (x2_box, y2_box), (0, 0, 255), 2)
            else:
                self.chess_detection_flag = False # Reset if chessboard not detected or out of ROI

        # Goodbox detection
        if self.goodbox_model is not None and self.aligned_depth_frame is not None:
            goodbox_results = self.goodbox_model.predict(self.color_img, conf=0.6, verbose=False)
            for box in goodbox_results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)
                
                # Ensure pixel is within depth image bounds
                if 0 <= u < self.depth_size_x and 0 <= v < self.depth_size_y:
                    depth = self.aligned_depth_frame.get_distance(u, v)
                    if depth > 0: # Check if depth is valid
                        point_3d = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [u, v], depth)
                        X, Y, Z = point_3d
                        self.get_logger().info(f"Detected goodbox at (u,v)=({u},{v}), 3D=({X:.2f}, {Y:.2f}, {Z:.2f})m")
                
                cv2.rectangle(self.color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(self.color_img, "goodbox", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw ROI on the original image for visualization
        cv2.rectangle(self.color_img, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 255, 0), 2)
        
        # Display line sums and target center line
        cv2.putText(self.color_img, f'L : {self.L_sum}   R : {self.R_sum}', (20, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.line(self.color_img, (self.target_center_x, y1_roi), (self.target_center_x, y2_roi), (0, 0, 255), 2)

        # Update and display center_x_detected for debugging
        # The center_x_detected is in ROI coordinates, need to convert to full image coordinates
        adjusted_center_x_detected = self.center_x_detected + int(self.img_size_x * self.roi_x_min_ratio)
        cv2.line(self.color_img, (adjusted_center_x_detected, y1_roi), (adjusted_center_x_detected, y2_roi), (255, 0, 0), 2)
        cv2.putText(self.color_img, f'Center_X: {adjusted_center_x_detected}', (20, 40),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        
        # Show images
        cv2.imshow("Color Image with Detections", self.color_img)
        # cv2.imshow("ROI for Line Detection (Processed U Channel)", U_img_treated) # Debug U_img_treated
        cv2.waitKey(1)

    def track_tracking(self):
        """Controls robot movement based on line detection, object detection, and IMU."""
        msg = Float32MultiArray()
        
        current_time_control = self.get_clock().now()
        dt_control = (current_time_control - self.prev_time_control).nanoseconds / 1e9
        self.prev_time_control = current_time_control

        if self.chess_detection_flag:
            self.L_joy = 0.0
            self.R_joy = 0.0
            self.pid_controller.reset() # Reset PID when stopped
            self.get_logger().info("Chessboard detected - Stopping robot.")
        elif self.joy_status:
            self.L_joy = self.joy_stick_data[0] * self.max_speed
            self.R_joy = self.joy_stick_data[1] * self.max_speed
            self.pid_controller.reset() # Reset PID when manual control
            self.get_logger().info(f"Manual Control - L: {self.L_joy:.2f}, R: {self.R_joy:.2f}")
        else:
            # Automatic tracking
            detect_sum = self.L_sum + self.R_sum
            
            # Line tracking PID control
            # Error is the difference between target center and detected line center
            # Normalize error based on ROI width to make PID general
            roi_width = int(self.img_size_x * (self.roi_x_max_ratio - self.roi_x_min_ratio))
            # The detected center_x_detected is already relative to the ROI
            error = (self.target_center_x - (self.center_x_detected + int(self.img_size_x * self.roi_x_min_ratio))) / roi_width * 2 # Normalize to -1 to 1 range
            
            steering_output = self.pid_controller.compute(error, dt_control)
            
            base_speed = 0.0
            if detect_sum < 100: # Line almost lost or very small
                base_speed = self.max_speed * self.line_lost_speed_ratio
                self.get_logger().info("Line almost lost - Moving slowly forward.")
            elif abs(self.L_sum - self.R_sum) < detect_sum * self.line_diff_threshold_ratio:
                base_speed = self.max_speed * self.line_straight_speed_ratio
                self.get_logger().info("Straight mode.")
            else:
                # If we are turning, ensure a reasonable base speed
                base_speed = self.max_speed * self.line_straight_speed_ratio 
            
            # Apply steering output to motor speeds
            # Positive steering_output means robot needs to turn left (left wheel faster, right wheel slower)
            # Negative steering_output means robot needs to turn right (right wheel faster, left wheel slower)
            self.L_joy = base_speed - steering_output
            self.R_joy = base_speed + steering_output

            # IMU-based correction
            if self.robot_roll == -1: # Leaning left, need to increase right wheel speed
                self.imu_correction_L = max(self.imu_correction_L - self.imu_correction_gain, -self.max_imu_correction)
                self.imu_correction_R = min(self.imu_correction_R + self.imu_correction_gain, self.max_imu_correction)
                self.get_logger().debug(f"IMU Roll -1: L correction {self.imu_correction_L:.2f}, R correction {self.imu_correction_R:.2f}")
            elif self.robot_roll == 1: # Leaning right, need to increase left wheel speed
                self.imu_correction_L = min(self.imu_correction_L + self.imu_correction_gain, self.max_imu_correction)
                self.imu_correction_R = max(self.imu_correction_R - self.imu_correction_gain, -self.max_imu_correction)
                self.get_logger().debug(f"IMU Roll 1: L correction {self.imu_correction_L:.2f}, R correction {self.imu_correction_R:.2f}")
            else: # Straight, decay correction
                self.imu_correction_L *= self.imu_correction_decay
                self.imu_correction_R *= self.imu_correction_decay
                self.get_logger().debug(f"IMU Roll 0: L correction decay {self.imu_correction_L:.2f}, R correction decay {self.imu_correction_R:.2f}")

            # Apply IMU correction and clamp speeds
            self.L_joy = np.clip(self.L_joy + self.imu_correction_L, -self.max_speed, self.max_speed)
            self.R_joy = np.clip(self.R_joy + self.imu_correction_R, -self.max_speed, self.max_speed)
            
            self.get_logger().info(f"Auto Control - L_sum:{self.L_sum}, R_sum:{self.R_sum}, Error:{error:.4f}, Steering:{steering_output:.2f}, Speed L: {self.L_joy:.2f}, R: {self.R_joy:.2f}")

        msg.data = [self.odrive_mode, self.L_joy, self.R_joy]
        self.control_publisher.publish(msg)

    def joy_msg_sampling(self, msg):
        """Processes joystick input for manual control."""
        axes = msg.axes
        # Axis 2 for enable/disable manual control (typically a trigger)
        # Assuming trigger value of 1 means auto, anything else means manual
        self.joy_status = axes[2] != 1.0 # If trigger is not pressed (value not 1.0), then joy is active
        if self.joy_status:
            # Axis 1 for left stick Y (forward/backward for left wheel)
            # Axis 4 for right stick Y (forward/backward for right wheel)
            self.joy_stick_data = [axes[1], axes[4]]
            self.get_logger().debug(f"Joystick active. L-stick Y: {axes[1]:.2f}, R-stick Y: {axes[4]:.2f}")
        else:
            self.get_logger().debug("Joystick inactive (auto mode).")


    def imu_msg_sampling(self, msg):
        """Processes IMU data for robot's roll angle."""
        # Assuming msg.data[0] is the roll angle
        # Define roll thresholds for leaning left/right
        roll_threshold_left = 85.0 # Example: if roll < 85 degrees, robot is leaning left
        roll_threshold_right = 95.0 # Example: if roll > 95 degrees, robot is leaning right

        if msg.data[0] < roll_threshold_left:
            self.robot_roll = -1 # Leaning left
        elif msg.data[0] > roll_threshold_right:
            self.robot_roll = 1 # Leaning right
        else:
            self.robot_roll = 0 # Relatively straight

        self.theta = msg.data[1] # Assuming msg.data[1] is yaw or other relevant angle
        # self.get_logger().info(f"IMU Roll: {msg.data[0]:.2f}, Yaw: {self.theta:.2f}, Robot Roll State: {self.robot_roll}") # Too verbose

    def destroy_node(self):
        """Clean up resources before node destruction."""
        if self.pipeline:
            self.pipeline.stop()
            self.get_logger().info("RealSense pipeline stopped.")
        cv2.destroyAllWindows()
        self.get_logger().info("OpenCV windows destroyed.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BlueRatioCirculator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (Ctrl-C detected)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
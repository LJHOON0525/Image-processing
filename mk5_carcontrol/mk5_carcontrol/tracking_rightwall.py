# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray, Float32
from sensor_msgs.msg import LaserScan
import numpy as np

class WallFollower(Node):
    def __init__(self):
        super().__init__('wall_follower')

        qos_profile_default = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        qos_profile_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        
        self.control_publisher = self.create_publisher(Float32MultiArray, 'Odrive_control', qos_profile_reliable)
        self.steering_publisher = self.create_publisher(Float32, '/steering_angle', qos_profile_reliable)
        
        self.wall_equation_sub = self.create_subscription(
            Float32MultiArray,
            '/wall_equation',
            self.wall_equation_callback,
            qos_profile_default
        )
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_default
        )

        # --------------- Parameters ---------------
        self.car_width = 0.78 
        self.car_length = 0.53 
        
        # --- 수정된 부분: 목표 벽면 거리를 양수 값으로 설정 ---
        # WallDetector가 오른쪽 벽을 y > 0 영역으로 감지하므로, 목표 거리도 양수입니다.
        self.desired_wall_distance = 1.39
        self.base_speed_straight = 6.0             
        self.base_speed_corner = 3.0               
        self.max_steering_angle = np.radians(30.0) 
        self.min_steering_angle = np.radians(-30.0) 

        # --- PID Controller Gains for Steering ---
        self.kp_distance = 2.0
        self.kp_angle = 12.0
        self.ki_distance = 0.01
        self.kd_distance = 0.5
        
        self.straight_slope_threshold = 0.5
        
        self.last_distance_error = 0.0
        self.integral_distance_error = 0.0
        self.last_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec / 1e9

        self.wall_m = 0.0
        self.wall_b = 0.0
        
        self.driving_mode = "WALL_FOLLOWING"
        self.corner_detection_ratio = 2.0 

        # --- 수정된 부분: kp_bisection의 부호를 반전 ---
        self.kp_bisection = -5.0 
        self.left_ll_coords = None
        self.right_ll_coords = None

        self.get_logger().info("WallFollower node initialized with Mixed-Mode.")

    def wall_equation_callback(self, msg: Float32MultiArray):
        if len(msg.data) == 2:
            self.wall_m = msg.data[0]
            self.wall_b = msg.data[1]
        else:
            self.get_logger().warn("Received invalid wall_equation message.")

    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        ranges[~np.isfinite(ranges)] = msg.range_max 

        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        angles_deg = np.degrees(angles)
        
        # --- 수정된 부분: 코너 감지 로직의 각도 범위 ---
        # 로봇의 정면(180도) 기준 왼쪽 벽과 오른쪽 벽을 감지합니다.
        
        # 1. 왼쪽 벽의 가장 먼 점 (LL) 찾기
        left_scan_indices = np.where( (angles_deg >= 180.0) & (angles_deg <= 270.0) )[0] 
        # 2. 오른쪽 벽의 가장 먼 점 (LR) 찾기
        right_scan_indices = np.where( (angles_deg >= 90.0) & (angles_deg <= 180.0))[0]

        if len(left_scan_indices) > 0 and len(right_scan_indices) > 0:
            ll_idx = left_scan_indices[np.argmax(ranges[left_scan_indices])]
            lr_idx = right_scan_indices[np.argmax(ranges[right_scan_indices])]
            
            ll_val = ranges[ll_idx]
            lr_val = ranges[lr_idx]

            ll_neighbor_idx = ll_idx + 1 if ll_idx + 1 < len(ranges) else ll_idx
            if ranges[ll_neighbor_idx] > 0.1 and ll_val / ranges[ll_neighbor_idx] > self.corner_detection_ratio:
                self.get_logger().info("Corner detected! Switching to Triangle Bisection Mode.")
                self.driving_mode = "TRIANGLE_BISECTION"
                self.left_ll_coords = (ll_val * np.cos(angles[ll_idx]), ll_val * np.sin(angles[ll_idx]))
                self.right_ll_coords = (lr_val * np.cos(angles[lr_idx]), lr_val * np.sin(angles[lr_idx]))
        
        # 코너를 벗어났는지 확인
        elif self.driving_mode == "TRIANGLE_BISECTION" and (
            (self.left_ll_coords and np.linalg.norm(self.left_ll_coords) > 5.0) or
            (self.right_ll_coords and np.linalg.norm(self.right_ll_coords) > 5.0)):
            self.driving_mode = "WALL_FOLLOWING"
            self.get_logger().info("Exiting Corner. Switching back to Wall Following Mode.")

        self.control_robot()

    def control_robot(self):
        current_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec / 1e9
        dt = current_time - self.last_time
        if dt == 0:
            dt = 0.001

        steering_angle = 0.0
        base_speed = self.base_speed_straight

        if self.driving_mode == "TRIANGLE_BISECTION" and self.left_ll_coords and self.right_ll_coords:
            base_speed = self.base_speed_corner

            mid_x = (self.left_ll_coords[0] + self.right_ll_coords[0]) / 2.0
            mid_y = (self.left_ll_coords[1] + self.right_ll_coords[1]) / 2.0
            
            # --- 수정된 부분: 180도를 기준으로 각도 오차 계산 ---
            # np.arctan2(y, x)는 0도를 기준으로 각도를 반환합니다. 
            # 로봇의 정면은 180도이므로, (mid_x, mid_y)가 로봇의 정면을 향할 때 180도가 되도록 오차를 계산합니다.
            angle_error = np.arctan2(mid_y, mid_x) - np.pi
            
            # steering_angle = self.kp_bisection * angle_error
            # self.kp_bisection을 음수로 설정하여 조향 명령의 부호를 반전시켰습니다.
            steering_angle = self.kp_bisection * angle_error
            
            self.get_logger().info(f'Triangle Bisection Mode | Angle Error: {np.degrees(angle_error):.2f} deg | Steering: {np.degrees(steering_angle):.2f} deg')

        elif self.driving_mode == "WALL_FOLLOWING":
            distance_error = self.wall_b - self.desired_wall_distance
            angle_error = self.wall_m

            if abs(angle_error) <= self.straight_slope_threshold:
                steering_angle = 0.0
                self.integral_distance_error = 0.0
                self.get_logger().info(f'Wall Following (Straight Mode) | Wall: m={self.wall_m:.2f}, b={self.wall_b:.2f} | Steering: {np.degrees(steering_angle):.2f} deg')
            else:
                self.integral_distance_error += distance_error * dt
                derivative_distance_error = (distance_error - self.last_distance_error) / dt

                # --- 수정된 부분: 조향 명령의 부호 변경 ---
                # distance_error와 angle_error가 양수일 때, 로봇은 오른쪽으로 조향해야 합니다.
                steering_output = -(self.kp_distance * distance_error + self.ki_distance * self.integral_distance_error + self.kd_distance * derivative_distance_error) \
                                - (self.kp_angle * angle_error)
                steering_angle = np.clip(steering_output, self.min_steering_angle, self.max_steering_angle)
                self.get_logger().info(f'Wall Following (PID Mode) | Wall: m={self.wall_m:.2f}, b={self.wall_b:.2f} | Dist Error: {distance_error:.2f}, Ang Error: {angle_error:.2f} | Steering: {np.degrees(steering_angle):.2f} deg')

            self.last_distance_error = distance_error

        self.last_time = current_time

        final_steering_angle = np.clip(steering_angle, self.min_steering_angle, self.max_steering_angle)

        left_wheel_speed = base_speed + final_steering_angle * 0.9
        right_wheel_speed = base_speed - final_steering_angle * 0.9
        print(left_wheel_speed,right_wheel_speed)
        
        odrive_msg = Float32MultiArray()
        odrive_msg.data = [1.0,float(left_wheel_speed), float(right_wheel_speed)]
        self.control_publisher.publish(odrive_msg)

        steering_msg = Float32()
        steering_msg.data = float(final_steering_angle)
        self.steering_publisher.publish(steering_msg)

def main(args=None):
    rclpy.init(args=args)
    node = WallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
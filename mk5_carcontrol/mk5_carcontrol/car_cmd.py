#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String  
import odrive
from odrive.enums import CONTROL_MODE_VELOCITY_CONTROL, INPUT_MODE_VEL_RAMP, AXIS_STATE_FULL_CALIBRATION_SEQUENCE, AXIS_STATE_IDLE, AXIS_STATE_CLOSED_LOOP_CONTROL
import time
import math
from tf_transformations import quaternion_from_euler
import tf2_ros

class ODriveMobileBase(Node):
    def __init__(self):
        super().__init__('odrive_mobile_base')

        # --- ODrive SET ---
        self.car_drive = odrive.find_any(serial_number="3678387D3333")
        self.get_logger().info("ODrive connected!")
        self.calibration()

        # --- 차체 파라미터 SET !! ---
        self.wheel_base = 0.465  # 단위 m
        self.CPR = 2000           
        self.wheel_radius = 0.06 

        # --- ODrive Initializing => Ramped Velocity----- 
        for axis in [self.car_drive.axis0, self.car_drive.axis1]:
            axis.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
            axis.controller.config.vel_ramp_rate = 5
            axis.controller.config.input_mode = INPUT_MODE_VEL_RAMP

        # --- Encoder ---> count_in_cpr (안되면 pos_estimate로 해보기)
        self.prev_count0 = self.car_drive.axis0.encoder.count_in_cpr
        self.prev_count1 = self.car_drive.axis1.encoder.count_in_cpr
        self.total_count0 = 0
        self.total_count1 = 0

        # --- 로봇 위치 초기화 ---
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        qos_profile = QoSProfile(depth=10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', qos_profile)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, qos_profile)

        # --- TF 브로드캐스터 ---
        self.odom_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.turn_flag = None  
        self.turn_sub = self.create_subscription(String, '/turn_dir', self.turn_callback, 10)


        self.create_timer(0.1, self.publish_odometry)

    def calibration(self):
        self.get_logger().info('-----CALIBRATION START-----')

        for axis, name in zip([self.car_drive.axis0, self.car_drive.axis1], ['RIGHT', 'LEFT']):
            axis.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
            while axis.current_state != AXIS_STATE_IDLE:
                time.sleep(0.1)
            axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
            self.get_logger().info(f'{name} Wheel calibration Complete!')

    def turn_callback(self, msg: String):
        self.turn_flag = msg.data
        self.get_logger().info(f"Turn flag set: {self.turn_flag}")

    def cmd_vel_callback(self, msg: Twist):
        linear = msg.linear.x
        angular = msg.angular.z

#         # diff drive 계산
        vel_left = linear + (angular * self.wheel_base / 2)
        vel_right = linear - (angular * self.wheel_base / 2)

#         self.car_drive.axis0.controller.input_vel = -vel_right
#         self.car_drive.axis1.controller.input_vel = vel_left

#         self.get_logger().info(f'Cmd_vel -> Left: {vel_left:.2f}, Right: {-vel_right:.2f}')

# 
        # --- 수정: flag에 따라 부호 반전 ---
        if self.turn_flag == "LEFT":
            self.car_drive.axis0.controller.input_vel = -vel_right
            self.car_drive.axis1.controller.input_vel = vel_left
        
        elif self.turn_flag == "RIGHT":
            self.car_drive.axis0.controller.input_vel = vel_right
            self.car_drive.axis1.controller.input_vel = -vel_left

        else:  # flag 없음 → 기본값
            self.car_drive.axis0.controller.input_vel = -vel_right
            self.car_drive.axis1.controller.input_vel = vel_left

        self.get_logger().info(
            f'Cmd_vel -> Left: {vel_left:.2f}, Right: {vel_right:.2f}, Flag={self.turn_flag}'
        )




    #---- 오돔 발행 ---- !!
    def publish_odometry(self):

        cur0 = self.car_drive.axis0.encoder.count_in_cpr
        cur1 = self.car_drive.axis1.encoder.count_in_cpr

        # delta 계산 & wrap-around 보정 ---> Encoder 맞춰서 발행 잘되는지 확인.
        delta0 = cur0 - self.prev_count0
        delta1 = cur1 - self.prev_count1
        if delta0 > self.CPR / 2: delta0 -= self.CPR
        elif delta0 < -self.CPR / 2: delta0 += self.CPR
        if delta1 > self.CPR / 2: delta1 -= self.CPR
        elif delta1 < -self.CPR / 2: delta1 += self.CPR

        self.prev_count0 = cur0
        self.prev_count1 = cur1

        # 거리 계산
        dist_left = (delta1 / self.CPR) * 2 * math.pi * self.wheel_radius
        dist_right = (delta0 / self.CPR) * 2 * math.pi * self.wheel_radius
        d_center = (dist_right + dist_left) / 2.0
        d_theta = (dist_right - dist_left) / self.wheel_base

        # 위치 갱신
        self.theta += d_theta
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
        self.x += d_center * math.cos(self.theta)
        self.y += d_center * math.sin(self.theta)
        
        # TF 브로드캐스트: odom -> base_footprint 
        # ** 세팅 LIDAR : base_footprint, Depth : base_link)
        now = self.get_clock().now().to_msg()
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'
        
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        q = quaternion_from_euler(0, 0, self.theta)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.odom_broadcaster.sendTransform(t)

        # Odometry 
        odom_msg = Odometry()
        odom_msg.header.stamp = now
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_footprint'
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0 

        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]

        odom_msg.twist.twist.linear.x = d_center / 0.1
        odom_msg.twist.twist.angular.z = d_theta / 0.1

        self.odom_pub.publish(odom_msg)
        self.get_logger().info(f'Publishing odom: x={self.x:.2f}, y={self.y:.2f}, theta={math.degrees(self.theta):.1f}')


def main(args=None):
    rclpy.init(args=args)
    node = ODriveMobileBase()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
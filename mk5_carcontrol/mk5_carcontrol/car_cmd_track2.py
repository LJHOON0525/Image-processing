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

        # -------------------------------- ODrive SET --------------------------------
        self.car_drive = odrive.find_any(serial_number="3678387D3333")
        self.get_logger().info("ODrive connected!")
        self.calibration()



        # -------------------------------- ODrive Initializing => Ramped Velocity --------------------------------
        for axis in [self.car_drive.axis0, self.car_drive.axis1]:
            axis.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
            axis.controller.config.vel_ramp_rate = 5
            axis.controller.config.input_mode = INPUT_MODE_VEL_RAMP

        # --------------------------------ODRIVE Encoder ---> count_in_cpr  --------------------------------
        self.prev_count0 = self.car_drive.axis0.encoder.count_in_cpr
        self.prev_count1 = self.car_drive.axis1.encoder.count_in_cpr
        self.total_count0 = 0
        self.total_count1 = 0
        # -------------------------------- 차체 파라미터 SET !!-------------------------------- 
        self.robot_state = 0
        self.wheel_base = 0.465  # 단위 m
        self.CPR = 2000           
        self.wheel_radius = 0.06 

        # -------------------------------- 로봇 위치 Initializing -------------------------------- 
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        qos_profile = QoSProfile(depth=10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', qos_profile)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, qos_profile)


        self.car_cmd_pub = self.create_publisher(Float32MultiArray, 'car_cmd', qos_profile)
        #CMD 기반 속도 명령 저장 변수
        self.final_cmd_left = 0.0
        self.final_cmd_right = 0.0

        # 미션으로 부터 오는 제어 명령 처리
        self.final_control_sub = self.create_subscription(String, '/tracking', self.final_control, qos_profile)
        self.final_control_flag = None



        # -------------------------------- IMU Data SET -------------------------------- 
        #IMU Threshold
        self.normal_max = 107
        self.normal_min = 77

        self.right_weak_max = 124
        self.left_weak_max = 65


        #IMU - DIRECTION FLAG
        self.robot_roll = 0  #IMU Pitch 
        self.robot_pitch = 0 # IMU Pitch 



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
            self.final_cmd_right = -vel_right
            self.final_cmd_left = vel_left
            self.get_logger().info(
                f'Cmd_vel -> Left: {vel_left:.2f}, Right: {vel_right:.2f}, Flag={self.turn_flag}'
            )

        
        
        elif self.turn_flag == "RIGHT":
            self.final_cmd_right = vel_right
            self.final_cmd_left = -vel_left
            self.get_logger().info(
                f'Cmd_vel -> Left: {vel_left:.2f}, Right: {vel_right:.2f}, Flag={self.turn_flag}'
            )
            return self.final_cmd_right, self.final_cmd_left
        else:  # flag 없음 → 기본값
            self.final_cmd_right = -vel_right
            self.final_cmd_left = vel_left
            self.get_logger().info(
                f'Cmd_vel -> Left: {vel_left:.2f}, Right: {vel_right:.2f}, Flag={self.turn_flag}'
            )






#----------------------------- TRACK CONTROL : CMD + IMU +MISSIN_TRACK2-----------

    def final_control(self, msg: String):
        self.final_control_flag = msg.data

        if self.final_control_flag == "STOP": 

            self.car_drive.axis0.controller.input_vel = 0
            self.car_drive.axis1.controller.input_vel = 0
        elif self.final_control_flag == "FRONT": 

            self.car_drive.axis0.controller.input_vel = -5
            self.car_drive.axis1.controller.input_vel = 5

        elif self.final_control_flag == "CMD": 
            self.cmd_processing()


    def cmd_processing(self):        
            # -------- 직진 -------- 
            if self.final_cmd_left == self.final_cmd_right: 
                if self.robot_state == 0:
                    self.normal_tracking()
                    self.get_logger().info("NORMAL FRONT")

                # 직진 & 오른쪽으로 기운 상태
                elif self.robot_state == 1: #오른쪽으로 살짝 기운 상태
                    self.state1_front()
                elif self.robot_state == 2: #오른쪽으로 많이 기운 상태
                    self.state2_front()

                # 직진 & 왼쪽으로 기운 상태

                elif self.robot_state == -1: #왼쪽으로 살짝 기운 상태
                    self.state3_front()
                elif self.robot_state == -2: #왼쪽으로 많이 기운 상태
                    self.state4_front()

                else:
                    self.normal_tracking()
                    self.get_logger().info("NORMAL FRONT")



            # -------- 좌회전 -------- 
            elif self.final_cmd_left < self.final_cmd_right: 
                if self.robot_state == 0:
                    self.normal_tracking()
                    self.get_logger().info("NORMAL LEFT")

                # 좌회전 & 오른쪽으로 기운 상태
                elif self.robot_state == 1: #오른쪽으로 살짝 기운 상태
                    self.state1_left()
                elif self.robot_state == 2: #오른쪽으로 많이 기운 상태
                    self.state2_left()

                # 좌회전 & 왼쪽으로 기운 상태

                elif self.robot_state == -1: #왼쪽으로 살짝 기운 상태
                    self.state3_left()
                elif self.robot_state == -2: #왼쪽으로 많이 기운 상태
                    self.state4_left()

                else:
                    self.normal_tracking()
                    self.get_logger().info("NORMAL LEFT")



            # -------- 우회전 -------- 
            elif self.final_cmd_left > self.final_cmd_right: 
                if self.robot_state == 0:
                    self.normal_tracking()
                    self.get_logger().info("NORMAL RIGHT")
                 # 우회전 & 오른쪽으로 기운 상태
                elif self.robot_state == 1: #오른쪽으로 살짝 기운 상태
                    self.state1_right()
                elif self.robot_state == 2: #오른쪽으로 많이 기운 상태
                    self.state2_right()

                # 우회전 & 왼쪽으로 기운 상태
                elif self.robot_state == -1: #왼쪽으로 살짝 기운 상태
                    self.state3_right()
                elif self.robot_state == -2: #왼쪽으로 많이 기운 상태
                    self.state4_right()

                else:
                    self.normal_tracking()
                    self.get_logger().info("NORMAL RIGHT")
            

#####################--------- CAR DIRECTION ###########################

#----------------------- NORMAL STATE

    def normal_tracking(self) :
        self.car_drive.axis0.controller.input_vel = self.final_cmd_right
        self.car_drive.axis1.controller.input_vel = self.final_cmd_left
        #self.get_logger().info("NORMAL TRACKING")

#-----------------------STATE 1 : 오른쪽으로 살짝 기운 상태 => 오른쪽 바퀴 살짝 가중치 더 주기.

    def state1_front(self) :
        self.get_logger().info("STATE 1 - GO FRONT")

        self.car_drive.axis0.controller.input_vel = self.final_cmd_right*1.1
        self.car_drive.axis1.controller.input_vel = self.final_cmd_left

    def state1_left(self) :
        self.get_logger().info("STATE 1- GO LEFT")
        self.car_drive.axis0.controller.input_vel = self.final_cmd_right*1.2
        self.car_drive.axis1.controller.input_vel = self.final_cmd_left


    def state1_right(self) :
        self.get_logger().info("STATE 1- GO RIGHT")
        self.car_drive.axis0.controller.input_vel = self.final_cmd_right*0.9
        self.car_drive.axis1.controller.input_vel = self.final_cmd_left*1.2

#----------------------- STATE 2 : 오른쪽으로 많이 기운 상태 => 왼쪽 바퀴 살짝 가중치 더 주기.

    def state2_front(self) :

        self.get_logger().info("STATE 2 - GO FRONT")
        self.car_drive.axis0.controller.input_vel = self.final_cmd_right*1.15
        self.car_drive.axis1.controller.input_vel = self.final_cmd_left

    def state2_left(self) :
        self.get_logger().info("STATE 2- GO LEFT")
        self.car_drive.axis0.controller.input_vel = self.final_cmd_right*1.25
        self.car_drive.axis1.controller.input_vel = self.final_cmd_left

    def state2_right(self) :
        self.get_logger().info("STATE 2- GO RIGHT")
        self.car_drive.axis0.controller.input_vel = self.final_cmd_right*0.9
        self.car_drive.axis1.controller.input_vel = self.final_cmd_left*1.3
 


#----------------------- STATE 3 : 왼쪽으로 살짝 기운 상태 => 왼쪽 바퀴 가중치 살짝 주기

    def state3_front(self) :
        self.get_logger().info("STATE 3 - GO FRONT")
        self.car_drive.axis0.controller.input_vel = self.final_cmd_right
        self.car_drive.axis1.controller.input_vel = self.final_cmd_left*1.1


    def state3_left(self) :
        self.get_logger().info("STATE 3- GO LEFT")
        self.car_drive.axis0.controller.input_vel = self.final_cmd_right
        self.car_drive.axis1.controller.input_vel = self.final_cmd_left*1.2



    def state3_right(self) :
        self.get_logger().info("STATE 3- GO RIGHT")
        self.car_drive.axis0.controller.input_vel = self.final_cmd_right*1.2
        self.car_drive.axis1.controller.input_vel = self.final_cmd_left*0.9


    #STATE 4 : Robot => STRONG LEFT => GO RIGHT STRONG

    def state4_front(self) :

        self.get_logger().info("STATE 4 - GO FRONT")
        self.car_drive.axis0.controller.input_vel = self.final_cmd_right
        self.car_drive.axis1.controller.input_vel = self.final_cmd_left*1.15


    def state4_left(self) :
        self.get_logger().info("STATE 4- GO LEFT")
        self.car_drive.axis0.controller.input_vel = self.final_cmd_right
        self.car_drive.axis1.controller.input_vel = self.final_cmd_left*1.25

    def state4_right(self) :
        self.get_logger().info("STATE 4- GO RIGHT")
        self.car_drive.axis0.controller.input_vel = self.final_cmd_right*1.3
        self.car_drive.axis1.controller.input_vel = self.final_cmd_left*0.9
 




    #----------------------------- IMU MSG SAMPLING----------------------------
    def imu_msg_sampling(self, msg):
        robot_state = msg.data


        if robot_state == 0.0:
            self.get_logger().info("NORMAL")
            self.robot_state = 0


        if robot_state == -3.0:
            self.get_logger().info("뒤로 살짝 기운 상태")
            self.robot_state = -3

        elif robot_state == 3.0:
            self.get_logger().info("앞으로 살짝 기운 상태")
            self.robot_state = 3

        elif robot_state == 1.0:
            self.get_logger().info("오른쪽으로 살짝 기운 상태")
            self.robot_state = 1

        elif robot_state == 2.0:
            self.get_logger().info("오른쪽으로 많이 기운 상태")
            self.robot_state = 2

        elif robot_state == -1.0:
            self.get_logger().info("왼쪽으로 살짝 기운 상태")
            self.robot_state = -1

        elif robot_state == -2.0:
            self.get_logger().info("왼쪽으로 많이 기운 상태")
            self.robot_state = -2




    #----------------------------- Encoder 기반 Odom 발행-----------------------------
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
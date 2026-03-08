#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray, String, Bool
import math
import time

## TRACK 2 Algorithm Control

class TrackControl2(Node):

    def __init__(self):
        super().__init__('TrackControl2')

        # STATE 
        self.state1_done = False
        self.state2_done = False
        self.state3_done = False
        self.state_done = False 

        qos_profile = QoSProfile(depth=10)

        # 장애물 처리
        self.brick_sub = self.create_subscription(Bool, 'brick_detect', self.brick_callback, qos_profile)
        self.stick_sub = self.create_subscription(Bool, 'stick_detected', self.wood_callback, qos_profile)
        self.stair_flag = False
        self.brick_flag = False
        self.wood_flag = False

        # 계단 STATE 전이용 FLAG
        self.imu_sub = self.create_subscription(Float32MultiArray, 'imu_processing', self.imu_callback, qos_profile)

        # 계단 오르는 구간 flag
        self.robot_state = 0
        self.stair_step = 0

        # Flipper Data Publish
        self.flip_control_pub = self.create_publisher(String, 'Odrive_Flip_control', qos_profile)
        self.flipper_msg = String()
        self.on_stairs = False

        # 플리퍼 완료 플래그
        self.flipON_done_flag = False
        self.flipON_done_sub = self.create_subscription(Bool, 'flipON_done', self.flipON_done_callback, qos_profile)
        self.flipINIT_done_flag = False
        self.flipINIT_done_sub = self.create_subscription(Bool, 'flipINIT_done', self.flipINIT_done_callback, qos_profile)

        # 방향 설정
        self.direction_publisher = self.create_publisher(String, 'tracking', qos_profile)
        self.timer = self.create_timer(0.1, self.tracking_Function)

        self.direction_msg = String()

        # 🟢 STEP3 전이 체크용 타이머 & 기록 변수
        self.flip_init_time = None  # 🟢 플리퍼 펼친 시점 기록
        self.create_timer(0.1, self.check_flipon_timer)  # 🟢 STEP3 전이 체크

    # 플리퍼 동작 완료 플래그 함수
    def flipON_done_callback(self, msg: Bool):
        self.flipON_done_flag = msg.data

    def flipINIT_done_callback(self, msg: Bool):
        self.flipINIT_done_flag = msg.data

    # 상황 인식 플래그 함수
    def brick_callback(self, msg: Bool):
        self.brick_flag = msg.data

    def wood_callback(self, msg: Bool):
        self.wood_flag = msg.data

    def imu_callback(self, msg: Float32MultiArray):
        pass

    # TRACKING
    def tracking_Function(self):
        if not self.state1_done:
            direction = str(self.state1())
        else:
            direction = str(self.state2())
        
        self.direction_msg.data = direction
        self.direction_publisher.publish(self.direction_msg)

    def flipper_tracking(self):
        flipper = self.execute_stair_sequence()
        self.flipper_msg.data = flipper
        self.flip_control_pub.publish(self.flipper_msg)

    def state1(self):
        if self.stair_flag or True:
            result = self.execute_stair_sequence()
            if result == "DONE":
                self.state1_done = True
            return result
        else:
            self.get_logger().info("No stair mission, skipping to state2")
            self.state1_done = True
            return "SKIP"

    def state2(self):
        if self.state1_done:
            self.get_logger().info("STATE2 => MOVE BY CMD")
            self.direction_msg.data = "CMD"
            self.direction_publisher.publish(self.direction_msg)
            return "CMD"

    # ---------------- STATE 별 액션 함수 ----------------
    def execute_stair_sequence(self):
        self.get_logger().info("STATE1: EXECUTE STAIR SEQUENCE")
        
        if self.stair_step == 0:
            self.get_logger().info("STEP1. 차량 정지")
            self.direction_msg.data = "STOP"
            self.stair_step = 1
            return "STOP"

        elif self.stair_step == 1 and not self.on_stairs:
            self.get_logger().info("STEP2. FLIPON")
            self.flipper_msg.data = "FLIPON"
            self.flip_control_pub.publish(self.flipper_msg)

            if self.flipON_done_flag:
                self.flipON_done_flag = False
                self.stair_step = 2
                self.on_stairs = True
                # 🟢 STEP3 시작 시점 기록 !!!!
                self.flip_init_time = self.get_clock().now().nanoseconds * 1e-9
                self.get_logger().info(f"FLIPON 시점 기록: {self.flip_init_time}")


        elif self.stair_step == 2 and self.on_stairs:
            self.get_logger().info("STEP3. FRONT")
            self.direction_msg.data = "FRONT"
            return "FRONT"

        elif self.stair_step == 3 and self.on_stairs:
            self.get_logger().info("STEP4. CAR STOP & FLIPINIT")
            self.direction_msg.data = "STOP"
            self.direction_publisher.publish(self.direction_msg)

            self.flipper_msg.data = "FLIPINIT"
            self.flip_control_pub.publish(self.flipper_msg)

            if self.flipINIT_done_flag:
                self.flipINIT_done_flag = False
                self.stair_step = 4
            return "STOP" if not self.flipINIT_done_flag else "MOVING"

        elif self.stair_step == 4:
            self.get_logger().info("Stair sequence completed.")
            return "DONE"

    # STEP3 전이 조건 체크 타이머
    def check_flipon_timer(self):
        if self.flip_init_time is not None and self.stair_step == 2:
            elapsed = self.get_clock().now().nanoseconds * 1e-9 - self.flip_init_time
            imu_normal = (self.robot_state == 0)
            if elapsed >= 30.0 and imu_normal:
                self.get_logger().info("🟢 10초 경과 & IMU NORMAL -> STEP3로 전이")
                self.stair_step = 3
                self.flip_init_time = None  # 플리퍼 타이머 초기화
                
    # ---------------- IMU MSG SAMPLING ----------------
    def imu_msg_sampling(self, msg):
        self.robot_state = int(msg.data[0])
        # 기존 상태 체크 유지
        if self.robot_state == 0.0:
            self.get_logger().info("NORMAL")
            self.robot_state = 0
        elif self.robot_state == -3.0:
            self.get_logger().info("뒤로 살짝 기운 상태")
            self.robot_state = -3
        elif self.robot_state == 3.0:
            self.get_logger().info("앞으로 살짝 기운 상태")
            self.robot_state = 3
        elif self.robot_state == 1.0:
            self.get_logger().info("오른쪽으로 살짝 기운 상태")
            self.robot_state = 1
        elif self.robot_state == 2.0:
            self.get_logger().info("오른쪽으로 많이 기운 상태")
            self.robot_state = 2
        elif self.robot_state == -1.0:
            self.get_logger().info("왼쪽으로 살짝 기운 상태")
            self.robot_state = -1
        elif self.robot_state == -2.0:
            self.get_logger().info("왼쪽으로 많이 기운 상태")
            self.robot_state = -2

def main(args=None):
    rclpy.init(args=args)
    node = TrackControl2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

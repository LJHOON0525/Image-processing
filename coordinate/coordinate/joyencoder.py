import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import odrive
from odrive.enums import CONTROL_MODE_VELOCITY_CONTROL, INPUT_MODE_VEL_RAMP, AXIS_STATE_FULL_CALIBRATION_SEQUENCE, AXIS_STATE_IDLE, AXIS_STATE_CLOSED_LOOP_CONTROL
import time
from std_msgs.msg import Float32MultiArray

class Subscriber(Node):
    def __init__(self):
        super().__init__('joy')

        self.car_drive = odrive.find_any(serial_number="3678387D3333")
        # self.flip_drive = odrive.find_any(serial_number="336F37603433")
        self.calibration()

        qos_profile = QoSProfile(depth=10)
        self.motor_control_sub = self.create_subscription(
            Float32MultiArray,
            'joycmd',
            self.subscribe_joy_message,
            qos_profile)
        
        self.encoder_publisher = self.create_publisher(Float32MultiArray, '/encoder_data', qos_profile)
        
        self.create_timer(0.1, self.publish_encoder_data)  # 10Hz 퍼블리시

        # CPR 값 (엔코더 해상도)
        self.CountsPerRevolution = 2000  # 실제 값으로 수정하세요

        # 초기 엔코더 값 저장
        self.prev_count0 = self.car_drive.axis0.encoder.count_in_cpr
        self.prev_count1 = self.car_drive.axis1.encoder.count_in_cpr

        # 누적 펄스
        self.total_count0 = 0
        self.total_count1 = 0

    def calibration(self):
        self.get_logger().info('Calibration START')
        # CAR Axis 0 (Right)
        self.car_drive.axis0.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
        while self.car_drive.axis0.current_state != AXIS_STATE_IDLE:
            time.sleep(0.1)
        self.car_drive.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        self.get_logger().info('CAR RIGHT Calibration COMPLETE.')

        # CAR Axis 1 (Left)
        self.car_drive.axis1.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
        while self.car_drive.axis1.current_state != AXIS_STATE_IDLE:
            time.sleep(0.1)
        self.car_drive.axis1.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL       
        self.get_logger().info('CAR LEFT Calibration COMPLETE.')

    def publish_encoder_data(self):
        cur_count0 = self.car_drive.axis0.encoder.count_in_cpr
        cur_count1 = self.car_drive.axis1.encoder.count_in_cpr

        # --- delta 계산 ---
        delta0 = cur_count0 - self.prev_count0
        delta1 = cur_count1 - self.prev_count1

        # --- wrap-around 보정 ---
        if delta0 > self.CountsPerRevolution / 2:
            delta0 -= self.CountsPerRevolution
        elif delta0 < -self.CountsPerRevolution / 2:
            delta0 += self.CountsPerRevolution

        if delta1 > self.CountsPerRevolution / 2:
            delta1 -= self.CountsPerRevolution
        elif delta1 < -self.CountsPerRevolution / 2:
            delta1 += self.CountsPerRevolution

        # --- 누적 업데이트 ---
        self.total_count0 += delta0
        self.total_count1 += delta1

        # --- 이전 값 갱신 ---
        self.prev_count0 = cur_count0
        self.prev_count1 = cur_count1

        # --- 퍼블리시 ---
        encoder_data = Float32MultiArray()
        # 왼쪽 바퀴(self.total_count1), 오른쪽 바퀴(self.total_count0)
        # ⚠️ 오른쪽 바퀴는 부호 반전
        encoder_data.data = [float(self.total_count1), -float(self.total_count0)]

        self.encoder_publisher.publish(encoder_data)
        self.get_logger().info(f'Publishing encoder data: {encoder_data.data}')

    def subscribe_joy_message(self, msg): 
        joy_stick_data = msg.data
        self.get_logger().info(f'Received joy stick data: {joy_stick_data}')

        self.car_drive.axis0.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
        self.car_drive.axis1.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
        self.car_drive.axis0.controller.config.vel_ramp_rate = 5
        self.car_drive.axis1.controller.config.vel_ramp_rate = 5
        self.car_drive.axis0.controller.config.input_mode = INPUT_MODE_VEL_RAMP
        self.car_drive.axis1.controller.config.input_mode = INPUT_MODE_VEL_RAMP

        self.car_drive.axis0.controller.input_vel = 0
        self.car_drive.axis1.controller.input_vel = 0

        if joy_stick_data[0] == 0.0:
            self.car_drive.axis0.controller.input_vel = 0
            self.car_drive.axis1.controller.input_vel = 0
            self.get_logger().info('STOP')
        else:
            if joy_stick_data[0] == 1.0:
                self.get_logger().info('CAR mode')
                if joy_stick_data[1] > 0 and joy_stick_data[2] >= 0.8:
                    self.car_drive.axis0.controller.input_vel = -5
                    self.car_drive.axis1.controller.input_vel = 5
                    self.get_logger().info('FRONT')
                elif joy_stick_data[1] > 0.5 and joy_stick_data[2] < -0.5:
                    self.car_drive.axis0.controller.input_vel = 5
                    self.car_drive.axis1.controller.input_vel = 5
                    self.get_logger().info('RIGHT')
                elif joy_stick_data[1] < -0.5 and joy_stick_data[2] >= 0.5:
                    self.car_drive.axis0.controller.input_vel = -5
                    self.car_drive.axis1.controller.input_vel = -5
                    self.get_logger().info('LEFT')
                elif joy_stick_data[1] < -0.5 and joy_stick_data[2] < -0.5:
                    self.car_drive.axis0.controller.input_vel = 5
                    self.car_drive.axis1.controller.input_vel = -5
                    self.get_logger().info('BACK')
        time.sleep(0.1)

def main():
    rclpy.init()
    sub = Subscriber()
    rclpy.spin(sub)
    sub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
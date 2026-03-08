import rclpy                               # ROS 2 Python 클라이언트 라이브러리
from rclpy.node import Node                # ROS 2 노드 기본 클래스
from rclpy.qos import QoSProfile           # 퍼블리셔/서브스크립션의 QoS 설정용
import odrive                              # ODrive Python API
from odrive.enums import (                 # ODrive 제어/상태 관련 enum 상수
    CONTROL_MODE_VELOCITY_CONTROL,
    INPUT_MODE_VEL_RAMP,
    AXIS_STATE_FULL_CALIBRATION_SEQUENCE,
    AXIS_STATE_IDLE,
    AXIS_STATE_CLOSED_LOOP_CONTROL
)
import time                                
from std_msgs.msg import Float32MultiArray 

class Subscriber(Node):                  
    def __init__(self):
        super().__init__('joy')            

        self.car_drive = odrive.find_any(serial_number="3678387D3333")
        #self.flip_drive = odrive.find_any(serial_number="205C327D4D31")
        self.calibration()               

        qos_profile = QoSProfile(depth=10)
        self.motor_control_sub = self.create_subscription(
            Float32MultiArray,             
            'joycmd',                     
            self.subscribe_joy_message,   
            qos_profile)                   



    def calibration(self):
        self.get_logger().info('Calibration START')

#------------------------------ CAR CALIBRATION ------------------------
        self.car_drive.axis0.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE

        while self.car_drive.axis0.current_state != AXIS_STATE_IDLE:
            time.sleep(0.1)              
        self.car_drive.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
                                         
        self.get_logger().info('CAR RIGHT Calibration COMPLETE.')

        # --- Axis 1 (왼쪽)도 동일 절차 ---
        self.car_drive.axis1.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
        while self.car_drive.axis1.current_state != AXIS_STATE_IDLE:
            time.sleep(0.1)
        self.car_drive.axis1.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL       
        self.get_logger().info('CAR LEFT Calibration COMPLETE.')

#------------------------------ FLIPPER CALIBRATION ------------------------
        # self.flip_drive.axis0.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE

        # while self.flip_drive.axis0.current_state != AXIS_STATE_IDLE:
        #     time.sleep(0.1)              
        # self.flip_drive.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
                                         
        # self.get_logger().info('FLIPPER RIGHT Calibration COMPLETE.')

        # # --- Axis 1 (왼쪽)도 동일 절차 ---
        # self.flip_drive.axis1.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
        # while self.flip_drive.axis1.current_state != AXIS_STATE_IDLE:
        #     time.sleep(0.1)
        # self.flip_drive.axis1.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL       
        # self.get_logger().info('FLIPPER LEFT Calibration COMPLETE.')





    def subscribe_joy_message(self, msg):
        joy_stick_data = msg.data                    
        self.get_logger().info(f'Received joy stick data: {joy_stick_data}')

        # --- 제어모드/램프/입력모드 설정 (보통 초기 1회만 해도 됨) ---
        self.car_drive.axis0.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
        self.car_drive.axis1.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
        self.car_drive.axis0.controller.config.vel_ramp_rate = 5   # 속도 램프 기울기
        self.car_drive.axis1.controller.config.vel_ramp_rate = 5
        self.car_drive.axis0.controller.config.input_mode = INPUT_MODE_VEL_RAMP
        self.car_drive.axis1.controller.config.input_mode = INPUT_MODE_VEL_RAMP

        # self.flip_drive.axis0.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
        # self.flip_drive.axis1.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
        # self.flip_drive.axis0.controller.config.vel_ramp_rate = 5   # 속도 램프 기울기
        # self.flip_drive.axis1.controller.config.vel_ramp_rate = 5
        # self.flip_drive.axis0.controller.config.input_mode = INPUT_MODE_VEL_RAMP
        # self.flip_drive.axis1.controller.config.input_mode = INPUT_MODE_VEL_RAMP



        # 안전을 위해 일단 0으로 초기화
        self.car_drive.axis0.controller.input_vel = 0
        self.car_drive.axis1.controller.input_vel = 0
        # self.flip_drive.axis0.controller.input_vel = 0
        # self.flip_drive.axis1.controller.input_vel = 0


        if joy_stick_data[0] == 0.0:
            self.car_drive.axis0.controller.input_vel = 0
            self.car_drive.axis1.controller.input_vel = 0
            # self.flip_drive.axis0.controller.input_vel = 0
            # self.flip_drive.axis1.controller.input_vel = 0

            self.get_logger().info('STOP')


        else:
            if joy_stick_data[0] == 1.0:            # CAR 모드
                self.get_logger().info('CAR mode')
                # self.car_drive.axis0.controller.input_vel = joy_stick_data[2] * 6
                # self.car_drive.axis1.controller.input_vel = joy_stick_data[1] * 6
                # # #CAR FRONT
                if joy_stick_data[1] > 0 and joy_stick_data[2] >= 0.8:

                    self.car_drive.axis0.controller.input_vel = -8
                    self.car_drive.axis1.controller.input_vel = 8 
                    self.get_logger().info('FRONT')
                #CAR RIGHT    
                elif joy_stick_data[1] > 0.5 and joy_stick_data[2] < -0.5:
                    # RIGHT 회전: 두 바퀴 같은 부호로 지정 (여기선 둘 다 +)
                    self.car_drive.axis0.controller.input_vel = 8
                    self.car_drive.axis1.controller.input_vel = 8
                    self.get_logger().info('RIGHT')

                #CAR LEFT
                elif joy_stick_data[1] < -0.5 and joy_stick_data[2] >= 0.5:
                    # LEFT 회전: 두 바퀴 같은 부호로 지정 (여기선 둘 다 -)
                    self.car_drive.axis0.controller.input_vel = -8
                    self.car_drive.axis1.controller.input_vel = -8
                    self.get_logger().info('LEFT')

                #CAR BACK    
                elif joy_stick_data[1] < -0.5 and joy_stick_data[2] < -0.5:
                    # BACK: 직진 반대 (오른쪽 +, 왼쪽 -)
                    self.car_drive.axis0.controller.input_vel = 8
                    self.car_drive.axis1.controller.input_vel = -8
                    self.get_logger().info('BACK')


            
            elif joy_stick_data[0] == 2.0: #Flipper Mode
  
                self.get_logger().info('FLIPPER mode')
                # self.flip_drive.axis0.controller.input_vel = -joy_stick_data[2] * 6
                # self.flip_drive.axis1.controller.input_vel = joy_stick_data[1] * 6

        time.sleep(0.1)                              



def main():
    rclpy.init()                                     
    sub = Subscriber()                               
    rclpy.spin(sub)                                  
    sub.destroy_node()                               
    rclpy.shutdown()                                 

if __name__ == '__main__':
    main()                                          

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
import serial
import time
from .motor_utils import *

class NuriJoy(Node):
    def __init__(self):
        super().__init__('nuri_motor_sub')
        self.sub = self.create_subscription(Int32MultiArray, 'nuri_joy', self.cb, 10)
        self.ser = serial.Serial('/dev/ttyUSB0', baudrate=115200, timeout=0.1)


        #self.nuri_JOINT1 = 0x02  # ID 2만 사용
        self.nuri_JOINT2 = 0x02  # ID 2만 사용

        # JOINT2 Initializing
        send_control_on(self.ser, self.nuri_JOINT2)
        time.sleep(0.05)
        send_speed_ctrl_params(self.ser, self.nuri_JOINT2)
        time.sleep(0.05)
        send_position_ctrl_params(self.ser, self.nuri_JOINT2)
        time.sleep(0.05)
        set_position_mode(self.ser, self.nuri_JOINT2, 0x00)
        time.sleep(0.05)
        self.get_logger().info("JOINT2 INITIALIZED")
        self.get_logger().info("-------------------------------------------")


        # 현재 위치 읽기 (피드백으로 초기화 가능)
        self.current_pos = 0.0

        #Direction
        self.CW=0
        self.CCW=1

        #Velocity
        self.Nuri_Vel = 0.6

    def cb(self, msg: Int32MultiArray):
        try:

            # joint 2
            joint2_direction = msg.data[1]

            #조인트2 - 시계방향
            if joint2_direction == 1: 
                send_velocity_mode(self.ser, self.nuri_JOINT2, self.CW, self.Nuri_Vel)
            elif joint2_direction == -1:
                send_velocity_mode(self.ser, self.nuri_JOINT2, self.CCW, self.Nuri_Vel)
            else:
                send_velocity_mode(self.ser, self.nuri_JOINT2, self.CCW, 0.0)


            #ENCODER FEEDBACK
            request_encoder_feedback(self.ser, self.nuri_JOINT2)
            time.sleep(0.1)

            #응답 Read
            resp = read_response(self.ser)

            if resp:
                self.get_logger().info(f"[RAW RX] {resp['raw'].hex() if 'raw' in resp else resp}")
                if resp["mode"] == 0xD9:  # 엔코더 피드백
                    enc = parse_encoder(resp["data"])
                    if enc:
                        dir_str = "CCW" if enc["direction"] == 0 else "CW"
                        self.get_logger().info(f"[ENC] 방향: {dir_str}, 각도: {enc['angle']:.2f}°")
                        self.get_logger().info("-------------------------------------------")
            else:
                self.get_logger().warn("피드백 없음")

        except Exception as e:
            self.get_logger().error(f"에러: {e}")
            self.get_logger().error(f"에러: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = NuriJoy()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
import serial
import time
from .motor_utils import *
from .robotis_def import *
from .protocol2_packet_handler import * 
from .packet_handler import * 
from .port_handler import * 

# -------------------- DYNAMIXEL SETUP --------------------
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_OPERATING_MODE = 11

TORQUE_ENABLE = 1
DEVICENAME = '/dev/ttyUSB1'
BAUDRATE = 57600

DXL3_ID = 3
DXL4_ID = 4
DXL5_ID = 5
DXL6_ID = 6
DXL7_ID = 7  # 그리퍼

# -------------------- NURI MOTOR SETUP --------------------

NURI_SER = '/dev/ttyUSB2'
NURI_BAUD = 115200
NURI_IDS = [0x00, 0x01]  # ID0, ID1

class MotorController(Node):
    def __init__(self):
        super().__init__('motor_controller')

        self.dy3_init = 9000
        self.dy3_push = 4800

        # -------------------- SUBSCRIPTIONS --------------------
        # 버튼 시작 Data
        self.button_flag_sub = self.create_subscription(Bool, 'button_start_flag', self.track3_button_callback, 10)
        self.button_sub = self.create_subscription(Float32MultiArray, 'button_coordinate', self.button_callback, 10)
        
        # 문열기 시작 Data
        self.door_flag_sub = self.create_subscription(Bool, 'door_start_flag', self.track3_handle_callback, 10)
        self.door_sub = self.create_subscription(Float32MultiArray, 'handle_coordinate', self.handle_callback, 10)

        # -------------------- DYNAMIXEL INITIALIZE --------------------
        self.portHandler = PortHandler(DEVICENAME)
        self.packetHandler = PacketHandler(2.0)

        if not self.portHandler.openPort():
            self.get_logger().error("Failed to open Dynamixel port")
            exit(1)
        if not self.portHandler.setBaudRate(BAUDRATE):
            self.get_logger().error("Failed to set Dynamixel baudrate")
            exit(1)

        self.init_dynamixel_positions()

        # -------------------- NURI INITIALIZE --------------------
        self.ser = serial.Serial(NURI_SER, baudrate=NURI_BAUD, timeout=0.1)
        for mid in NURI_IDS:
            send_control_on(self.ser, mid)
            time.sleep(0.05)
            send_speed_ctrl_params(self.ser, mid)
            time.sleep(0.05)
            send_velocity_mode(self.ser, mid, 0, 0.0)  # Velocity Mode 초기화
            time.sleep(0.05)
            self.get_logger().info(f"[INIT] Nuri Motor ID={mid} Initialized in Velocity Mode")

        self.obj_x = None
        self.obj_y = None
        self.obj_z = None

    # -------------------- DYNAMIXEL INITIAL POSITION --------------------
    def init_dynamixel_positions(self):
        init_positions = {
            DXL3_ID: self.dy3_init,
            DXL4_ID: 3900,
            DXL5_ID: 3200,
            DXL6_ID: 4000
        }
        for dxl_id, pos in init_positions.items():
            # **!!! 토크 Enable 명령 추가 !!!**
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
            self.get_logger().info(f"[DXL{dxl_id}] Torque Enabled.")
            
            # 초기 위치 설정
            self.packetHandler.write4ByteTxRx(self.portHandler, dxl_id, ADDR_GOAL_POSITION, pos)
            self.get_logger().info(f"[DXL{dxl_id}] Initialized at {pos}")
    # -------------------- CALLBACKS --------------------

 
    def track3_button_callback(self, msg: Bool):
        if msg.data:
            self.get_logger().info("Button start flag received: True -> Waiting for coordinates")
            if self.obj_x is not None:
                self.execute_button_action()


    def button_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 3:
            #self.get_logger().warn("Invalid button_coordinate message")
            return
        self.obj_x, self.obj_y, self.obj_z = msg.data
        self.execute_button_action()
        self.get_logger().info(f"Button coordinates received: X={self.obj_x:.3f}, Y={self.obj_y:.3f}")


    # 문열기 동작
    def track3_handle_callback(self, msg: Bool):
        if msg.data:
            self.get_logger().info("Handle start flag received: True -> Execute handle_callback")
            self.execute_handle_action()
        else:
            self.get_logger().info("Handle start flag received: False -> No action")

    def handle_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 3:
            #self.get_logger().warn("Invalid Handle_coordinate message")
            return
        self.obj_x = msg.data[0]
        self.obj_y = msg.data[1]
        self.obj_z = msg.data[2]
        self.get_logger().info(f"Handle coordinates received: X={self.obj_x:.3f}, Y={self.obj_y:.3f}")



    # -------------------- BUTTON ACTION --------------------

    def execute_button_action(self):
        if self.obj_x is None:
            self.get_logger().warn("No button coordinates available")
            return

        # -------------------- CASE EXAMPLE --------------------
        if -0.17 <= self.obj_x <= -0.15:
            # ------------------------ 버튼 PUSH

            self.packetHandler.write4ByteTxRx(self.portHandler, DXL3_ID, ADDR_GOAL_POSITION, self.dy3_push) #초기
            self.get_logger().info("[DXL3] PUSH -> 4800")
            time.sleep(2.0)
            self.packetHandler.write4ByteTxRx(self.portHandler, DXL3_ID, ADDR_GOAL_POSITION, self.dy3_init)
            self.get_logger().info("[DXL3] RETURN -> 9000")

        elif self.obj_x > -0.15:
            # ------------------------ 누리 이동
            send_velocity_mode(self.ser, 0x00, 1, 0.5)
            self.get_logger().info("[NURI0] Velocity Run dir=1, vel=0.5")
            time.sleep(2.0)
            send_velocity_mode(self.ser, 0x00, 0, 0.0)
            self.get_logger().info("[NURI0] Stop")
            time.sleep(0.5)

            #------------------------ 다이나믹셀 PUSH
            self.packetHandler.write4ByteTxRx(self.portHandler, DXL3_ID, ADDR_GOAL_POSITION, self.dy3_push)
            self.get_logger().info("[DXL3] PUSH -> 1000")
            time.sleep(2.0)
            self.packetHandler.write4ByteTxRx(self.portHandler, DXL3_ID, ADDR_GOAL_POSITION, self.dy3_init)
            self.get_logger().info("[DXL3] RETURN ")

            # ------------------------  BASE 복귀 ------------------------
            send_velocity_mode(self.ser, 0x00, 1, 0.5)
            self.get_logger().info("[NURI0] Velocity Run dir=1, vel=0.5")
            time.sleep(2.0)
            send_velocity_mode(self.ser, 0x00, 0, 0.0)
            self.get_logger().info("[NURI0] Stop")





    # -------------------- Handle ACTION --------------------

    def execute_handle_action(self):
        if self.obj_x is None:
            self.get_logger().warn("No handle coordinates available")
            return

        # -------------------- CASE EXAMPLE --------------------
        if -0.17 <= self.obj_x <= -0.15:
            self.get_logger().warn("동작 추후 추가")


def main():
    rclpy.init()
    node = MotorController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

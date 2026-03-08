#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from .robotis_def import *
from .protocol2_packet_handler import * 
from .packet_handler import * 
from .port_handler import * 

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE  = 64
ADDR_GOAL_POSITION  = 116
ADDR_PRESENT_POSITION = 132

PROTOCOL_VERSION = 2.0
BAUDRATE = 57600

DXL3_ID = 3
DXL4_ID = 4
DXL5_ID = 5
DXL6_ID = 6
DXL7_ID = 7

DEVICENAME1 = '/dev/ttyUSB1'
TORQUE_ENABLE = 1
EXT_POSITION_CONTROL_MODE = 4

DXL_MOVING_STATUS_THRESHOLD = 20

portHandler = PortHandler(DEVICENAME1)
packetHandler = PacketHandler(PROTOCOL_VERSION)

class DynamixSub(Node):
    def __init__(self):
        super().__init__('dynamix_sub_delta')
        self.create_subscription(Int32MultiArray, 'dynamix_joy', self.callback, 10)

        # 초기 목표 위치 읽기 (조인트 4개만 사용)
        self.goal_positions = [0]*4

        # 포트 열기
        if not portHandler.openPort():
            raise Exception("Failed to open port")
        if not portHandler.setBaudRate(BAUDRATE):
            raise Exception("Failed to set baudrate")

        # Dynamixel 초기화
        self.init_dxl(DXL3_ID)
        self.init_dxl(DXL4_ID)
        self.init_dxl(DXL5_ID)
        self.init_dxl(DXL6_ID)
        self.init_dxl(DXL7_ID)

        # 현재 위치 읽어서 goal_positions 초기화
        self.goal_positions[0] = self.read_position(DXL3_ID)
        #self.goal_positions[1] = self.read_position(DXL4_ID)
        self.goal_positions[1] = self.read_position(DXL5_ID)
        self.goal_positions[2] = self.read_position(DXL6_ID)
        self.goal_positions[3] = self.read_position(DXL7_ID)

    def init_dxl(self, dxl_id):
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_OPERATING_MODE, EXT_POSITION_CONTROL_MODE)
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        self.get_logger().info(f"ID={dxl_id} initialized.")

    def read_position(self, dxl_id):
        pos, result, error = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRESENT_POSITION)
        if result != 0:
            self.get_logger().error(f"Failed to read position ID={dxl_id}")
            return 0
        return pos

    def callback(self, msg: Int32MultiArray):
        # delta 누적 (4개만)
        for i in range(4):
            self.goal_positions[i] += int(msg.data[i] / 0.088)

        # 목표 위치 쓰기
        dxl_ids = [DXL3_ID,DXL5_ID, DXL6_ID, DXL7_ID]
        for i, dxl_id in enumerate(dxl_ids):
            packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, self.goal_positions[i])
            self.get_logger().info(f"ID{dxl_id} goal: {self.goal_positions[i]}")

def main(args=None):
    rclpy.init(args=args)
    node = DynamixSub()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


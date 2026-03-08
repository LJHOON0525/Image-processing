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
NURI_IDS = [0x01, 0x02]  # ID0, ID1

class MotorController(Node):
    def __init__(self):
        super().__init__('motor_controller')
        self.base_id = 0x01


        # -------------------- SUBSCRIPTIONS --------------------
        self.button_flag_sub = self.create_subscription(Bool, 'button_start_flag', self.track3_button_callback, 10)
        self.button_sub = self.create_subscription(Float32MultiArray, 'button_coordinate', self.button_callback, 10)
        self.door_flag_sub = self.create_subscription(Bool, 'door_start_flag', self.track3_handle_callback, 10)
        self.door_sub = self.create_subscription(Float32MultiArray, 'handle_coordinate', self.handle_callback, 10)

        # -------------------- DYNAMIXEL INITIALIZE (NO POSE SET) --------------------
        self.portHandler = PortHandler(DEVICENAME)
        self.packetHandler = PacketHandler(2.0)

        if not self.portHandler.openPort():
            self.get_logger().error("Failed to open Dynamixel port")
            exit(1)
        if not self.portHandler.setBaudRate(BAUDRATE):
            self.get_logger().error("Failed to set Dynamixel baudrate")
            exit(1)

        self.init_dynamixel_positions()

        self.move_offset = 4200
        self.dy3_moved = False
        self.nuri_moved = False
        self.nuri_returned = False

        # -------------------- NURI INITIALIZE --------------------
        self.ser = serial.Serial(NURI_SER, baudrate=NURI_BAUD, timeout=0.1)
        for mid in NURI_IDS:
            send_control_on(self.ser, mid)
            time.sleep(0.05)
            send_speed_ctrl_params(self.ser, mid)
            time.sleep(0.05)
            set_position_mode(self.ser, mid, 0x00)
            time.sleep(0.05)
            self.get_logger().info(f"[INIT] Nuri Motor ID={mid} Initialized in Velocity Mode")



        self.obj_x = None
        self.obj_y = None
        self.obj_z = None

    def init_dynamixel_positions(self):
        for dxl_id in [DXL3_ID, DXL4_ID, DXL5_ID, DXL6_ID]:
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
            self.get_logger().info(f"[DXL{dxl_id}] Torque Enabled.")

    def get_present_position(self, dxl_id):
        pos, result, error = self.packetHandler.read4ByteTxRx(self.portHandler, dxl_id, ADDR_PRESENT_POSITION)
        if result != 0 or error != 0:
            self.get_logger().warn(f"[DXL{dxl_id}] Failed to get present position")
            return None
        return pos

    # -------------------- CALLBACKS --------------------
    def track3_button_callback(self, msg: Bool):
        if msg.data:
            self.get_logger().info("Button start flag received: True -> Waiting for coordinates")
            if self.obj_x is not None:
                self.execute_button_action()

    def button_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 3:
            return
        self.obj_x, self.obj_y, self.obj_z = msg.data
        self.execute_button_action()
        self.get_logger().info(f"Button coordinates received: X={self.obj_x:.3f}, Y={self.obj_y:.3f}")

    def track3_handle_callback(self, msg: Bool):
        if msg.data:
            self.get_logger().info("Handle start flag received: True -> Execute handle_callback")
            self.execute_handle_action()
        else:
            self.get_logger().info("Handle start flag received: False -> No action")

    def handle_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 3:
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
        
        # FRONT
        if -0.08 < self.obj_x < -0.03:
            self.get_logger().info("FRONT")
            if not self.dy3_moved:
                self.push()

        # ----------------------------- Object => LEFT -----------------------------

        # -- LEFT -> STATE 1
        elif -0.18 < self.obj_x < -0.08:
            self.get_logger().info("LEFT - STATE 1")
            # 동작 순서: 누리 전진 → 다이나믹셀 → 누리 복귀 (각 1회만) #1: LEFT, 2:RIGHT => Y:0.29
            if not self.nuri_moved:
                
                self.nuri_left()
                time.sleep(0.4)
                self.nuri_stop()
                time.sleep(0.5)
                self.nuri_moved = True

            if self.nuri_moved and not self.dy3_moved:
                self.push()

            if self.nuri_moved and self.dy3_moved and not self.nuri_returned:
                
                self.nuri_right()
                time.sleep(0.4)
                self.nuri_stop()
                self.nuri_returned = True


        # -- LEFT -> STATE 2

        elif -0.22  < self.obj_x < -0.18:
            self.get_logger().info("LEFT - STATE 2")
            # 동작 순서: 누리 전진 → 다이나믹셀 → 누리 복귀 (각 1회만) #1: LEFT, 2:RIGHT => Y:0.29
            if not self.nuri_moved:
                self.nuri_left()
                time.sleep(1.7)
                self.nuri_stop()
                time.sleep(0.5)
                self.nuri_moved = True

            if self.nuri_moved and not self.dy3_moved:
                self.push()

            if self.nuri_moved and self.dy3_moved and not self.nuri_returned:

                self.nuri_right()
                time.sleep(1.7)
                self.nuri_stop()
                self.nuri_returned = True

        # -- LEFT -> STATE 3


        elif -0.38  < self.obj_x <= -0.22:
            self.get_logger().info("LEFT - STATE 2")
            # 동작 순서: 누리 전진 → 다이나믹셀 → 누리 복귀 (각 1회만) #1: LEFT, 2:RIGHT => Y:0.29
            if not self.nuri_moved:
                self.nuri_left_fast()
                time.sleep(0.5)
                self.nuri_stop()
                time.sleep(0.5)
                self.nuri_moved = True

            if self.nuri_moved and not self.dy3_moved:
                self.push()

            if self.nuri_moved and self.dy3_moved and not self.nuri_returned:

                self.nuri_right_fast()
                time.sleep(0.5)
                self.nuri_stop()
                self.nuri_returned = True



        #----------------------------- Object => Right -----------------------------
        # -- RIGHT -> STATE 1  
        elif -0.03<self.obj_x < 0.02:
            self.get_logger().info("RIGHT-> STATE1")
            # 동작 순서: 누리 전진 → 다이나믹셀 → 누리 복귀 (각 1회만) #1: LEFT, 2:RIGHT => Y:0.29
            if not self.nuri_moved:
                
                
                self.nuri_right()
                time.sleep(0.4)

                self.nuri_stop()
                time.sleep(0.5)

                self.nuri_moved = True

            if self.nuri_moved and not self.dy3_moved:
                self.push()
                
            #NURI RETURN
            if self.nuri_moved and self.dy3_moved and not self.nuri_returned:

                self.nuri_left()
                time.sleep(0.4)

                self.nuri_stop()
 
                self.nuri_returned = True

        # -- RIGHT -> STATE 2        
        
        elif 0.02<self.obj_x < 0.07:
            self.get_logger().info("RIGHT-> STATE2")
            # 동작 순서: 누리 전진 → 다이나믹셀 → 누리 복귀 (각 1회만) #1: LEFT, 2:RIGHT => Y:0.29
            if not self.nuri_moved:
                
                
                self.nuri_right()
                time.sleep(1.3)

                self.nuri_stop()
                time.sleep(0.5)

                self.nuri_moved = True

            if self.nuri_moved and not self.dy3_moved:
                self.push()
                
            #NURI RETURN
            if self.nuri_moved and self.dy3_moved and not self.nuri_returned:

                self.nuri_left()
                time.sleep(1.3)

                self.nuri_stop()
 
                self.nuri_returned = True


        # -- RIGHT -> STATE 3  
        elif 0.07<self.obj_x < 0.17:
            self.get_logger().info("RIGHT-> STATE3")
            # 동작 순서: 누리 전진 → 다이나믹셀 → 누리 복귀 (각 1회만) #1: LEFT, 2:RIGHT => Y:0.29
            if not self.nuri_moved:
                
                
                self.nuri_right_fast()
                time.sleep(0.5)

                self.nuri_stop()
                time.sleep(0.5)

                self.nuri_moved = True

            if self.nuri_moved and not self.dy3_moved:
                self.push()
                
            #NURI RETURN
            if self.nuri_moved and self.dy3_moved and not self.nuri_returned:

                self.nuri_left_fast()
                time.sleep(0.5)

                self.nuri_stop()
 
                self.nuri_returned = True



        else:
            self.get_logger().info("TOO FAR")

    def push(self):
        dxl3_present = self.get_present_position(DXL3_ID)
        if dxl3_present is None:
            return
        self.packetHandler.write4ByteTxRx(self.portHandler, DXL3_ID, ADDR_GOAL_POSITION, dxl3_present - self.move_offset)
        self.get_logger().info(f"[DXL3] Move to {dxl3_present - self.move_offset}")
        time.sleep(2.0)
        dxl3_back_present = self.get_present_position(DXL3_ID)
        if dxl3_back_present is None:
            dxl3_back_present = dxl3_present - self.move_offset
        self.packetHandler.write4ByteTxRx(self.portHandler, DXL3_ID, ADDR_GOAL_POSITION, dxl3_back_present + self.move_offset)
        self.get_logger().info(f"[DXL3] Move to {dxl3_back_present + self.move_offset}")
        self.dy3_moved = True

    def nuri_right(self):
        send_velocity_mode(self.ser, self.base_id, 1, 0.8)
        self.get_logger().info("GO RIGHT")

    def nuri_left(self):
        send_velocity_mode(self.ser, self.base_id, 0, 0.8)
        self.get_logger().info("GO LEFT ")



    def nuri_right_fast(self):
        send_velocity_mode(self.ser, self.base_id, 1, 1.5)
        self.get_logger().info("GO RIGHT FAST")

    def nuri_left_fast(self):
        send_velocity_mode(self.ser, self.base_id, 0, 1.5)
        self.get_logger().info("GO LEFT FAST")


    def nuri_stop(self):
        send_velocity_mode(self.ser, self.base_id, 0, 0.0)
        self.get_logger().info("[NURI0] Stop")

    def execute_handle_action(self):
        if self.obj_x is None:
            self.get_logger().warn("No handle coordinates available")
            return
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

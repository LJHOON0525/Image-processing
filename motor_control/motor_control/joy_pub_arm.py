#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import Joy
from rclpy.qos import QoSProfile

class JoyPub(Node):
    def __init__(self):
        super().__init__('joy_pub')
        qos_profile = QoSProfile(depth=10)

        self.pub_dynamix = self.create_publisher(Int32MultiArray, 'dynamix_joy', qos_profile)
        self.pub_nuri = self.create_publisher(Int32MultiArray, 'nuri_joy', qos_profile)
        self.sub = self.create_subscription(Joy, '/joy', self.joy_callback, qos_profile)
        self.grip_close = 20
        self.grip_open = -10

        self.scale = 5  # axes[3] 값 스케일링 (속도/델타 조절용)

    def joy_callback(self, msg: Joy):
        
        #Dynamixel
        JOINT3_delta = 0
        JOINT4_delta = 0
        JOINT5_delta = 0
        JOINT6_delta = 0
        JOINT7_delta = 0

        #NURI
        JOINT1_delta = 0
        JOINT2_delta = 0


        # axes[5] 기준: 3,4 관절
       
        if len(msg.axes) > 4:
             #3joint
            if msg.axes[5] == 1:
                JOINT3_delta = int(-msg.axes[3] * self.scale)
                self.get_logger().info('3JOINT')
            #4joint
            elif msg.axes[4] == 1:
                JOINT5_delta = int(-msg.axes[3] * self.scale)
                self.get_logger().info('5JOINT')

            elif msg.axes[5] == -1:
                JOINT6_delta = int(-msg.axes[3] * self.scale)
                self.get_logger().info('6JOINT')

            elif msg.axes[4] == -1:
                JOINT7_delta = int(-msg.axes[3] * self.scale)
                self.get_logger().info('7JOINT')

        # buttons 기준: 5,6,7 관절
        if len(msg.buttons) > 0:

            #BASE
            if msg.buttons[0] == 1: #BASE CW
                if msg.axes[1] >0:
                    JOINT1_delta = 1
                    self.get_logger().info('BASE-CW')


                elif msg.axes[1] <0: #BASE CCW
                    JOINT1_delta = -1
                    self.get_logger().info('BASE-CCW')
                else:
                    JOINT1_delta = 0
                self.get_logger().info('BASE-STOP')

            #2JOINT
            elif msg.buttons[1] == 1:
                if msg.axes[1] >0: 
                    JOINT2_delta = -1
                    self.get_logger().info('JOINT2-CW')
                elif msg.axes[1] <0:
                    JOINT2_delta = 1
                    self.get_logger().info('JOINT2-CCW')
                else:
                    JOINT2_delta = 0
                    self.get_logger().info('JOINT2-STOP')
                self.get_logger().info('JOINT2')

            
            # elif msg.buttons[2] == 1:
            #     JOINT7_delta = int(self.grip_close)  # Close
            #     self.get_logger().info('CLOSE')
            # elif msg.buttons[3] == 1:
            #     JOINT7_delta = int(self.grip_open)   # Open
            #     self.get_logger().info('OPEN')
        #Dynamixel
        dynamix_out_msg = Int32MultiArray()
        dynamix_out_msg.data = [JOINT3_delta, JOINT5_delta, JOINT6_delta, JOINT7_delta]
        self.pub_dynamix.publish(dynamix_out_msg)
        #self.get_logger().info(f'Published deltas: {dynamix_out_msg.data}')

        #Nuri
        nuri_out_msg = Int32MultiArray()
        nuri_out_msg.data = [JOINT1_delta, JOINT2_delta]
        self.pub_nuri.publish(nuri_out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = JoyPub()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

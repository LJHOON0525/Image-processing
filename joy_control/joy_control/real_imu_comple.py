import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import pyrealsense2 as rs
import numpy as np
import math

class ComplementaryFilterNode(Node):
    def __init__(self):
        super().__init__('complementary_filter_node')

        # ROS2 publisher
        self.publisher_ = self.create_publisher(Float32, 'filtered_roll', 10)
        self.timer = self.create_timer(0.01, self.timer_callback)

        # RealSense 설정
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.accel)
        config.enable_stream(rs.stream.gyro)
        self.pipeline.start(config)

        # 필터 상수
        self.alpha = 0.98
        self.dt = 0.01  # 100Hz (10ms)
        self.roll = 0.0  # 초기 Roll

    def timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        accel = frames.first_or_default(rs.stream.accel).as_motion_frame().get_motion_data()
        gyro = frames.first_or_default(rs.stream.gyro).as_motion_frame().get_motion_data()

        # 가속도로부터 Roll 계산 (라디안 → 도)
        accel_angle = math.atan2(accel.y, accel.z) * 180 / math.pi

        # 자이로 데이터로 Roll 변화량 계산 (Roll 속도 → Roll 변화)
        gyro_rate = gyro.x * 180 / math.pi  # rad/s → deg/s
        self.roll = self.alpha * (self.roll + gyro_rate * self.dt) + (1 - self.alpha) * accel_angle

        # 메시지 퍼블리시
        msg = Float32()
        msg.data = self.roll
        self.publisher_.publish(msg)

        self.get_logger().info(f'Filtered Roll: {self.roll:.2f}°')

def main(args=None):
    rclpy.init(args=args)
    node = ComplementaryFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

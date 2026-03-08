import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import pyrealsense2 as rs
import numpy as np
import math

class ComplementaryFilterNode(Node):
    def __init__(self):
        super().__init__('complementary_filter_node')

        # ROS2 publisher
        self.publisher_ = self.create_publisher(Float32MultiArray, 'filtered_angles', 10)
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

        # 초기 Roll, Pitch, Yaw 값
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

    def timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        accel = frames.first_or_default(rs.stream.accel).as_motion_frame().get_motion_data()
        gyro = frames.first_or_default(rs.stream.gyro).as_motion_frame().get_motion_data()

        # 가속도로부터 Roll, Pitch 계산 (라디안 → 도)
        roll = math.atan2(accel.y, accel.z) * 180 / math.pi
        pitch = math.atan2(-accel.x, math.sqrt(accel.y**2 + accel.z**2)) * 180 / math.pi

        # 자이로 데이터로 Yaw 계산 (Yaw는 자이로 데이터로 계산)
        gyro_rate_yaw = gyro.z * 180 / math.pi  # Z축 자이로 데이터로 Yaw 계산 (rad/s → deg/s)
        self.yaw += gyro_rate_yaw * self.dt

        # 필터를 통한 Roll, Pitch 보정 (Complementary Filter 사용)
        self.roll = self.alpha * (self.roll + gyro.x * 180 / math.pi * self.dt) + (1 - self.alpha) * roll
        self.pitch = self.alpha * (self.pitch + gyro.y * 180 / math.pi * self.dt) + (1 - self.alpha) * pitch

        # 결과를 Float32MultiArray로 퍼블리시
        msg = Float32MultiArray()
        msg.data = [round(self.roll, 2), round(self.pitch, 2), round(self.yaw, 2)]
        self.publisher_.publish(msg)

        # 로그로 출력
        self.get_logger().info(f'Filtered Roll: {self.roll:.2f}°, Pitch: {self.pitch:.2f}°, Yaw: {self.yaw:.2f}°')

def main(args=None):
    rclpy.init(args=args)
    node = ComplementaryFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

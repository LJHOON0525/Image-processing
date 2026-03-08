import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np

class IMULowPassFilterNode(Node):
    def __init__(self):
        super().__init__('imu_low_pass_filter_node')

        # RealSense pipeline 시작
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.accel)
        config.enable_stream(rs.stream.gyro)
        self.pipeline.start(config)

        # 필터 상수 (0.0 ~ 1.0): 낮을수록 더 부드러움
        self.alpha = 0.2  

        # 이전값 저장
        self.prev_accel = np.zeros(3)
        self.prev_gyro = np.zeros(3)

        # 타이머: 10Hz 주기로 데이터 출력
        self.timer = self.create_timer(0.1, self.timer_callback)

    def low_pass_filter(self, current, previous):
        return self.alpha * current + (1 - self.alpha) * previous

    def timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        accel_frame = frames.first_or_default(rs.stream.accel)
        gyro_frame = frames.first_or_default(rs.stream.gyro)

        accel_data = np.array([
            accel_frame.as_motion_frame().get_motion_data().x,
            accel_frame.as_motion_frame().get_motion_data().y,
            accel_frame.as_motion_frame().get_motion_data().z
        ])

        gyro_data = np.array([
            gyro_frame.as_motion_frame().get_motion_data().x,
            gyro_frame.as_motion_frame().get_motion_data().y,
            gyro_frame.as_motion_frame().get_motion_data().z
        ])

        filtered_accel = self.low_pass_filter(accel_data, self.prev_accel)
        filtered_gyro = self.low_pass_filter(gyro_data, self.prev_gyro)

        self.prev_accel = filtered_accel
        self.prev_gyro = filtered_gyro

        self.get_logger().info(f'Filtered Accel: {filtered_accel}')
        self.get_logger().info(f'Filtered Gyro : {filtered_gyro}')


def main(args=None):
    rclpy.init(args=args)
    node = IMULowPassFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

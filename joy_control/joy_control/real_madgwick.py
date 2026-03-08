import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np
from ahrs.filters import Madgwick
from ahrs.common.orientation import q2euler  # 쿼터니언 -> 오일러 각

class IMUMadgwickNode(Node):
    def __init__(self):
        super().__init__('imu_madgwick_node')

        # RealSense pipeline 구성
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.accel)
        config.enable_stream(rs.stream.gyro)
        self.pipeline.start(config)

        # Madgwick 필터 초기화
        self.madgwick = Madgwick()
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # 초기 쿼터니언

        # 타이머 설정 (10Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        accel_frame = frames.first_or_default(rs.stream.accel)
        gyro_frame = frames.first_or_default(rs.stream.gyro)

        accel = np.array([
            accel_frame.as_motion_frame().get_motion_data().x,
            accel_frame.as_motion_frame().get_motion_data().y,
            accel_frame.as_motion_frame().get_motion_data().z
        ])

        gyro = np.radians(np.array([  # Madgwick은 rad/s 필요
            gyro_frame.as_motion_frame().get_motion_data().x,
            gyro_frame.as_motion_frame().get_motion_data().y,
            gyro_frame.as_motion_frame().get_motion_data().z
        ]))

        # 필터 업데이트
        self.q = self.madgwick.updateIMU(self.q, gyr=gyro, acc=accel)

        # 오일러 각 변환 (Roll, Pitch, Yaw)
        euler = np.degrees(q2euler(self.q))  # rad → deg
        self.get_logger().info(f'Orientation (RPY): {euler}')

def main(args=None):
    rclpy.init(args=args)
    node = IMUMadgwickNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

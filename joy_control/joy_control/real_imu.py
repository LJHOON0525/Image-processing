import rclpy
from rclpy.node import Node
import numpy as np
import pyrealsense2 as rs
from std_msgs.msg import Float32

class AccelPublisher(Node):

    def __init__(self):
        super().__init__('accel_publisher')
        self.accel_publisher_ = self.create_publisher(Float32, 'accel_values', 10)
        timer_period = 0.1  # 0.1초마다 데이터 퍼블리시
        self.timer = self.create_timer(timer_period, self.publish_accel_data)

        # RealSense 파이프라인 설정
        self.p = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.accel)  # 가속도계 스트림 활성화
        self.pipeline_profile = self.p.start(self.config)

    def publish_accel_data(self):
        # 프레임 받기
        frames = self.p.wait_for_frames()
        accel_frame = frames.first(rs.stream.accel)  # 가속도계 데이터

        if accel_frame:
            accel_data = accel_frame.as_motion_frame().get_motion_data()  # 가속도계 데이터 (x, y, z)
            accel_value = accel_data.x  # x축 가속도

            # Float32 메시지로 가속도값 퍼블리시
            msg_accel = Float32()
            msg_accel.data = accel_value
            self.accel_publisher_.publish(msg_accel)

            # 로그로 출력
            self.get_logger().info(f'Accel X: {accel_value}')

def main(args=None):
    rclpy.init(args=args)

    accel_publisher = AccelPublisher()

    rclpy.spin(accel_publisher)

    accel_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import pyrealsense2 as rs
from collections import deque
import numpy as np

class MovingAverageFilterNode(Node):
    def __init__(self):
        super().__init__('moving_average_filter_node')

        # ROS2 퍼블리셔
        self.publisher_ = self.create_publisher(Float32, 'filtered_accel_x', 10)
        self.timer = self.create_timer(0.01, self.timer_callback)  # 100Hz

        # RealSense 설정
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.accel)
        self.pipeline.start(config)

        # 이동 평균 필터 설정
        self.window_size = 10
        self.accel_x_window = deque(maxlen=self.window_size)

    def timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        accel_frame = frames.first_or_default(rs.stream.accel).as_motion_frame()
        accel_data = accel_frame.get_motion_data()

        # 새 값 추가
        self.accel_x_window.append(accel_data.x)

        # 이동 평균 계산
        filtered_value = float(np.mean(self.accel_x_window))

        # 퍼블리시
        msg = Float32()
        msg.data = filtered_value
        self.publisher_.publish(msg)

        self.get_logger().info(f'Moving Average Filtered Accel X: {filtered_value:.4f}')

def main(args=None):
    rclpy.init(args=args)
    node = MovingAverageFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

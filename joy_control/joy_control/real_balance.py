import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
from std_msgs.msg import Float32

class IMUDualPublisher(Node):

    def __init__(self):
        super().__init__('imu_dual_publisher')

        # 퍼블리셔 생성
        self.accel_publisher_ = self.create_publisher(Float32, 'accel_values', 10)
        self.gyro_publisher_ = self.create_publisher(Float32, 'gyro_values', 10)

        # 타이머 생성 (0.1초 간격)
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.publish_imu_data)

        # RealSense 파이프라인 설정
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.accel)
        self.config.enable_stream(rs.stream.gyro)
        self.pipeline_profile = self.pipeline.start(self.config)

    def publish_imu_data(self):
        frames = self.pipeline.wait_for_frames()

        accel_frame = frames.first(rs.stream.accel)
        gyro_frame = frames.first(rs.stream.gyro)

        if accel_frame:
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            accel_x = accel_data.x
            msg_accel = Float32()
            msg_accel.data = accel_x
            self.accel_publisher_.publish(msg_accel)
            self.get_logger().info(f'Accel X: {accel_x}')

        if gyro_frame:
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            gyro_x = gyro_data.x
            msg_gyro = Float32()
            msg_gyro.data = gyro_x
            self.gyro_publisher_.publish(msg_gyro)
            self.get_logger().info(f'Gyro X: {gyro_x}')

def main(args=None):
    rclpy.init(args=args)
    imu_publisher = IMUDualPublisher()
    rclpy.spin(imu_publisher)
    imu_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

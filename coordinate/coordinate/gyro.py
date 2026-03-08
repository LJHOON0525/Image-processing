import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
from std_msgs.msg import Float32MultiArray

class GyroPublisher(Node):

    def __init__(self):
        super().__init__('gyro_publisher')
        self.gyro_publisher_ = self.create_publisher(Float32MultiArray, 'gyro_values', 10)
        timer_period = 0.1  # 0.1초마다 데이터 퍼블리시
        self.timer = self.create_timer(timer_period, self.publish_gyro_data)

        # RealSense 파이프라인 설정
        self.p = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.gyro)  # 자이로 스트림 활성화
        self.pipeline_profile = self.p.start(self.config)

    def publish_gyro_data(self):
        # 프레임 받기
        frames = self.p.wait_for_frames()
        gyro_frame = frames.first(rs.stream.gyro)  # 자이로 데이터

        if gyro_frame:
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()  # 자이로 데이터 (x, y, z)
            gyro_values = [gyro_data.x, gyro_data.y, gyro_data.z]  # Roll, Pitch, Yaw 값 (각속도)

            # 소수점 2자리로 반올림
            gyro_values_rounded = [round(value, 2) for value in gyro_values]

            # Float32MultiArray 메시지로 자이로 값 퍼블리시
            msg_gyro = Float32MultiArray()
            msg_gyro.data = gyro_values_rounded  
            self.gyro_publisher_.publish(msg_gyro)

            self.get_logger().info(f'Gyro Roll: {gyro_values_rounded[0]}, Pitch: {gyro_values_rounded[1]}, Yaw: {gyro_values_rounded[2]}')

def main(args=None):
    rclpy.init(args=args)

    gyro_publisher = GyroPublisher()

    rclpy.spin(gyro_publisher)

    gyro_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

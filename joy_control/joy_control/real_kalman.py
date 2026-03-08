import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np

class KalmanFilter:
    def __init__(self, process_noise_cov, measurement_noise_cov, estimate_error_cov):
        self.process_noise_cov = process_noise_cov
        self.measurement_noise_cov = measurement_noise_cov
        self.estimate_error_cov = estimate_error_cov
        self.x = 0  # 초기 추정값
        self.P = 1  # 초기 추정 오차

    def predict(self):
        # 예측 단계 (단순한 경우 x_k = x_(k-1))
        self.x = self.x
        self.P = self.P + self.process_noise_cov

    def update(self, z):
        # 갱신 단계 (센서 측정값 z)
        K = self.P / (self.P + self.measurement_noise_cov)  # 칼만 이득
        self.x = self.x + K * (z - self.x)  # 예측값과 실제 측정값 차이를 보정
        self.P = (1 - K) * self.P  # 오차 공분산 갱신
        return self.x

class IMUKalmanFilterNode(Node):
    def __init__(self):
        super().__init__('imu_kalman_filter_node')

        # RealSense pipeline 시작
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.accel)
        config.enable_stream(rs.stream.gyro)
        self.pipeline.start(config)

        # 칼만 필터 인스턴스 생성 (공정 잡음, 측정 잡음, 추정 오차 공분산)
        self.kf_accel = KalmanFilter(0.1, 0.5, 1.0)  # 가속도계 필터
        self.kf_gyro = KalmanFilter(0.1, 0.5, 1.0)   # 자이로스코프 필터

        # 타이머: 10Hz 주기로 데이터 출력
        self.timer = self.create_timer(0.1, self.timer_callback)

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

        # 칼만 필터로 가속도계와 자이로스코프 값 필터링
        filtered_accel = np.array([
            self.kf_accel.update(accel_data[0]),
            self.kf_accel.update(accel_data[1]),
            self.kf_accel.update(accel_data[2])
        ])

        filtered_gyro = np.array([
            self.kf_gyro.update(gyro_data[0]),
            self.kf_gyro.update(gyro_data[1]),
            self.kf_gyro.update(gyro_data[2])
        ])

        self.get_logger().info(f'Filtered Accelerometer: {filtered_accel}')
        self.get_logger().info(f'Filtered Gyroscope: {filtered_gyro}')


def main(args=None):
    rclpy.init(args=args)
    node = IMUKalmanFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

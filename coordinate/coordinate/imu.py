import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32MultiArray
import pyrealsense2 as rs
import numpy as np
import math
import cv2
import time

class CameraPoseCirculate(Node):
    def __init__(self):
        super().__init__('imu_node')
        qos_profile = QoSProfile(depth=2)
        self.camera_pose = self.create_publisher(
            Float32MultiArray,
            'imu_data',
            qos_profile)

        self.sample_freq = 1 / 60.0  # 60Hz 샘플링

        # RealSense 파이프라인 설정
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.accel)
        self.config.enable_stream(rs.stream.gyro)
        self.pipeline.start(self.config)

        # 보정용 변수
        self.last_time = time.time()
        self.roll = 0.0
        self.pitch = 0.0
        self.alpha = 0.98  # complementary filter 계수 #가장 무난

        self.timer = self.create_timer(self.sample_freq, self.circulate_pose)

    def circulate_pose(self):
        msg = Float32MultiArray()

        frames = self.pipeline.wait_for_frames()
        accel_frame = frames.first_or_default(rs.stream.accel, rs.format.motion_xyz32f)
        gyro_frame = frames.first_or_default(rs.stream.gyro, rs.format.motion_xyz32f)

        if not accel_frame or not gyro_frame:
            return

        accel = accel_frame.as_motion_frame().get_motion_data()
        gyro = gyro_frame.as_motion_frame().get_motion_data()

        dt = time.time() - self.last_time
        self.last_time = time.time()

        accel_roll = math.atan2(accel.y, accel.z) * 180.0 / math.pi
        accel_pitch = math.atan2(-accel.x, math.sqrt(accel.y**2 + accel.z**2)) * 180.0 / math.pi

        # 자이로 각속도 (deg/s)
        gyro_roll_rate = gyro.x * 180.0 / math.pi
        gyro_pitch_rate = gyro.y * 180.0 / math.pi

        # 보정된 roll, pitch (complementary filter)
        self.roll = self.alpha * (self.roll + gyro_roll_rate * dt) + (1 - self.alpha) * accel_roll
        self.pitch = self.alpha * (self.pitch + gyro_pitch_rate * dt) + (1 - self.alpha) * accel_pitch

        msg.data = [self.roll, self.pitch]
        self.get_logger().info(f'Roll: {self.roll:.2f}  Pitch: {self.pitch:.2f}')
        self.camera_pose.publish(msg)

        # HUD 표시
        self.display_hud(self.roll, self.pitch)

    def display_hud(self, roll, pitch):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        center = (320, 240)

        cv2.putText(image, f"Pitch: {abs(roll):.2f}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, f"roll: {abs(pitch):.2f}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 간단한 시각화
        cv2.line(image, center, (center[0] + int(pitch * 2), center[1]), (0, 255, 0), 3)
        cv2.line(image, center, (center[0], center[1] - int(roll * 2)), (0, 0, 255), 3)
        cv2.drawMarker(image, center, (255, 255, 255), cv2.MARKER_CROSS, 10, 2)

        cv2.imshow("IMU HUD", image)
        cv2.waitKey(1)

    def destroy_node(self):
        super().destroy_node()
        self.pipeline.stop()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPoseCirculate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

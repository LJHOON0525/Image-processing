#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32MultiArray

import cv2
import pyrealsense2 as rs
import numpy as np
import math


class CameraPoseCirculate(Node):
    def __init__(self):
        super().__init__('imu_node')
        qos_profile = QoSProfile(depth=2)

        # 퍼블리셔
        self.camera_pose = self.create_publisher(
            Float32MultiArray,
            'imu_data',
            qos_profile
        )

        # RealSense 설정
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.accel)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

        # 주기 설정 (약 15Hz)
        self.sample_freq = 1 / 15
        self.timer = self.create_timer(self.sample_freq, self.circulate_pose)

        # HUD 경고 임계값
        self.ROLL_THRESHOLD = 30.0
        self.PITCH_THRESHOLD = 30.0

    def circulate_pose(self):
        msg = Float32MultiArray()

        frames = self.pipeline.wait_for_frames()
        accel_frame = frames.first_or_default(rs.stream.accel)
        color_frame = frames.get_color_frame()

        if accel_frame and color_frame:
            accel_datas = accel_frame.as_motion_frame().get_motion_data()
            accel_data = [accel_datas.x, accel_datas.y, accel_datas.z]

            # Roll, Pitch 계산 (Yaw는 없음)
            norm = math.sqrt(accel_data[0]**2 + accel_data[1]**2 + accel_data[2]**2)
            if norm == 0:
                return

            roll = math.atan2(accel_data[1], accel_data[2]) * 180.0 / math.pi
            pitch = math.atan2(-accel_data[0], math.sqrt(accel_data[1]**2 + accel_data[2]**2)) * 180.0 / math.pi
            yaw = 0.0  # Yaw는 가속도만으로 불가능

            msg.data = [roll, pitch, yaw]
            self.camera_pose.publish(msg)

            self.get_logger().info(f"Roll: {roll:.2f}, Pitch: {pitch:.2f}, Yaw: {yaw:.2f}")

            # HUD 표시
            color_image = np.asanyarray(color_frame.get_data())
            center_distance = 0.3  # 거리 측정 센서가 없으므로 임의값
            self.display_hud(color_image, roll, pitch, yaw, center_distance)

    def display_hud(self, image, roll, pitch, yaw, center_distance):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        line_length = 100

        pitch_offset = int(pitch * 2)
        roll_angle = -roll

        hud_layer = np.zeros_like(image)

        # 수평선 표시 (Pitch)
        cv2.line(hud_layer,
                 (center[0] - line_length, center[1] + pitch_offset),
                 (center[0] + line_length, center[1] + pitch_offset),
                 (0, 255, 255), 2)

        # 롤에 따라 회전
        M = cv2.getRotationMatrix2D(center, roll_angle, 1.0)
        hud_layer = cv2.warpAffine(hud_layer, M, (w, h))

        # 중심 십자가
        cv2.drawMarker(hud_layer, center, (0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)

        # 텍스트 표시
        cv2.putText(hud_layer, f"Roll : {roll:+06.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(hud_layer, f"Pitch: {pitch:+06.2f}", (240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(hud_layer, f"Yaw  : {yaw:+06.2f}", (440, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(hud_layer, f"Center Distance: {center_distance:.2f} m", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # 최종 HUD 이미지 합성 및 표시
        hud_image = cv2.add(image, hud_layer)

        #cv2.imshow("original", image.copy())
        cv2.imshow("HUD only", hud_layer)
        cv2.imshow("D435i HUD", hud_image)
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

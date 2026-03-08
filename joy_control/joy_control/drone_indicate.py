#!/usr/bin/env python3
from rclpy.qos import QoSProfile, ReliabilityPolicy
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from px4_msgs.msg import VehicleAttitude
import pyrealsense2 as rs
from ultralytics import YOLO
import time
import signal
import sys

class PX4HUDBirdNode(Node):
    def __init__(self):
        super().__init__('px4_hud_bird_node')

        # RealSense 초기화
        try:
            self.pipeline = rs.pipeline()
            self.align = rs.align(rs.stream.color)
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
            self.pipeline.start(config)
        except Exception as e:
            self.get_logger().error(f"[RealSense 시작 실패] {e}")
            sys.exit(1)

        # PX4 자세 데이터 구독
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.subscription = self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.attitude_callback,
            qos_profile
        )

        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.roll = self.pitch = self.yaw = 0.0

        self.timer = self.create_timer(0.03, self.timer_callback)

        self.ROLL_THRESHOLD = 15.0
        self.PITCH_THRESHOLD = 10.0
        self.last_time = time.time()
        self.frame_count = 0

        # ---------------- Bird Avoidance ----------------
        self.model = YOLO("yolov8n.pt")  # Bird 클래스 포함
        self.distance_threshold = 2.0  # m 단위
        self.avoid_active = False
        self.avoid_counter = 0
        self.avoid_duration = 30  # 프레임 단위
        self.avoid_dir = "RIGHT"

    def attitude_callback(self, msg):
        self.q = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]])
        r = R.from_quat([self.q[1], self.q[2], self.q[3], self.q[0]])
        self.roll, self.pitch, self.yaw = r.as_euler('xyz', degrees=True)

    def process_bird_detection(self, color_image, depth_image):
        results = self.model.predict(color_image, conf=0.4, max_det=3, classes=[14])
        if len(results[0].boxes):
            box = results[0].boxes.xywh[0].cpu().numpy().astype(int)
            x_center, y_center = int(box[0]), int(box[1])
            distance = depth_image[y_center, x_center]*0.001  # mm -> m
            if distance < self.distance_threshold:
                self.avoid_active = True
                self.avoid_counter = 0
                self.avoid_dir = "RIGHT" if x_center < depth_image.shape[1]//2 else "LEFT"
                self.get_logger().warn(f"조류감지- 우회 경로 생성, 거리: {distance:.2f} m")
                return True
        return False

    def timer_callback(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                self.get_logger().warn("[RealSense] 유효한 프레임을 수신하지 못했습니다.")
                return

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            h, w = color_image.shape[:2]
            center_distance = depth_frame.get_distance(w // 2, h // 2)

            # Bird Detection
            bird_warning = self.process_bird_detection(color_image, depth_image)

            # 회피 상태 처리
            if self.avoid_active:
                self.avoid_counter += 1
                if self.avoid_counter > self.avoid_duration:
                    self.avoid_active = False

            self.display_hud(color_image, self.roll, self.pitch, self.yaw, center_distance, bird_warning)

            # FPS 계산
            self.frame_count += 1
            elapsed = time.time() - self.last_time
            if elapsed >= 1.0:
                fps = self.frame_count / elapsed
                self.get_logger().info(f"[FPS] {fps:.2f}")
                self.frame_count = 0
                self.last_time = time.time()

        except Exception as e:
            self.get_logger().warn(f"[HUD 오류] {e}")

    def display_hud(self, image, roll, pitch, yaw, object_distance, bird_warning):
        h, w = image.shape[:2]
        hud_layer = np.zeros_like(image)

        # 기존 HUD 표시
        pitch_offset = int(pitch * 2)
        center = (w // 2, h // 2)
        line_length = 100
        cv2.line(hud_layer,
                 (center[0] - line_length, center[1] + pitch_offset),
                 (center[0] + line_length, center[1] + pitch_offset),
                 (0, 255, 255), 2)
        M = cv2.getRotationMatrix2D(center, -roll, 1.0)
        hud_layer = cv2.warpAffine(hud_layer, M, (w, h))
        cv2.drawMarker(hud_layer, center, (0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
        cv2.putText(hud_layer, f"Roll : {roll:+06.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(hud_layer, f"Pitch: {pitch:+06.2f}", (240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(hud_layer, f"Yaw  : {yaw:+06.2f}", (440, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(hud_layer, f"Object Distance: {object_distance:.2f} m", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # Bird Warning
        if bird_warning:
            cv2.putText(hud_layer, "WARNING: BIRD AVOID!", (w // 2 - 100, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        hud_image = cv2.add(image, hud_layer)
        cv2.imshow("PX4 HUD with Bird Avoid", hud_image)
        cv2.waitKey(5)

# 안전 종료
def signal_handler(sig, frame):
    print("\n[종료] Ctrl+C 신호 수신. 종료합니다.")
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main(args=None):
    rclpy.init(args=args)
    node = PX4HUDBirdNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()

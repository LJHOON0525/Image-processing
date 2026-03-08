from rclpy.qos import QoSProfile, ReliabilityPolicy
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from px4_msgs.msg import VehicleAttitude
import pyrealsense2 as rs
import time
import signal
import sys

class PX4HUDNode(Node):
    def __init__(self):
        super().__init__('px4_hud_node')

        # RealSense 초기화 (IMU 미사용)
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

        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # 초기 quaternion (w, x, y, z)
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.timer = self.create_timer(0.03, self.timer_callback)

        self.ROLL_THRESHOLD = 15.0
        self.PITCH_THRESHOLD = 10.0
        self.last_time = time.time()
        self.frame_count = 0

    def attitude_callback(self, msg):
        self.q = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]])
        r = R.from_quat([self.q[1], self.q[2], self.q[3], self.q[0]])  # (x, y, z, w)
        self.roll, self.pitch, self.yaw = r.as_euler('xyz', degrees=True)

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
            h, w = color_image.shape[:2]
            center_distance = depth_frame.get_distance(w // 2, h // 2)

            self.display_hud(color_image, self.roll, self.pitch, self.yaw, center_distance)

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

    def draw_yaw_side_scale(self, image, yaw):
        h, w = image.shape[:2]
        center_y = h // 2
        for offset in range(-30, 31, 10):
            display_yaw = abs(int(round(yaw / 10.0) * 10 + offset))
            y_pos = center_y - offset * 5
            if 0 < y_pos < h:
                text = f"{display_yaw:02d}"
                cv2.putText(image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.line(image, (60, y_pos), (70, y_pos), (255, 255, 0), 1)
                cv2.putText(image, text, (w - 60, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.line(image, (w - 80, y_pos), (w - 90, y_pos), (255, 255, 0), 1)

    def display_hud(self, image, roll, pitch, yaw, object_distance):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        line_length = 100

        pitch_offset = int(pitch * 2)
        roll_angle = -roll
        hud_layer = np.zeros_like(image)

        # 수평선 + 회전
        cv2.line(hud_layer,
                 (center[0] - line_length, center[1] + pitch_offset),
                 (center[0] + line_length, center[1] + pitch_offset),
                 (0, 255, 255), 2)

        M = cv2.getRotationMatrix2D(center, roll_angle, 1.0)
        hud_layer = cv2.warpAffine(hud_layer, M, (w, h))

        cv2.drawMarker(hud_layer, center, (0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)

        # 텍스트 출력
        cv2.putText(hud_layer, f"Roll : {roll:+06.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(hud_layer, f"Pitch: {pitch:+06.2f}", (240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(hud_layer, f"Yaw  : {yaw:+06.2f}", (440, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(hud_layer, f"Object Distance: {object_distance:.2f} m", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        if object_distance < 0.5 and object_distance > 0:
            cv2.putText(hud_layer, "WARNING: Too Close!", (w // 2 - 100, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        #if abs(roll) > self.ROLL_THRESHOLD or abs(pitch) > self.PITCH_THRESHOLD:
        #    cv2.putText(hud_layer, "WARNING:  Limit Twist!", (w // 2 - 200, h // 2 - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        #기울어짐은 미터단위 환산 후 다시 사용

        dy_pixel = pitch_offset
        pixel_to_meter = 0.0025
        dy_m = dy_pixel * pixel_to_meter

        cv2.putText(hud_layer,f"Offset from Center: {dy_m:+.2f} m",(300, 60),cv2.FONT_HERSHEY_SIMPLEX,0.6, (255, 255, 0), 2)

        self.draw_yaw_side_scale(hud_layer, yaw)
        hud_image = cv2.add(image, hud_layer)

        #cv2.imshow("original", image.copy())
        cv2.imshow("HUD only", hud_layer)
        cv2.imshow("PX4 HUD", hud_image)
        cv2.waitKey(5)

# 안전한 종료 처리
def signal_handler(sig, frame):
    print("\n[종료] Ctrl+C 신호 수신. 종료합니다.")
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main(args=None):
    rclpy.init(args=args)
    node = PX4HUDNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

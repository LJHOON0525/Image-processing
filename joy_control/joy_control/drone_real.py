import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np
import cv2
from ahrs.filters import Mahony
from scipy.spatial.transform import Rotation as R
import time

class D435iHUDNode(Node):
    def __init__(self):
        super().__init__('d435i_hud_node')

        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        config = rs.config()
        config.enable_stream(rs.stream.accel)
        config.enable_stream(rs.stream.gyro)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        self.mahony = Mahony(sample_period=0.01)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

        self.timer = self.create_timer(0.01, self.timer_callback)

        self.ROLL_THRESHOLD = 15.0
        self.PITCH_THRESHOLD = 10.0
        self.CENTER_THRESHOLD = 1.0  # 중앙 십자가 허용 오차
        self.center_hold_time = 5  

        self.last_time = time.time()
        self.frame_count = 0

        self.last_correction_time = time.time()
        self.center_start_time = None  

    def timer_callback(self):
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            accel = frames.first_or_default(rs.stream.accel)
            gyro = frames.first_or_default(rs.stream.gyro)

            if not color_frame or not depth_frame or not accel or not gyro:
                self.get_logger().warn("필요한 프레임이 없습니다.")
                return

            color_image = np.asanyarray(color_frame.get_data())
            accel_data = np.array([
                accel.as_motion_frame().get_motion_data().x,
                accel.as_motion_frame().get_motion_data().y,
                accel.as_motion_frame().get_motion_data().z
            ])
            gyro_data = np.radians(np.array([
                gyro.as_motion_frame().get_motion_data().x,
                gyro.as_motion_frame().get_motion_data().y,
                gyro.as_motion_frame().get_motion_data().z
            ]))

            # Mahony 필터 업데이트
            self.q = self.mahony.updateIMU(q=self.q, acc=accel_data, gyr=gyro_data)
            r = R.from_quat([self.q[1], self.q[2], self.q[3], self.q[0]])
            roll, pitch, yaw = r.as_euler('zyx', degrees=True)

            h, w = color_image.shape[:2]
            center_distance = depth_frame.get_distance(w // 2, h // 2)

            # 중앙 십 Roll, Pitch 리셋
            if abs(roll) < self.CENTER_THRESHOLD and abs(pitch) < self.CENTER_THRESHOLD:
                if self.center_start_time is None:
                    self.center_start_time = time.time()  # 중앙 시간 기록 
                else:
                    elapsed_time = time.time() - self.center_start_time
                    if elapsed_time >= self.center_hold_time:  # 일정 시간 이상 
                        # Roll과 Pitch 리셋
                        roll = 0
                        pitch = 0
                        self.get_logger().info("Roll, Pitch 리셋됨!")
            else:
                self.center_start_time = None  

            self.display_hud(color_image, roll, pitch, yaw, center_distance)

            # FPS 계산
            self.frame_count += 1
            current_time = time.time()
            elapsed = current_time - self.last_time
            if elapsed >= 1.0:
                fps = self.frame_count / elapsed
                self.get_logger().info(f"[FPS] {fps:.2f}")
                self.frame_count = 0
                self.last_time = current_time

        except Exception as e:
            self.get_logger().warn(f"HUD 생성 중 오류: {e}")

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

    def display_hud(self, image, roll, pitch, yaw, center_distance):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        line_length = 100

        pitch_offset = int(pitch * 2)
        roll_angle = -roll

        hud_layer = np.zeros_like(image)

        cv2.line(hud_layer,
                 (center[0] - line_length, center[1] + pitch_offset),
                 (center[0] + line_length, center[1] + pitch_offset),
                 (0, 255, 255), 2)

        M = cv2.getRotationMatrix2D(center, roll_angle, 1.0)
        hud_layer = cv2.warpAffine(hud_layer, M, (w, h))

        cv2.drawMarker(hud_layer, center, (0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)

        cv2.putText(hud_layer, f"Roll : {roll:+06.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(hud_layer, f"Pitch: {pitch:+06.2f}", (240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(hud_layer, f"Yaw  : {yaw:+06.2f}", (440, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(hud_layer, f"Center Distance: {center_distance:.2f} m", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        if center_distance < 0.5 and center_distance > 0:
            cv2.putText(hud_layer, "WARNING: Too Close!", (w // 2 - 100, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        if abs(roll) > self.ROLL_THRESHOLD or abs(pitch) > self.PITCH_THRESHOLD:
            cv2.putText(hud_layer, "WARNING:  Limit Twist!",
                        (w // 2 - 200, h // 2 - 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 3)

        self.draw_yaw_side_scale(hud_layer, yaw)
        hud_image = cv2.add(image, hud_layer)

        cv2.imshow("original", image.copy())
        cv2.imshow("HUD only", hud_layer)
        cv2.imshow("D435i HUD", hud_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = D435iHUDNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import math


class YoloDepthIMUNode(Node):
    def __init__(self):
        super().__init__('yolo_depth_imu_node')

        # YOLOv8 모델
        self.model = YOLO('/home/ljh/goodbox-project/train_goodbox_highacc/weights/goodbox.pt')
        self.model.conf = 0.5

        # RealSense 초기화
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        self.pipeline.start(config)

        # RealSense 필터
        self.temporal_filter = rs.temporal_filter()
        self.spatial_filter = rs.spatial_filter()
        self.hole_filling_filter = rs.hole_filling_filter()

        # CV Bridge
        self.bridge = CvBridge()

        # IMU 구독
        qos_profile = QoSProfile(depth=2)
        self.imu_sub = self.create_subscription(
            Float32MultiArray,
            'imu_data',
            self.imu_callback,
            qos_profile
        )
        self.imu_angles = [0.0, 0.0, 0.0]  # x, y, z 각도

        # 타이머 콜백
        self.timer = self.create_timer(0.1, self.process_frame)  # 10Hz

    def imu_callback(self, msg):
        # IMU에서 x,y,z 각도(가속도계 기반) 구독
        self.imu_angles = msg.data

    def process_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return

        # 깊이 필터 적용
        depth_frame = self.temporal_filter.process(depth_frame).as_depth_frame()
        depth_frame = self.spatial_filter.process(depth_frame).as_depth_frame()
        depth_frame = self.hole_filling_filter.process(depth_frame).as_depth_frame()

        color_image = np.asanyarray(color_frame.get_data())

        # YOLO 예측
        results = self.model.predict(color_image, conf=0.5, verbose=False, max_det=1)

        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            u = int((x1 + x2) / 2)
            v = int((y1 + y2) / 2)
            depth = depth_frame.get_distance(u, v)

            # 카메라 Intrinsics
            intr = color_frame.profile.as_video_stream_profile().intrinsics
            point_3d = rs.rs2_deproject_pixel_to_point(intr, [u, v], depth)
            X, Y, Z = point_3d  # 카메라 좌표계

            # ----- IMU 회전 보정 -----
            x_angle, y_angle, z_angle = self.imu_angles
            rx, ry, rz = np.deg2rad([x_angle, y_angle, z_angle])

            Rx = np.array([[1, 0, 0],
                           [0, np.cos(rx), -np.sin(rx)],
                           [0, np.sin(rx), np.cos(rx)]])
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                           [0, 1, 0],
                           [-np.sin(ry), 0, np.cos(ry)]])
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                           [np.sin(rz), np.cos(rz), 0],
                           [0, 0, 1]])

            R = Rz @ Ry @ Rx
            point_cam = np.array([[X], [Y], [Z]])
            point_world = R @ point_cam
            Xw, Yw, Zw = point_world.flatten()

            # 출력
            self.get_logger().info(
                f"Object (u,v)=({u},{v}), 3D raw=({X:.2f},{Y:.2f},{Z:.2f}), corrected=({Xw:.2f},{Yw:.2f},{Zw:.2f})"
            )

            # 시각화
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(color_image, (u, v), 4, (0, 0, 255), -1)
            cv2.putText(color_image, f"Xw:{Xw:.2f} Yw:{Yw:.2f} Zw:{Zw:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 225, 0), 2)

        cv2.imshow("YOLO + Depth + IMU", color_image)
        cv2.waitKey(1)

    def destroy_node(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloDepthIMUNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

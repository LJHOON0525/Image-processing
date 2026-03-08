import rclpy
from rclpy.node import Node
import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class YoloDepthNode(Node):
    def __init__(self):
        super().__init__('yolo_depth_node')

        # YOLOv8 모델 불러오기
        self.model = YOLO('/home/ljh/goodbox-project/train_goodbox_highacc/weights/goodbox.pt')
        self.model.conf = 0.5

        # RealSense 초기화
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)

        # RealSense 필터 초기화
        self.temporal_filter = rs.temporal_filter()
        self.spatial_filter = rs.spatial_filter()
        self.hole_filling_filter = rs.hole_filling_filter()

        # 브리지 초기화
        self.bridge = CvBridge()

        # 타이머 콜백
        self.timer = self.create_timer(0.1, self.process_frame)  # 10Hz

    def process_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return

        # 깊이 프레임 필터 적용
        depth_frame = self.temporal_filter.process(depth_frame).as_depth_frame()
        depth_frame = self.spatial_filter.process(depth_frame).as_depth_frame()
        depth_frame = self.hole_filling_filter.process(depth_frame).as_depth_frame()

        # 이미지 가져오기
        color_image = np.asanyarray(color_frame.get_data())

        # 모델 예측
        results = self.model.predict(color_image, conf=0.5, verbose=False, max_det=1)

        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            u = int((x1 + x2) / 2)
            v = int((y1 + y2) / 2)
            depth = depth_frame.get_distance(u, v)

            # Intrinsics 가져오기
            intr = color_frame.profile.as_video_stream_profile().intrinsics
            point_3d = rs.rs2_deproject_pixel_to_point(intr, [u, v], depth)
            X, Y, Z = point_3d

            # 출력
            self.get_logger().info(f"Detected object at (u,v)=({u},{v}), 3D=({X:.2f}, {Y:.2f}, {Z:.2f})")

            # 시각화
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(color_image, (u, v), 4, (0, 0, 255), -1)
            cv2.putText(color_image, f"X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 225, 0), 2)

        # 화면 출력
        cv2.imshow("YOLO + Depth", color_image)
        cv2.waitKey(1)

    def destroy_node(self):
        self.pipeline.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

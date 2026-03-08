#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, String
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO

class BirdDistancePub(Node):
    def __init__(self):
        super().__init__('bird_distance_node')
        qos_profile = QoSProfile(depth=10)

        # Publishers
        self.depth_frame_pub = self.create_publisher(Image, 'depth_data', qos_profile)
        self.color_frame_pub = self.create_publisher(Image, 'color_data', qos_profile)
        self.distance_data_pub = self.create_publisher(Float64, 'distance_data', qos_profile)
        self.avoid_pub = self.create_publisher(String, 'avoid_direction', qos_profile)

        # RealSense 설정
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        depth_profile = self.pipeline.start(self.config)
        depth_sensor = depth_profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.clipping_distance = 1.0 / self.depth_scale  # 1미터 기준

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.model = YOLO("yolov8n.pt")  # 전체 모델 로드

        self.cvbrid = CvBridge()
        self.timer = self.create_timer(1/30, self.process_frames)

        self.get_logger().info("감지 노드 시작 (YOLO + D435i Depth)")

    def process_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            self.get_logger().warn("프레임을 가져오지 못했습니다.")
            return

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Bird(클래스 14)만 감지
        results = self.model.predict(color_image, conf=0.4, max_det=3, classes=[14])
        result = results[0]
        annotated_img = result.plot()  # 이제 Bird만 표시됨

        if len(result.boxes):
            box = result.boxes.xywh[0].cpu().numpy().astype(int)
            x_center, y_center = box[0], box[1]

            if 0 <= x_center < depth_image.shape[1] and 0 <= y_center < depth_image.shape[0]:
                distance = depth_image[y_center, x_center] * self.depth_scale
                self.get_logger().info(f"거리: {distance:.3f} m")

                # 거리 데이터 publish
                distance_msg = Float64()
                distance_msg.data = float(distance)
                self.distance_data_pub.publish(distance_msg)

                # 회피 판단
                if distance < 2.5:
                    avoid_msg = String()
                    if x_center < depth_image.shape[1] // 2:
                        avoid_msg.data = "RIGHT"
                    else:
                        avoid_msg.data = "LEFT"
                    self.avoid_pub.publish(avoid_msg)
                    self.get_logger().warn(f"[AVOID] 장애물 감지, 방향: {avoid_msg.data}")
        else:
            self.get_logger().info("14번 클래스(Bird) 감지되지 않음.")

        # 이미지 publish
        self.color_frame_pub.publish(self.cvbrid.cv2_to_imgmsg(annotated_img, encoding="bgr8"))
        cv2.imshow("YOLOv8 Bird Detection", annotated_img)
        cv2.waitKey(1)

    def destroy_node(self):
        super().destroy_node()
        self.pipeline.stop()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = BirdDistancePub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("종료 요청됨 (Ctrl+C)")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
import cv2

class RealSenseMultiCamera(Node):
    def __init__(self):
        super().__init__('realsense_multi_camera')

        # ROS2 퍼블리셔 설정
        self.bridge = CvBridge()
        self.color_pub_1 = self.create_publisher(Image, '/camera_1/color_image', 10)
        self.depth_pub_1 = self.create_publisher(Image, '/camera_1/depth_image', 10)
        self.color_pub_2 = self.create_publisher(Image, '/camera_2/color_image', 10)
        self.depth_pub_2 = self.create_publisher(Image, '/camera_2/depth_image', 10)

        # 두 개의 RealSense 카메라 파이프라인 초기화
        self.pipeline_1 = rs.pipeline()
        self.pipeline_2 = rs.pipeline()
        self.config_1 = rs.config()
        self.config_2 = rs.config()

        # 각 카메라에 대해 스트림 설정
        self.config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth 활성화
        self.config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color 활성화
        self.config_2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth 활성화
        self.config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color 활성화

        # 두 카메라 시작
        self.pipeline_1.start(self.config_1)
        self.pipeline_2.start(self.config_2)

        # 두 카메라의 데이터를 정렬하기 위한 align 객체 생성
        self.align = rs.align(rs.stream.color)

        # 타이머로 30FPS 실행
        self.timer = self.create_timer(1 / 30.0, self.process_frames)

    def process_frames(self):
        # 두 카메라의 프레임 얻기
        frames_1 = self.pipeline_1.wait_for_frames()
        frames_2 = self.pipeline_2.wait_for_frames()

        # 데이터 정렬
        aligned_frames_1 = self.align.process(frames_1)
        aligned_frames_2 = self.align.process(frames_2)

        # 각 카메라에서 깊이와 컬러 프레임 추출
        depth_frame_1 = aligned_frames_1.get_depth_frame()
        color_frame_1 = aligned_frames_1.get_color_frame()
        depth_frame_2 = aligned_frames_2.get_depth_frame()
        color_frame_2 = aligned_frames_2.get_color_frame()

        if not depth_frame_1 or not color_frame_1 or not depth_frame_2 or not color_frame_2:
            return

        # 프레임을 NumPy 배열로 변환
        depth_image_1 = np.asanyarray(depth_frame_1.get_data())
        color_image_1 = np.asanyarray(color_frame_1.get_data())
        depth_image_2 = np.asanyarray(depth_frame_2.get_data())
        color_image_2 = np.asanyarray(color_frame_2.get_data())

        # ROS2 퍼블리시
        self.color_pub_1.publish(self.bridge.cv2_to_imgmsg(color_image_1, encoding="bgr8"))
        self.depth_pub_1.publish(self.bridge.cv2_to_imgmsg(depth_image_1, encoding="mono16"))
        self.color_pub_2.publish(self.bridge.cv2_to_imgmsg(color_image_2, encoding="bgr8"))
        self.depth_pub_2.publish(self.bridge.cv2_to_imgmsg(depth_image_2, encoding="mono16"))

        # 디버깅용 출력
        cv2.imshow("Camera 1 Color", color_image_1)
        cv2.imshow("Camera 2 Color", color_image_2)
        cv2.waitKey(1)

    def destroy_node(self):
        self.pipeline_1.stop()
        self.pipeline_2.stop()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RealSenseMultiCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np

class RealSensePublisher(Node):
    def __init__(self):
        super().__init__('realsense_publisher')

        qos_profile = rclpy.qos.QoSProfile(depth=10)

        # 퍼블리셔 생성 (토픽명은 RTAB-Map 실행 시 지정한 것과 동일하게)
        self.color_pub = self.create_publisher(Image, '/camera/camera/color/image_raw', qos_profile)
        self.depth_pub = self.create_publisher(Image, '/camera/camera/depth/image_rect_raw', qos_profile)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/camera/color/camera_info', qos_profile)

        self.bridge = CvBridge()

        # RealSense 파이프라인 설정
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.pipeline.start(config)

        # 카메라 내부 파라미터 수동 세팅 (실제 카메라 파라미터에 맞게 수정 필요)
        self.camera_info_msg = CameraInfo()
        self.camera_info_msg.header.frame_id = 'camera_link'
        self.camera_info_msg.width = 640
        self.camera_info_msg.height = 480
        # 기본 D435i Intrinsics 예시 (fx, fy, cx, cy)
        fx = 615.0
        fy = 615.0
        cx = 320.0
        cy = 240.0
        self.camera_info_msg.k = [fx, 0.0, cx,
                                 0.0, fy, cy,
                                 0.0, 0.0, 1.0]
        # Distortion 모델과 계수 (D435i 기본값 예시)
        self.camera_info_msg.distortion_model = 'plumb_bob'
        # Distortion 계수 (float형 리스트로)
        self.camera_info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]


        # 타이머로 30Hz 주기 호출
        self.timer = self.create_timer(1/30, self.publish_frames)

    def publish_frames(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            self.get_logger().warn("No frame received")
            return

        # Color 이미지
        color_image = np.asanyarray(color_frame.get_data())
        color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
        color_msg.header.stamp = self.get_clock().now().to_msg()
        color_msg.header.frame_id = 'camera_link'

        # Depth 이미지 (16bit)
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='mono16')
        depth_msg.header.stamp = color_msg.header.stamp
        depth_msg.header.frame_id = 'camera_link'

        # 카메라 정보 헤더 타임스탬프 동기화
        self.camera_info_msg.header.stamp = color_msg.header.stamp

        # 퍼블리시
        self.color_pub.publish(color_msg)
        self.depth_pub.publish(depth_msg)
        self.camera_info_pub.publish(self.camera_info_msg)

        self.get_logger().info('Published color, depth, and camera info frames')

    def destroy_node(self):
        self.pipeline.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RealSensePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

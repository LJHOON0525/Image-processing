import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
from ultralytics import YOLO
from std_msgs.msg import Bool, Float32MultiArray
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import pyrealsense2 as rs
import numpy as np


class YOLODepthWebcamNode(Node):
    def __init__(self):
        super().__init__('yolo_depth_webcam_node')

        box_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # YOLO 모델 로드 (모델 1개 기준)
        self.model = YOLO("/home/ljh/goodbox-project/train_goodbox_highacc/weights/goodbox.pt")

        # 플래그 퍼블리셔
        self.box_pub = self.create_publisher(Bool, "box_detect", box_qos_profile)
        self.image_pub = self.create_publisher(Image, 'box_image', 10)
        # 3D 좌표 퍼블리셔
        self.box_coor_pub = self.create_publisher(Float32MultiArray, "box_coordinate", box_qos_profile)

        self.bridge = CvBridge()

        # 해상도
        self.img_size_x = 848
        self.img_size_y = 480

        # ROI 영역 설정
        self.x1_roi = int(self.img_size_x * 0.05)
        self.y1_roi = int(self.img_size_y * 0.7)
        self.x2_roi = int(self.img_size_x * 0.95)
        self.y2_roi = int(self.img_size_y * 0.9)

        # RealSense 파이프라인 설정
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.img_size_x, self.img_size_y, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, self.img_size_x, self.img_size_y, rs.format.z16, 30)
        self.profile = self.pipeline.start(config)

        # Depth → Color 정렬
        self.align = rs.align(rs.stream.color)

        # Depth 카메라 Intrinsics
        self.depth_intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

        # 타이머 (30 FPS)
        self.timer = self.create_timer(1/30, self.timer_callback)

    def _in_roi(self, u, v):
        return self.x1_roi <= u <= self.x2_roi and self.y1_roi <= v <= self.y2_roi

    def timer_callback(self):
        # 프레임 읽기
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return

        color_image = np.asanyarray(color_frame.get_data())

        # YOLO 추론 (한 프레임당 1개 객체)
        results = self.model.predict(color_image, conf=0.4, verbose=False, max_det=1)

        # ROI 안에서 감지된 객체 플래그 초기화
        box_flag = False

        # YOLO 박스 시각화 & ROI 체크
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            u = int((x1 + x2) / 2)
            v = int((y1 + y2) / 2)

            in_roi = self._in_roi(u, v)

            if in_roi:
                depth = depth_frame.get_distance(u, v)
                if depth != 0:
                    X, Y, Z = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [u, v], depth)
                    msg_3d = Float32MultiArray()
                    msg_3d.data = [X, Y, Z, 1.0]
                    self.box_coor_pub.publish(msg_3d)

                # ROI 안이면 플래그 발행
                box_flag = True
                self.get_logger().info("파란 감지")

                color = (136, 175, 0)  # ROI 안 색
            else:
                color = (128, 128, 128)  # ROI 밖 색

            # 시각화
            cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
            cv2.circle(color_image, (u, v), 4, (0, 0, 255), -1)
            if in_roi and depth != 0:
                cv2.putText(color_image, f"X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 225, 0), 2)
                

        # ROI 표시
        cv2.rectangle(color_image, (self.x1_roi, self.y1_roi), (self.x2_roi, self.y2_roi), (255, 0, 0), 2)
        ros_image = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
        self.image_pub.publish(ros_image)

        cv2.imshow("YOLO Depth + Flag Detection", color_image)

        # 플래그 퍼블리시
        self.box_pub.publish(Bool(data=box_flag))

        # ESC 키로 종료
        if cv2.waitKey(1) & 0xFF == 27:
            self.destroy_node()
            rclpy.shutdown()

    def destroy_node(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YOLODepthWebcamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt")
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()

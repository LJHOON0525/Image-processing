#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import cv2


class YOLOWebcamNode(Node):
    def __init__(self):
        super().__init__('yolo_webcam_node')

        # QoS 설정
        img_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # CvBridge
        self.cvbridge = CvBridge()

        # 구독
        self.sub_img_data = self.create_subscription(
            Image,
            '/web_handle_image',
            self.img_data_callback,
            img_qos_profile
        )

        # 퍼블리셔
        self.img_publisher = self.create_publisher(
            CompressedImage,
            'web_handle_image/compressed',
            img_qos_profile
        )
        self.detect_publisher = self.create_publisher(
            Bool,
            'web_handle_detect',
            img_qos_profile
        )

        # YOLO 모델 로드
        self.model = YOLO("/home/ljh/goodbox-project/train_goodbox_highacc/weights/doorhandle2.pt")

        # 웹캠 초기화
        self.cap = cv2.VideoCapture(2)
        if not self.cap.isOpened():
            self.get_logger().error("웹캠 열 수 없음 (카메라 2번 확인 필요)")
            self.cleanup()
            return

        # 타이머
        self.timer = self.create_timer(1/30, self.process_frame)

    # ---------- 이미지 압축 퍼블리시 ----------
    def _publish_compressed(self, msg: Image, publisher, topic_name: str):
        try:
            cv_image = self.cvbridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
            success, encoded_image = cv2.imencode('.jpg', cv_image, encode_param)

            if not success:
                self.get_logger().error(f"JPEG encoding 실패: {topic_name}")
                return

            compressed = CompressedImage()
            compressed.header = msg.header
            compressed.format = "jpeg"
            compressed.data = encoded_image.tobytes()
            publisher.publish(compressed)

        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error at {topic_name}: {e}")
        except Exception as e:
            self.get_logger().error(f"Compression error at {topic_name}: {e}")

    # ---------- 이미지 구독 콜백 ----------
    def img_data_callback(self, msg: Image):
        self._publish_compressed(msg, self.img_publisher, "web_handle_image/compressed")

    # ---------- 웹캠 프레임 처리 ----------
    def process_frame(self):
        if not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("웹캠 프레임 없음")
            return

        # YOLO 예측
        results = self.model.predict(frame, conf=0.5, classes=[1], verbose=False, max_det=1)
        annotated = results[0].plot() if results else frame

        # Bool 퍼블리시
        detect_flag = len(results[0].boxes) > 0 if results else False
        self.detect_publisher.publish(Bool(data=detect_flag))

        # 화면 출력
        cv2.imshow("YOLO Webcam Handle Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("'q' 키 눌러 종료")
            self.cleanup()

    # ---------- 종료 처리 ----------
    def cleanup(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.destroy_node()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = YOLOWebcamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt")
    finally:
        node.cleanup()


if __name__ == '__main__':
    main()

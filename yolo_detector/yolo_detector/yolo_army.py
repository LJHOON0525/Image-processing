import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class YoloDetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_detection_node')

        # 모델, 브릿지, 클래스 초기화
        self.bridge = CvBridge()
        self.model = YOLO("/home/ljh/goodbox-project/train_goodbox_highacc/weights/army.pt")
        self.class_names = {0: "army", 1: "enemy"}

        # QoS 설정
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.publisher_ = self.create_publisher(Image, 'yolo_detection_image', qos_profile)

        # 카메라 해상도 및 FPS 설정
        self.img_size_x = 640
        self.img_size_y = 480
        self.frame_rate = 10

        # 두 카메라 초기화
        self.cap_right = cv2.VideoCapture(2)
        self.cap_left = cv2.VideoCapture(4)

        for cap in [self.cap_right, self.cap_left]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size_x)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size_y)
            cap.set(cv2.CAP_PROP_FPS, self.frame_rate)

        # 타이머 설정
        self.timer = self.create_timer(0.1, self.timer_callback)

    def process_frame(self, frame):
        # YOLO 추론
        result = self.model.predict(frame, conf=0.4, verbose=False, max_det=1)
        result_frame = frame.copy()

        # 결과 처리 및 시각화
        for r in result[0].boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            label_id = int(r.cls[0])
            conf = float(r.conf[0])
            label_name = self.class_names.get(label_id, f"class_{label_id}")

            if label_name == "army":
                color = (0, 255, 0)
            elif label_name == "enemy":
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)

            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result_frame, f"{label_name}:{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return result_frame

    def timer_callback(self):
        # 두 카메라 프레임 읽기
        ret_right, frame_right = self.cap_right.read()
        ret_left, frame_left = self.cap_left.read()

        if not ret_right or not ret_left:
            self.get_logger().error("카메라 프레임 읽기 실패")
            return

        # YOLO 감지
        processed_right = self.process_frame(frame_right)
        processed_left = self.process_frame(frame_left)

        # 좌우 프레임 합치기 (np.hstack 사용)
        combined_frame = np.hstack([processed_left, processed_right])

        # ROS2 이미지 퍼블리시
        ros_image = self.bridge.cv2_to_imgmsg(combined_frame, encoding='bgr8')
        self.publisher_.publish(ros_image)
        self.get_logger().info("YOLO Detection 이미지 퍼블리시 완료")

        # OpenCV로 출력
        cv2.imshow("YOLO Detection (Left + Right)", combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("'q' 키가 눌려서 종료합니다.")
            self.cleanup()

    def cleanup(self):
        # 자원 해제
        self.cap_left.release()
        self.cap_right.release()
        cv2.destroyAllWindows()
        rclpy.shutdown()

    def destroy_node(self):
        self.cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
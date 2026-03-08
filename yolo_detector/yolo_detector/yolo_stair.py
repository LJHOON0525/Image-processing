import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class StairDetectionNode(Node):
    def __init__(self):
        super().__init__('stair_detection_node')

        # QoS 설정 (실시간 이미지 스트리밍용)
        stair_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 퍼블리셔
        self.image_pub = self.create_publisher(CompressedImage, 'stair_image', stair_qos_profile)
        self.stair_flag_pub = self.create_publisher(Bool, 'stair_detected', stair_qos_profile)

        # YOLO 모델 로드
        self.model = YOLO("/home/ljh/goodbox-project/train_goodbox_highacc/weights/stair.pt")

        # 카메라 열기
        self.cap = cv2.VideoCapture(2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # CvBridge 초기화
        self.bridge = CvBridge()

        self.detect_count = 0
        self.detect_threshold = 3

        # 타이머 (0.1초 간격)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("카메라 프레임 읽기 실패")
            return

        # YOLO 예측
        results = self.model.predict(frame, conf=0.7, verbose=False, max_det=5)
        result_frame = results[0].plot()

        stair_boxes = [box for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls) if int(cls) == 1]       
        detected = len(stair_boxes) > 0

        # 항상 발행
        flag_msg = Bool()
        flag_msg.data = detected
        self.stair_flag_pub.publish(flag_msg)

        # 로그
        if detected:
            self.get_logger().info("계단 감지")
        else:
            self.get_logger().info("계단 없음")

        # 이미지 ROS 메시지 변환 및 발행
        ros_image = self.bridge.cv2_to_compressed_imgmsg(result_frame, dst_format='jpeg')
        self.image_pub.publish(ros_image)

        cv2.imshow("STAIR", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("'q' 키가 눌려 종료합니다.")
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = StairDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt (SIGINT)")
    finally:
        node.cap.release()
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
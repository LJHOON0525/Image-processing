import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32MultiArray, Bool
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2


class YOLOButtonHandleNode(Node):
    def __init__(self):
        super().__init__('yolo_button_handle_node')

        # QoS 설정
        qos_profile = QoSProfile(depth=10)
        img_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ---------------- 퍼블리셔 ----------------
        self.blue_pub = self.create_publisher(Bool, 'blue_detect', qos_profile)
        self.green_pub = self.create_publisher(Bool, 'green_detect', qos_profile)
        self.yellow_pub = self.create_publisher(Bool, 'yellow_detect', qos_profile)
        self.button_coord_pub = self.create_publisher(Float32MultiArray, 'button_coordinate', qos_profile)
        self.handle_coord_pub = self.create_publisher(Float32MultiArray, 'handle_coordinate', qos_profile)
        self.raw_pub = self.create_publisher(Image, 'yolo_combined_image/raw', img_qos_profile)
        self.compressed_pub = self.create_publisher(CompressedImage, 'yolo_combined_image/compressed', img_qos_profile)

        # ---------------- CvBridge ----------------
        self.bridge = CvBridge()

        # ---------------- YOLO 모델 ----------------
        self.button_model = YOLO("/home/ljh/goodbox-project/train_goodbox_highacc/weights/button.pt")
        self.handle_model = YOLO("/home/ljh/goodbox-project/train_goodbox_highacc/weights/doorhandle2.pt")

        # ---------------- 웹캠 ----------------
        self.cap = cv2.VideoCapture(2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 848)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            self.get_logger().error("웹캠 열기 실패 (카메라 2번 확인)")
            rclpy.shutdown()
            return

        # ---------------- ROI ----------------
        self.x1_roi = int(848 * 0.05)
        self.y1_roi = int(480 * 0.7)
        self.x2_roi = int(848 * 0.95)
        self.y2_roi = int(480 * 0.9)

        # ---------------- 상태 초기화 ----------------
        self.button_detected_flag = False

        # ---------------- 타이머 (30fps) ----------------
        self.timer = self.create_timer(1/30, self.timer_callback)

    def _in_roi(self, u, v):
        return self.x1_roi <= u <= self.x2_roi and self.y1_roi <= v <= self.y2_roi

    def _publish_compressed(self, cv_image):
        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
            success, encoded_image = cv2.imencode('.jpg', cv_image, encode_param)
            if not success:
                self.get_logger().error("JPEG 인코딩 실패")
                return
            compressed = CompressedImage()
            compressed.header.stamp = self.get_clock().now().to_msg()
            compressed.format = "jpeg"
            compressed.data = encoded_image.tobytes()
            self.compressed_pub.publish(compressed)
        except Exception as e:
            self.get_logger().error(f"압축 이미지 퍼블리시 오류: {e}")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("웹캠 프레임 없음")
            return

        # 원본 이미지 퍼블리시
        self.raw_pub.publish(self.bridge.cv2_to_imgmsg(frame, encoding="bgr8"))

        # ---------------- 버튼 YOLO ----------------
        button_results = self.button_model.predict(frame, conf=0.4, classes=[0,1,2], verbose=False, max_det=1)
        blue_flag = green_flag = yellow_flag = False
        button_coords = []

        for i, box in enumerate(button_results[0].boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            u = int((x1 + x2)/2)
            v = int((y1 + y2)/2)
            cls_id = int(button_results[0].boxes.cls[i].item())

            in_roi = self._in_roi(u, v)
            color = (136, 175, 0) if in_roi else (128, 128, 128)

            # 플래그 설정
            if in_roi:
                if cls_id == 0: blue_flag = True
                elif cls_id == 1: green_flag = True
                elif cls_id == 2: yellow_flag = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (u, v), 4, (0, 0, 255), -1)
            button_coords.append(u)
            button_coords.append(v)

        cv2.rectangle(frame, (self.x1_roi, self.y1_roi), (self.x2_roi, self.y2_roi), (255, 0, 0), 2)

        self.blue_pub.publish(Bool(data=blue_flag))
        self.green_pub.publish(Bool(data=green_flag))
        self.yellow_pub.publish(Bool(data=yellow_flag))

        if blue_flag or green_flag or yellow_flag:
            self.button_detected_flag = True

        # ---------------- 핸들 YOLO ----------------
        handle_coords = []
        if self.button_detected_flag:
            handle_results = self.handle_model.predict(frame, conf=0.5, classes=[1], verbose=False, max_det=1)
            for box in handle_results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                u = int((x1 + x2)/2)
                v = int((y1 + y2)/2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(frame, (u, v), 4, (0, 255, 0), -1)
                handle_coords.append(u)
                handle_coords.append(v)

        # 좌표 퍼블리시
        self.button_coord_pub.publish(Float32MultiArray(data=button_coords))
        self.handle_coord_pub.publish(Float32MultiArray(data=handle_coords))

        # 압축 이미지 퍼블리시
        self._publish_compressed(frame)

        # 화면 출력
        cv2.imshow("YOLO Button+Handle", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cleanup_and_exit()

    def cleanup_and_exit(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = YOLOButtonHandleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt")
    finally:
        node.cleanup_and_exit()


if __name__ == "__main__":
    main()
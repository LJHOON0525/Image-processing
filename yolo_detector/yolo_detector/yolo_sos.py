import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np

class YoloDetectionHUDNode(Node):
    def __init__(self):
        super().__init__('yolo_detection_hud_node')

        # QoS 설정 (실시간 이미지 스트리밍용)
        sos_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # YOLO 모델
        self.model = YOLO("/home/ljh/goodbox-project/train_goodbox_highacc/weights/sos1.pt")

        # 카메라 초기화
        self.cap = cv2.VideoCapture(2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # ROS 퍼블리셔
        self.image_pub = self.create_publisher(CompressedImage, 'sos_image', sos_qos_profile)
        self.flag_pub = self.create_publisher(Bool, 'sos_flag', sos_qos_profile)
        self.bridge = CvBridge()

        # 프레임 중심
        self.center = (320, 240)

        # 한 번이라도 감지 여부
        self.detected_once = False  

        # 타이머 (약 30fps)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("카메라 프레임 읽기 실패")
            return

        detected_now = False

        # YOLO 탐지
        results = self.model.predict(frame, conf=0.4, verbose=False, max_det=1)

        for result in results:
            if len(result.boxes) > 0:
                detected_now = True
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box.astype(int)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # 바운딩 박스, 중심점
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # 프레임 중심과 연결선
                cv2.line(frame, self.center, (cx, cy), (255, 255, 0), 2)

                # 중심점 오프셋 표시
                offset_x = cx - self.center[0]
                offset_y = cy - self.center[1]
                cv2.putText(frame,
                            f"Offset X:{offset_x}  Y:{offset_y}",
                            (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)

        # 기준 십자 표시
        cv2.drawMarker(frame, self.center, (255, 255, 255),
                       cv2.MARKER_CROSS, 20, 2)

        # 한 번이라도 감지되면 계속 True
        if detected_now:
            self.detected_once = True

        flag_msg = Bool()
        flag_msg.data = self.detected_once
        self.flag_pub.publish(flag_msg)

        # ROS 이미지 퍼블리시 (HUD 표시된 프레임)
        try:
            ros_image = self.bridge.cv2_to_compressed_imgmsg(frame, dst_format='jpeg')
            self.image_pub.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f"Image publishing failed: {e}")

        # 로그 출력
        self.get_logger().info(f"객체 인식 상태 → flag={self.detected_once}")

        # OpenCV 창 표시
        cv2.imshow("YOLO Detection HUD", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.destroy_node()
            rclpy.shutdown()

    def destroy_node(self):
        super().destroy_node()
        self.cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectionHUDNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

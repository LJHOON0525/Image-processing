import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import math


class YoloDetectionHUDNode(Node):
    def __init__(self):
        super().__init__('yolo_detection_hud_node')

        # YOLO 모델
        self.model = YOLO('yolov8n.pt')

        # 웹캠 초기화 (비디오 2번, 해상도 1920x1080)
        self.cap = cv2.VideoCapture(2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # ROS 퍼블리셔
        self.image_pub = self.create_publisher(Image, 'yolo_detection_image', 10)
        self.flag_pub = self.create_publisher(Bool, 'robotdog_detection_flag', 10)
        self.coord_pub = self.create_publisher(Float32MultiArray, 'robotdog_coordinates', 10)
        self.bridge = CvBridge()

        # 프레임 중심
        self.center = (960, 540)

        # Tracking 관련
        self.tracker = None
        self.tracking = False
        self.found_once = False
        self.last_box = None  # 마지막으로 잡은 박스 저장

        # 타이머
        self.timer = self.create_timer(0.033, self.timer_callback)

    # IoU 계산
    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
        if boxAArea + boxBArea - interArea == 0:
            return 0
        return interArea / float(boxAArea + boxBArea - interArea)

    def start_tracker(self, frame, x1, y1, x2, y2):
        """트래커 초기화"""
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
        self.tracking = True
        self.found_once = True
        self.last_box = [x1, y1, x2, y2]

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("웹캠에서 프레임을 가져오지 못했습니다.")
            return

        detected = False
        roi_zone = "NONE"
        coord_msg = Float32MultiArray()
        coord_msg.data = []

        h, w, _ = frame.shape
        left_bound = w // 3
        right_bound = 2 * w // 3
        cv2.line(frame, (left_bound, 0), (left_bound, h), (255, 0, 0), 2)
        cv2.line(frame, (right_bound, 0), (right_bound, h), (255, 0, 0), 2)

        # ---------------- YOLO 또는 트래킹 ----------------
        if not self.tracking:
            results = self.model.predict(frame, classes=[0], conf=0.5, max_det=1)
            for result in results:
                for box in result.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box.astype(int)
                    if not self.found_once or (self.last_box and self.iou([x1, y1, x2, y2], self.last_box) > 0.5):
                        detected = True
                        self.start_tracker(frame, x1, y1, x2, y2)
                        break
                if detected:
                    break
        else:
            # Tracker 업데이트
            success, bbox = self.tracker.update(frame)
            if success:
                x, y, w_box, h_box = [int(v) for v in bbox]
                x1, y1, x2, y2 = x, y, x + w_box, y + h_box

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if cx < left_bound:
                    self.tracking = False
                    self.last_box = None
                    x1 = y1 = x2 = y2 = None
                else:
                    detected = True
                    self.last_box = [x1, y1, x2, y2]
            else:
                self.tracking = False
                x1 = y1 = x2 = y2 = None

        # ---------------- HUD ----------------
        if detected and x1 is not None:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if cx < left_bound:
                roi_zone = "LEFT"
            elif cx < right_bound:
                roi_zone = "CENTER"
            else:
                roi_zone = "RIGHT"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.line(frame, self.center, (cx, cy), (255, 255, 0), 2)
            cv2.putText(frame, "person detect", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 깊이 정보 없으므로 Z=0 처리
            coord_msg.data = [float(cx), float(cy), 0.0]

        cv2.drawMarker(frame, self.center, (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

        flag_msg = Bool()
        flag_msg.data = detected
        self.flag_pub.publish(flag_msg)

        if coord_msg.data:
            self.coord_pub.publish(coord_msg)

        try:
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.image_pub.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f"Image publishing failed: {e}")

        cv2.imshow("YOLO+Tracker Webcam HUD", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.destroy_node()
            rclpy.shutdown()

    def destroy_node(self):
        super().destroy_node()
        if hasattr(self, 'cap'):
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

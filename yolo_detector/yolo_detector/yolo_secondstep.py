from ultralytics import YOLO
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np

class YOLODetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        self.publisher = self.create_publisher(Image, 'yolo_detected_image', 10)
        self.detection_publisher = self.create_publisher(String, 'yolo_detections', 10)

        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(2)  # USB 카메라 사용

        self.model = YOLO('yolov8n.pt')  # YOLO 모델 로드
        self.timer = self.create_timer(1/15, self.detect_objects)  # 15FPS 

    def detect_objects(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("카메라에서 프레임을 읽을 수 없습니다.")
            return

        results = self.model(frame)
        result = results[0]

        detection_list = []  # 감지된 객체 리스트

        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표
            confidences = result.boxes.conf.cpu().numpy()  # 신뢰도 점수
            classes = result.boxes.cls.cpu().numpy().astype(int)  # 클래스 ID

            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])
                conf = float(confidences[i])
                cls = classes[i]
                label = self.model.names[cls]  # 클래스 이름

                # 바운딩 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 감지된 객체 리스트 추가
                detection_list.append(f'{{"label": "{label}", "confidence": {conf:.2f}, "bbox": [{x1}, {y1}, {x2}, {y2}]}}')

        # 감지된 객체 정보 퍼블리시 (JSON 형식)
        detection_msg = String()
        detection_msg.data = "[" + ", ".join(detection_list) + "]"
        self.detection_publisher.publish(detection_msg)

        # ROS2 이미지 퍼블리시
        img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.publisher.publish(img_msg)

        # 화면 출력
        cv2.imshow("YOLO Detection", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YOLODetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

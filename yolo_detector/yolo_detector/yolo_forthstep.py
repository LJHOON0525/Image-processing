import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class YoloDetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_detection_node')
        
        # 모델 로드
        self.model = YOLO("/home/ljh/goodbox-project/train_goodbox_highacc/weights/stair.pt")
        
        # 카메라 열기
        self.cap = cv2.VideoCapture(2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # 이미지 퍼블리셔 설정
        self.publisher_ = self.create_publisher(Image, 'yolo_detection_image', 10)
        
        # CvBridge 초기화
        self.bridge = CvBridge()
        
        # 주기적으로 실행될 타이머 생성
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error(" 카메라 프레임 읽기 실패")
            return

        # YOLO 모델을 사용하여 예측 (classes, conf, verbose, max_det 추가)
        self.result = self.model.predict(frame, conf=0.7, verbose=False, max_det=5)
        
        # 결과 이미지 가져오기
        result_frame = self.result[0].plot()

        # ROS 메시지로 변환
        ros_image = self.bridge.cv2_to_imgmsg(result_frame, encoding='bgr8')
        
        # 이미지 퍼블리시
        self.publisher_.publish(ros_image)
        self.get_logger().info("YOLO Detection 이미지 퍼블리시 완료")

        # OpenCV로 감지된 이미지 출력
        cv2.imshow("YOLO Detection", result_frame)

        # 'q' 키를 눌러 창을 닫을 수 있게 처리
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info(" 'q' 키가 눌려서 종료합니다.")
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)

    yolo_detection_node = YoloDetectionNode()

    rclpy.spin(yolo_detection_node)

    # 종료 시 자원 해제
    yolo_detection_node.cap.release()
    yolo_detection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

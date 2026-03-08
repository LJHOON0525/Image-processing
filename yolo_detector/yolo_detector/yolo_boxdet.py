import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class ArmyDetectionNode(Node):
    def __init__(self):
        super().__init__('army_detection_node')
        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")  # 작은 따옴표 오류 수정
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1  
        )
        self.publisher = self.create_publisher(Image, 'side_camera', qos_profile)
        
        ### parameter setting ###
        self.img_size_x = 1280
        self.img_size_y = 720
        self.frame_rate = 10
        #########################
        
        self.cap0 = cv2.VideoCapture(2)  # cam0
        self.cap1 = cv2.VideoCapture(4)  # cam1
        self.cvbrid = CvBridge()
        
        self.cap0.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size_x)
        self.cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size_y)
        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size_x)
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size_y)
        self.cap0.set(cv2.CAP_PROP_FPS, self.frame_rate)
        self.cap1.set(cv2.CAP_PROP_FPS, self.frame_rate)
        
        self.timer_finder = self.create_timer(1/self.frame_rate, self.image_callback)

    def image_callback(self):
        ret0, img0 = self.cap0.read()
        ret1, img1 = self.cap1.read()
        
        if not (ret0 and ret1):
            if not ret0:
                print(f'cam0 cannot connect')
            if not ret1:
                print(f'cam1 cannot connect')
            return
        
        frame = img0
        frame1 = img1
        
        # 사람(class 0)만 탐지
        results = self.model.predict(frame, classes=[0], conf=0.4, verbose=False)
        
        if len(results[0].boxes.cls):
            for box in results[0].boxes:
                label = int(box.cls.item())
                confidence = box.conf.item()
                object_xyxy = np.array(box.xyxy.detach().cpu().numpy().tolist()[0], dtype='int')
                
                # 사람일 때만 표시
                if label == 0:
                    color = (0, 255, 0)  # 초록색
                    cv2.putText(frame, f'Person {(confidence*100):.2f}%', 
                                (object_xyxy[0], object_xyxy[1] - 20),
                                cv2.FONT_ITALIC, 1, color, 2)
                    cv2.rectangle(frame, (object_xyxy[0], object_xyxy[1]), 
                                  (object_xyxy[2], object_xyxy[3]), color, 2)
        
        cv2.imshow("cam0", frame)
        cv2.imshow("cam1", frame1)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ArmyDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

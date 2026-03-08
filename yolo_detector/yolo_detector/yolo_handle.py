import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2
import time

class MultiYOLONode(Node):
    def __init__(self):
        super().__init__('multi_yolo_node')

        # ===== RealSense 설정 =====
        self.img_size_x = 848
        self.img_size_y = 480
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.img_size_x, self.img_size_y, rs.format.bgr8, 15)
        self.config.enable_stream(rs.stream.depth, self.img_size_x, self.img_size_y, rs.format.z16, 15)
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        time.sleep(2)  # 카메라 안정화

        # ===== CvBridge =====
        self.bridge = CvBridge()

        # ===== YOLO 모델 로드 =====
        self.model_person = YOLO("yolov8n.pt")  # 사람
        self.model_box = YOLO("/home/ljh/goodbox-project/train_goodbox_highacc/weights/goodbox.pt")  # 박스
        self.model_number = YOLO("/home/ljh/goodbox-project/train_goodbox_highacc/weights/number.pt")  # 숫자

        # ===== HSV ROI 상태 판정 파라미터 =====
        self.RED1_LOWER = np.array([0, 100, 80], dtype=np.uint8)
        self.RED1_UPPER = np.array([10, 255, 255], dtype=np.uint8)
        self.RED2_LOWER = np.array([170, 100, 80], dtype=np.uint8)
        self.RED2_UPPER = np.array([180, 255, 255], dtype=np.uint8)
        self.BLUE_LOWER = np.array([100, 120, 60], dtype=np.uint8)
        self.BLUE_UPPER = np.array([130, 255, 255], dtype=np.uint8)
        self.morph_kernel_size = 5
        self.morph_iter = 2
        self.min_area_ratio = 0.02
        self.ROI_W_SCALE = 0.6
        self.ROI_H_SCALE = 0.6
        self.ROI_Y_OFFSET = -0.1

        # ===== 퍼블리셔 =====
        img_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.img_pub = self.create_publisher(Image, "multi_yolo_image", img_qos)
        self.person_pub = self.create_publisher(Bool, "person_detected", 10)
        self.one_pub = self.create_publisher(Bool, "one_detect", 10)
        self.two_pub = self.create_publisher(Bool, "two_detect", 10)
        self.three_pub = self.create_publisher(Bool, "three_detect", 10)
        self.roi_status_pub = self.create_publisher(String, "roi_status", 10)

        # ===== 타이머 =====
        self.timer = self.create_timer(1/15, self.process_frame)

    # ---------- HSV 헬퍼 ----------
    def _morph_clean(self, mask):
        if self.morph_iter <= 0:
            return mask
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=self.morph_iter)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=self.morph_iter)
        return mask

    def hsv_masks(self, img_bgr):
        blur = cv2.GaussianBlur(img_bgr, (3,3), 1)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, self.RED1_LOWER, self.RED1_UPPER)
        red2 = cv2.inRange(hsv, self.RED2_LOWER, self.RED2_UPPER)
        red_mask = cv2.bitwise_or(red1, red2)
        blue_mask = cv2.inRange(hsv, self.BLUE_LOWER, self.BLUE_UPPER)
        red_mask = self._morph_clean(red_mask)
        blue_mask = self._morph_clean(blue_mask)
        return red_mask, blue_mask

    def _clip_box(self, x1,y1,x2,y2,w,h):
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        if x2 <= x1: x2 = min(x1+1, w-1)
        if y2 <= y1: y2 = min(y1+1, h-1)
        return x1,y1,x2,y2

    def _roi_from_person(self, x1,y1,x2,y2,img_h,img_w):
        bw = x2 - x1
        bh = y2 - y1
        cx = (x1+x2)/2
        cy = (y1+y2)/2
        roi_w = int(bw*self.ROI_W_SCALE)
        roi_h = int(bh*self.ROI_H_SCALE)
        cy_shift = cy + self.ROI_Y_OFFSET*bh
        rx1 = int(cx - roi_w/2)
        ry1 = int(cy_shift - roi_h/2)
        rx2 = rx1 + roi_w
        ry2 = ry1 + roi_h
        return self._clip_box(rx1,ry1,rx2,ry2,img_w,img_h)

    def _largest_area(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return 0 if not contours else max(cv2.contourArea(c) for c in contours)

    def decide_status_in_roi(self, roi_bgr):
        red_mask, blue_mask = self.hsv_masks(roi_bgr)
        roi_area = roi_bgr.shape[0]*roi_bgr.shape[1]
        min_area_px = max(1,int(self.min_area_ratio*roi_area))
        def zero_small(mask):
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            cleaned = np.zeros_like(mask)
            for i in range(1,num_labels):
                if stats[i,cv2.CC_STAT_AREA] >= min_area_px:
                    cleaned[labels==i]=255
            return cleaned
        red_mask = zero_small(red_mask)
        blue_mask = zero_small(blue_mask)
        red_area = self._largest_area(red_mask)
        blue_area = self._largest_area(blue_mask)
        if red_area==0 and blue_area==0:
            return "stranger", red_mask, blue_mask
        if red_area >= blue_area:
            return "dead", red_mask, blue_mask
        else:
            return "survivor", red_mask, blue_mask

    # ---------- 메인 프로세싱 ----------
    def process_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            self.get_logger().warn("프레임 없음")
            return

        frame = np.asanyarray(color_frame.get_data())
        annotated_frame = frame.copy()
        img_h, img_w = frame.shape[:2]

        # 1️⃣ 사람 감지
        person_flag = False
        results_person = self.model_person.predict(frame, conf=0.6, classes=[0], verbose=False, max_det=1)
        roi_status_msg = "none"
        for box in results_person[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.tolist())
            annotated_frame = cv2.rectangle(annotated_frame,(x1,y1),(x2,y2),(0,255,0),2)
            annotated_frame = cv2.putText(annotated_frame,"person",(x1,max(y1-5,0)),
                                          cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            person_flag = True
            rx1, ry1, rx2, ry2 = self._roi_from_person(x1,y1,x2,y2,img_h,img_w)
            roi_bgr = frame[ry1:ry2, rx1:rx2]
            status, _, _ = self.decide_status_in_roi(roi_bgr)
            roi_status_msg = status
            roi_color = {"dead":(0,0,255),"survivor":(255,0,0),"stranger":(128,128,128)}[status]
            annotated_frame = cv2.rectangle(annotated_frame,(rx1,ry1),(rx2,ry2),roi_color,2)
            annotated_frame = cv2.putText(annotated_frame,status,(rx1,max(ry1-5,0)),
                                          cv2.FONT_HERSHEY_SIMPLEX,0.5,roi_color,2)

        self.person_pub.publish(Bool(data=person_flag))
        self.roi_status_pub.publish(String(data=roi_status_msg))

        # 2️⃣ 박스 감지
        results_box = self.model_box.predict(frame, conf=0.6, verbose=False, max_det=1)
        for box in results_box[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.tolist())
            annotated_frame = cv2.rectangle(annotated_frame,(x1,y1),(x2,y2),(255,0,0),2)

        # 3️⃣ 숫자 감지
        results_number = self.model_number.predict(frame, conf=0.6, classes=[0,1,2], verbose=False, max_det=1)
        one_flag = two_flag = three_flag = False
        for box in results_number[0].boxes:
            cls_id = int(box.cls[0].item())
            if cls_id==0: one_flag=True
            elif cls_id==1: two_flag=True
            elif cls_id==2: three_flag=True
        self.one_pub.publish(Bool(data=one_flag))
        self.two_pub.publish(Bool(data=two_flag))
        self.three_pub.publish(Bool(data=three_flag))

        # YOLO plot
        annotated_frame = results_number[0].plot()  # 마지막 숫자 결과로 시각화

        # 퍼블리시 및 OpenCV 표시
        self.img_pub.publish(self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8"))
        cv2.imshow("Multi YOLO Detection", annotated_frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC 종료
            self.get_logger().info("ESC 눌림, 종료합니다.")
            self.shutdown_node()

    def shutdown_node(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = MultiYOLONode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shutdown_node()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
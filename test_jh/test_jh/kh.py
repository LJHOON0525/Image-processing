import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from ultralytics import YOLO

class Lanedecttest(Node):
    def __init__(self):
        super().__init__('spring_color_checker')

        # ===== 기본 설정 =====
        self.img_size_x = 640
        self.img_size_y = 480

        # YOLO (사람만 감지)
        self.conf_thres = 0.4 # 이거 신뢰도다
        self.model = YOLO('yolov8n.pt')  # COCO, class 0 = person

        # ===== HSV 임계값 =====
        self.RED1_LOWER = np.array([0,   100,  80], dtype=np.uint8)
        self.RED1_UPPER = np.array([10,  255, 255], dtype=np.uint8)
        self.RED2_LOWER = np.array([170, 100,  80], dtype=np.uint8)
        self.RED2_UPPER = np.array([180, 255, 255], dtype=np.uint8)
        self.BLUE_LOWER = np.array([100, 120,  60], dtype=np.uint8)
        self.BLUE_UPPER = np.array([130, 255, 255], dtype=np.uint8)

        # ===== 노이즈 억제 파라미터 =====
        self.morph_kernel_size = 5
        self.morph_iter = 2
        self.min_area_ratio = 0.02  # ROI 면적 대비 최소 면적(2%) 이하면 무시

        # ===== ROI (상체) 스케일 =====
        self.ROI_W_SCALE = 0.6
        self.ROI_H_SCALE = 0.6
        self.ROI_Y_OFFSET = -0.1

        # OpenCV webcam
        self.cap = cv2.VideoCapture(2)   # USB 카메라 번호 (2번은 환경에 따라 변경)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size_x)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size_y)

        # 주기적으로 실행할 타이머 (30fps)
        self.timer = self.create_timer(1/30, self.image_processing)

    # ---------- 마스크/노이즈 ----------
    def _morph_clean(self, mask):
        if self.morph_iter <= 0:
            return mask
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=self.morph_iter)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=self.morph_iter)
        return mask

    def hsv_masks(self, img_bgr):
        blur = cv2.GaussianBlur(img_bgr, (3, 3), 1)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, self.RED1_LOWER, self.RED1_UPPER)
        red2 = cv2.inRange(hsv, self.RED2_LOWER, self.RED2_UPPER)
        red_mask = cv2.bitwise_or(red1, red2)
        blue_mask = cv2.inRange(hsv, self.BLUE_LOWER, self.BLUE_UPPER)
        red_mask = self._morph_clean(red_mask)
        blue_mask = self._morph_clean(blue_mask)
        return red_mask, blue_mask

    def _clip_box(self, x1, y1, x2, y2, w, h):
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1: x2 = min(x1 + 1, w - 1)
        if y2 <= y1: y2 = min(y1 + 1, h - 1)
        return x1, y1, x2, y2

    def _roi_from_person(self, x1, y1, x2, y2, img_h, img_w):
        bw = x2 - x1
        bh = y2 - y1
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        roi_w = int(bw * self.ROI_W_SCALE)
        roi_h = int(bh * self.ROI_H_SCALE)

        cy_shift = cy + self.ROI_Y_OFFSET * bh

        rx1 = int(cx - roi_w / 2)
        ry1 = int(cy_shift - roi_h / 2)
        rx2 = rx1 + roi_w
        ry2 = ry1 + roi_h

        return self._clip_box(rx1, ry1, rx2, ry2, img_w, img_h)

    def _largest_area(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return 0 if not contours else max(cv2.contourArea(c) for c in contours)

    def decide_status_in_roi(self, roi_bgr):
        red_mask, blue_mask = self.hsv_masks(roi_bgr)
        roi_area = roi_bgr.shape[0] * roi_bgr.shape[1]
        min_area_px = max(1, int(self.min_area_ratio * roi_area))

        def zero_small(mask):
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            cleaned = np.zeros_like(mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
                    cleaned[labels == i] = 255
            return cleaned

        red_mask = zero_small(red_mask)
        blue_mask = zero_small(blue_mask)

        red_area = self._largest_area(red_mask)
        blue_area = self._largest_area(blue_mask)

        if red_area == 0 and blue_area == 0:
            return "stranger", red_mask, blue_mask
        if red_area >= blue_area:
            return "dead", red_mask, blue_mask
        else:
            return "survivor", red_mask, blue_mask

    def image_processing(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        self.color_img = frame
        img_h, img_w = frame.shape[:2]

        original_viz = frame.copy()
        filt_overlay = np.zeros_like(frame)

        results = self.model.predict(frame, classes=[0], conf=self.conf_thres, verbose=False, max_det=1)
        person_boxes = []
        if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.xyxy is not None:
            for b, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
                if int(cls.item()) != 0:
                    continue
                x1, y1, x2, y2 = map(int, b.tolist())
                person_boxes.append((x1, y1, x2, y2, float(conf.item())))

        for (x1, y1, x2, y2, conf) in person_boxes:
            cv2.rectangle(original_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original_viz, f'person {conf:.2f}', (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            rx1, ry1, rx2, ry2 = self._roi_from_person(x1, y1, x2, y2, img_h, img_w)
            cv2.rectangle(original_viz, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

            roi_bgr = frame[ry1:ry2, rx1:rx2]
            status, red_mask_roi, blue_mask_roi = self.decide_status_in_roi(roi_bgr)

            if status == "dead":
                txt_color = (0, 0, 255)
            elif status == "survivor":
                txt_color = (255, 0, 0)
            else:
                txt_color = (128, 128, 128)
            cv2.putText(original_viz, f'ROI: {status}', (rx1, max(ry1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 2, cv2.LINE_AA)

        cv2.imshow("Webcam (YOLO + ROI)", original_viz)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = Lanedecttest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

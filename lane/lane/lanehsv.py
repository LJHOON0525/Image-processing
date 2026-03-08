#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class LaneTracingDualCamYellow(Node):
    def __init__(self):
        super().__init__('webcam_lane_tracing_dual_yellow')

        # --- pubs ---
        self.center_pub = self.create_publisher(Float32, 'center_x', 10)
        self.err_pub    = self.create_publisher(Float32, 'center_error', 10)
        self.image_pub  = self.create_publisher(Image, 'lane_image', 10)
        self.bridge = CvBridge()

        # --- cams ---
        self.cap1 = cv2.VideoCapture(2)
        self.cap2 = cv2.VideoCapture(4)
        for cap in (self.cap1, self.cap2):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

        # --- state ---
        self.last_center = 320
        self.center_smooth = 320
        self.w, self.h = 640, 480

        # --- params ---
        self.roi_top_ratio = 0.78
        self.edge_margin    = 20
        # Canny
        self.canny_low, self.canny_high = 60, 150
        # Hough
        self.hough_thresh     = 55
        self.min_line_length  = 60
        self.max_line_gap     = 12
        # 차선 각도 필터(도)
        self.min_angle_deg    = 20
        self.max_angle_deg    = 80
        # contour smoothing
        self.alpha = 0.2

        self.timer = self.create_timer(1/30, self.timer_cb)
        self.get_logger().info("DualCam Yellow Lane Tracing Initialized")

    # ---------- utility ----------
    def _draw_roi_box(self, dbg, y1):
        h, w = dbg.shape[:2]
        cv2.rectangle(dbg, (0, y1), (w, h), (0, 255, 0), 2)
        cv2.line(dbg, (self.edge_margin, y1), (self.edge_margin, h), (0, 0, 255), 1)
        cv2.line(dbg, (w - self.edge_margin, y1), (w - self.edge_margin, h), (0, 0, 255), 1)

    def _line_eq(self, x1, y1, x2, y2):
        if x2 == x1:
            return None
        m = (y2 - y1) / float(x2 - x1)
        b = y1 - m * x1
        return m, b

    def _pick_best(self, cands):
        if not cands:
            return None
        # 가장 긴 선
        cands.sort(key=lambda t: t[2], reverse=True)
        return cands[0]

    # ---------- single camera processing ----------
    def process_one(self, cap, cam_idx=0):
        ok, frame = cap.read()
        if not ok or frame is None:
            self.get_logger().warn(f"[CAM{cam_idx}] frame read failed")
            return None

        h, w = frame.shape[:2]
        self.h, self.w = h, w

        y_top = int(h * self.roi_top_ratio)
        roi = frame[y_top:h, :]

        # --- 노란색 필터링 (HSV) ---
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([18, 130, 165])
        upper_yellow = np.array([45, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        blur = cv2.GaussianBlur(mask, (5, 5), 0)
        blur = cv2.morphologyEx(blur, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))  # 잡음 제거######################3
        blur = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

        # --- Hough Lines ---
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=self.hough_thresh,
                                minLineLength=self.min_line_length,
                                maxLineGap=self.max_line_gap)

        left_cands, right_cands = [], []
        line_only = np.zeros_like(frame)
        if lines is not None:
            for (x1, y1, x2, y2) in lines[:, 0]:
                dx, dy = (x2 - x1), (y2 - y1)
                if dx == 0:
                    continue
                angle = np.degrees(np.arctan2(dy, dx))
                a = abs(angle)
                if not (self.min_angle_deg <= a <= self.max_angle_deg):
                    continue
                length = float(np.hypot(dx, dy))
                slope = dy / float(dx)
                xmid = 0.5 * (x1 + x2)
                if slope < 0 and xmid < w * 0.5:
                    left_cands.append(((x1, y1, x2, y2), slope, length))
                elif slope > 0 and xmid > w * 0.5:
                    right_cands.append(((x1, y1, x2, y2), slope, length))

                # 라인만 보이도록 frame에 그리기
                cv2.line(line_only[y_top:h, :], (x1, y1), (x2, y2), (0, 255, 255), 2)

        dbg = frame.copy()
        self._draw_roi_box(dbg, y_top)

        left = self._pick_best(left_cands)
        right = self._pick_best(right_cands)

        y_bottom = (h - y_top - 1)
        xL = xR = None

        if left is not None:
            (x1l, y1l, x2l, y2l), _, _ = left
            cv2.line(dbg, (x1l, y1l + y_top), (x2l, y2l + y_top), (255, 0, 0), 3)
            eqL = self._line_eq(x1l, y1l, x2l, y2l)
            if eqL:
                m, b = eqL
                xL = int((y_bottom - b) / m)

        if right is not None:
            (x1r, y1r, x2r, y2r), _, _ = right
            cv2.line(dbg, (x1r, y1r + y_top), (x2r, y2r + y_top), (0, 0, 255), 3)
            eqR = self._line_eq(x1r, y1r, x2r, y2r)
            if eqR:
                m, b = eqR
                xR = int((y_bottom - b) / m)

        cx = None
        conf = 0.0

        # --- contour 기반 보정 ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 100#선갯수
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        if max_contour is not None and max_area > 50: 
            # ROI 기반 mask
            max_contour_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(max_contour_mask, [max_contour], -1, 255, thickness=cv2.FILLED)

            # ROI bitwise_and
            filtered_roi = cv2.bitwise_and(roi, roi, mask=max_contour_mask)

            # 원본 frame에 ROI 덮어쓰기
            filtered = frame.copy()
            filtered[y_top:h, :] = filtered_roi

            cv2.imshow("Filtered Contour", filtered)
            cv2.imshow("Lines Only", line_only)
            cv2.waitKey(1)

            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(max_contour)
            left_c = x_rect
            right_c = x_rect + w_rect
            contour_center = int((left_c + right_c) / 2)

            # smoothing
            self.center_smooth = int((1 - self.alpha) * self.center_smooth + self.alpha * contour_center)
            if cx is None:
                cx = self.center_smooth
            else:
                cx = int(0.5 * (cx + self.center_smooth))

        # --- Hough 라인 중심 계산 ---
        if xL is not None and xR is not None and (xR - xL) > 12:
            cx = int(0.5 * (xL + xR))
            conf = 1.0
            cv2.circle(dbg, (xL, h - 1), 5, (255, 0, 0), -1)
            cv2.circle(dbg, (xR, h - 1), 5, (0, 0, 255), -1)
            cv2.line(dbg, (cx, y_top), (cx, h), (0, 255, 255), 2)
        else:
            if xL is not None:
                cx = int(0.5 * (xL + w * 0.5)); conf = 0.5
            elif xR is not None:
                cx = int(0.5 * (xR + w * 0.5)); conf = 0.4 ##우선도

        if cx is None:
            self.get_logger().info(f"[CAM{cam_idx}] no valid lane lines")
            return {'frame': frame, 'debug': dbg, 'center': None, 'confidence': 0.0}

        cx = int(np.clip(cx, self.edge_margin, w - self.edge_margin - 1))
        self.get_logger().info(f"[CAM{cam_idx}] xL={xL} xR={xR} center={cx} conf={conf:.2f}")
        return {'frame': frame, 'debug': dbg, 'center': cx, 'confidence': float(conf)}

    # ---------- main loop ----------
    def timer_cb(self):
        r1 = self.process_one(self.cap1, cam_idx=1)
        r2 = self.process_one(self.cap2, cam_idx=2)

        centers, confs, debugs = [], [], []
        for r in (r1, r2):
            if r is not None and r['center'] is not None and r['confidence'] > 0.0:
                centers.append(r['center']); confs.append(r['confidence'])
            if r is not None and r['debug'] is not None:
                debugs.append(r['debug'])

        if confs:
            confs = np.array(confs, dtype=np.float32)
            cx = int(np.dot(centers, confs) / max(confs.sum(), 1e-6))
            self.last_center = cx
        else:
            cx = self.last_center
            self.get_logger().warn("Both cameras lost the lane; keeping last center.")

        combined = None
        if len(debugs) == 2:
            combined = np.hstack((debugs[0], debugs[1]))
        elif len(debugs) == 1:
            combined = debugs[0]
        if combined is not None:
            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(combined, encoding='bgr8'))
            except Exception as e:
                self.get_logger().error(f"Image publish failed: {e}")

        cv2.imshow("Lane Tracing DualCam Yellow", combined)
        cv2.waitKey(1)

        self.center_pub.publish(Float32(data=float(cx)))
        err = (cx - (self.w / 2.0)) / (self.w / 2.0)
        self.err_pub.publish(Float32(data=float(err)))

    # ---------- shutdown ----------
    def destroy_node(self):
        super().destroy_node()
        for cap in (self.cap1, self.cap2):
            try: cap.release()
            except: pass
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = LaneTracingDualCamYellow()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

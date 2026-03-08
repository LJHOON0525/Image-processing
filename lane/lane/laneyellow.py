#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class LaneTracingDualCamYellow(Node):
    def __init__(self):
        super().__init__('webcam_lane_tracing_dual_yellow')

        # --- pubs ---
        self.center_pub = self.create_publisher(Float32MultiArray, 'center_x', 10)
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
        self.edge_margin = 20
        self.alpha = 0.2  # smoothing factor

        self.timer = self.create_timer(1/30, self.timer_cb)
        self.get_logger().info("DualCam Yellow Lane Tracing (Contour-based) Initialized")

    # ---------- contour 중심 계산 ----------
    def process_one(self, cap, cam_idx=0):
        ok, frame = cap.read()
        if not ok or frame is None:
            self.get_logger().warn(f"[CAM{cam_idx}] frame read failed")
            return None

        h, w = frame.shape[:2]
        self.h, self.w = h, w

        y_top = int(h * self.roi_top_ratio)
        roi = frame[y_top:h, :]

        # --- 노란색 필터링 ---
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([18, 130, 165])
        upper_yellow = np.array([45, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

        # --- 컨투어 검출 ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        left_sum = right_sum = 0
        left_center = right_center = None
        max_area = 50
        max_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        cx = None
        if max_contour is not None:
            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(max_contour)
            contour_center = int(x_rect + w_rect / 2)
            if contour_center < w / 2:
                left_center = contour_center
                left_sum = int(np.sum(mask[:, x_rect:x_rect + w_rect]) / 255)
            else:
                right_center = contour_center
                right_sum = int(np.sum(mask[:, x_rect:x_rect + w_rect]) / 255)

            # smoothing
            self.center_smooth = int((1 - self.alpha) * self.center_smooth + self.alpha * contour_center)
            cx = self.center_smooth

            # 시각화
            filtered_roi = cv2.bitwise_and(roi, roi, mask=mask)
            frame[y_top:h, :] = filtered_roi

        # --- 결과 퍼블리시 ---
        return {
            'frame': frame,
            'center': cx,
            'left_sum': left_sum,
            'right_sum': right_sum
        }

    # ---------- main loop ----------
    def timer_cb(self):
        r1 = self.process_one(self.cap1, cam_idx=1)
        r2 = self.process_one(self.cap2, cam_idx=2)

        centers = []
        left_sum_total = 0
        right_sum_total = 0
        combined = None

        for r in (r1, r2):
            if r is not None:
                if r['center'] is not None:
                    centers.append(r['center'])
                left_sum_total += r['left_sum']
                right_sum_total += r['right_sum']
                if combined is None:
                    combined = r['frame']
                else:
                    combined = np.hstack((combined, r['frame']))

        if centers:
            cx = int(np.mean(centers))
            self.last_center = cx
        else:
            cx = self.last_center
            self.get_logger().warn("Both cameras lost lane; using last center.")

        # 퍼블리시
        if cx is not None:
            center_msg = Float32MultiArray()
            center_msg.data = [float(cx)]
            self.center_pub.publish(center_msg)

            err = (cx - (self.w / 2.0)) / (self.w / 2.0)
            self.err_pub.publish(Float32(data=float(err)))

        if combined is not None:
            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(combined, encoding='bgr8'))
            except Exception as e:
                self.get_logger().error(f"Image publish failed: {e}")

        cv2.imshow("DualCam Lane Tracing (Yellow Contour)", combined)
        cv2.waitKey(1)

    # ---------- shutdown ----------
    def destroy_node(self):
        super().destroy_node()
        for cap in (self.cap1, self.cap2):
            try:
                cap.release()
            except:
                pass
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

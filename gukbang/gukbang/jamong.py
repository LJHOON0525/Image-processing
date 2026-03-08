import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import pyrealsense2 as rs
import math


class YoloDetectionHUDNode(Node):
    def __init__(self):
        super().__init__('yolo_detection_hud_node')

        # YOLO 모델 (사람 class=0)
        self.model = YOLO('yolov8n.pt')

        # RealSense 초기화
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(config)

        # Depth intrinsics
        depth_stream = self.profile.get_stream(rs.stream.depth)
        self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

        # Align 객체
        self.align = rs.align(rs.stream.color)

        # 필터
        self.temporal_filter = rs.temporal_filter()
        self.spatial_filter = rs.spatial_filter()
        self.hole_filling_filter = rs.hole_filling_filter()

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
        self.last_box = None

        # 모니터 해상도 (예: FHD)
        self.screen_w, self.screen_h = 1920, 1080

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

    # def start_tracker(self, frame, x1, y1, x2, y2):
    #     """트래커 초기화"""
    #     self.tracker = cv2.TrackerCSRT_create()
    #     self.tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
    #     self.tracking = True
    #     self.found_once = True
    #     self.last_box = [x1, y1, x2, y2]

    def timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            self.get_logger().warning("프레임을 가져오지 못했습니다.")
            return

        filtered_depth = self.temporal_filter.process(aligned_depth_frame)
        filtered_depth = self.spatial_filter.process(filtered_depth)
        self.filled_depth_frame = self.hole_filling_filter.process(filtered_depth).as_depth_frame()

        frame = np.asanyarray(color_frame.get_data())
        detected = False
        # roi_zone = "NONE"
        coord_msg = Float32MultiArray()
        coord_msg.data = []

        h, w, _ = frame.shape
        left_bound = w // 3
        right_bound = 2 * w // 3
        #cv2.line(frame, (left_bound, 0), (left_bound, h), (255, 0, 0), 2)
        #cv2.line(frame, (right_bound, 0), (right_bound, h), (255, 0, 0), 2)

        # ---------------- YOLO 또는 트래킹 ----------------
        if not self.tracking:
            results = self.model.predict(frame, classes=[0], conf=0.5, max_det=5)
            closest_box = None
            closest_dist = float('inf')

            for result in results:
                for box in result.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box.astype(int)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    depth = self.filled_depth_frame.get_distance(cx, cy)
                    if depth > 0 and depth < closest_dist:
                        closest_dist = depth
                        closest_box = [x1, y1, x2, y2]

            if closest_box is not None:
                x1, y1, x2, y2 = closest_box
                detected = True
                self.start_tracker(frame, x1, y1, x2, y2)

        else:
            # Tracker 업데이트
            success, bbox = self.tracker.update(frame)
            if success:
                x, y, w_box, h_box = [int(v) for v in bbox]
                x1, y1, x2, y2 = x, y, x + w_box, y + h_box
                cx = int((x1 + x2) / 2)

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

        # ---------------- HUD & 좌표 계산 ----------------
        if detected and x1 is not None:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if cx < left_bound:
                roi_zone = "LEFT"
            elif cx < right_bound:
                roi_zone = "CENTER"
            else:
                roi_zone = "RIGHT"

            try:
                depth = self.filled_depth_frame.get_distance(cx, cy)
                point_3d = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [cx, cy], depth)
                X, Y, Z = point_3d
            except RuntimeError:
                X = Y = Z = depth = 0.0

            try:
                depth_center = self.filled_depth_frame.get_distance(*self.center)
                point_center_3d = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, list(self.center), depth_center)
                dist_3d = math.sqrt((X - point_center_3d[0])**2 +
                                    (Y - point_center_3d[1])**2 +
                                    (Z - point_center_3d[2])**2)
            except RuntimeError:
                dist_3d = 0.0

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.line(frame, self.center, (cx, cy), (255, 255, 0), 2)
            cv2.putText(frame, f"person | {roi_zone} | X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f} Dist:{dist_3d:.2f}m",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

            coord_msg.data = [float(X), float(Y), float(Z), float(dist_3d)]
            self.get_logger().info(
                f"Tracking 좌표: ({cx}, {cy}), ROI={roi_zone}, 3D 위치: X={X:.2f} Y={Y:.2f} Z={Z:.2f}, 거리={dist_3d:.2f}m"
            )

        #cv2.drawMarker(frame, self.center, (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

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

        # 모니터 해상도에 맞춰 리사이즈해서 출력
        resized_frame = cv2.resize(frame, (self.screen_w, self.screen_h))
        cv2.imshow("YOLO+Tracker Depth HUD", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.destroy_node()
            rclpy.shutdown()

    def destroy_node(self):
        super().destroy_node()
        self.pipeline.stop()
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

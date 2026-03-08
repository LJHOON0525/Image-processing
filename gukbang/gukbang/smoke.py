import cv2
import numpy as np
import pyrealsense2 as rs

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_frame = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)
        pixel = hsv_frame[y, x]
        h, s, v = pixel
        print(f"HSV at ({x}, {y}): H={h}, S={s}, V={v}")

# RealSense 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

cv2.namedWindow("Frame")

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())

    cv2.imshow("Frame", frame)
    cv2.setMouseCallback("Frame", mouse_callback, frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

pipeline.stop()
cv2.destroyAllWindows()

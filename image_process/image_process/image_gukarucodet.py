import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2.aruco as aruco

class ArucoDetection(Node):
    def __init__(self):
        super().__init__('image_gukarucodet')
        self.bridge = CvBridge()
        
        qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1  
        )
        
        self.publisher = self.create_publisher(Image, 'image_raw', qos_profile)
        
        ### parameter setting ###
        self.img_size_x = 640
        self.img_size_y = 480
        self.frame_rate = 10
        
        ########################
        
        ### params ###
        self.threshold = 120
        ##############
        
        ### aruco setting ###
        
        self.aruco_dict = aruco.Dictionary(9, 5)
        self.aruco_dict.bytesList = np.empty(shape=(9, 4, 4), dtype=np.uint8)
        
        self.markers = [
            [[1,0,0,0,1],[1,0,0,1,0],[1,1,1,0,0],[1,0,0,1,0],[1,0,0,0,1]],  # K
            [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],  # O
            [[1,1,1,1,0],[1,0,0,0,1],[1,1,1,1,0],[1,0,0,1,0],[1,0,0,0,1]],  # R
            [[1,1,1,1,1],[1,0,0,0,0],[1,1,1,1,0],[1,0,0,0,0],[1,1,1,1,1]],  # E
            [[0,0,1,0,0],[0,1,0,1,0],[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1]],  # A
            [[1,1,1,1,0],[1,0,0,0,1],[1,1,1,1,0],[1,0,0,1,0],[1,0,0,0,1]],  # R
            [[1,0,0,0,1],[1,1,0,1,1],[1,0,1,0,1],[1,0,0,0,1],[1,0,0,0,1]],  # M
            [[1,0,0,0,1],[1,1,0,1,0],[1,0,1,0,0],[1,0,0,0,1],[1,0,0,0,1]],  # Y
            [[0,1,0,1,0],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]]   # heart
        ]
        
        for i, bits in enumerate(self.markers):
            mybits = np.array(bits, dtype=np.uint8)
            self.aruco_dict.bytesList[i] = aruco.Dictionary.getByteListFromBits(mybits)
        self.marker_chars = ["K", "O", "R", "E", "A", "R", "M", "Y", "Heart"]    
        self.aruco_param = aruco.DetectorParameters()
        
        #########################
        
        
        self.cap = cv2.VideoCapture(2)  # 단일 카메라 설정
        self.cvbrid = CvBridge()
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size_x)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size_y)
        self.cap.set(cv2.CAP_PROP_FPS, self.frame_rate)
        
        self.timer_finder = self.create_timer(1/self.frame_rate, self.image_callback)

    def image_callback(self):
        
        ret, img = self.cap.read()
        
        if not ret :
            print(f'Camera cannot connect')
        else :
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            bw = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)[1]
            
            corners, ids, points = aruco.detectMarkers(bw, self.aruco_dict, parameters=self.aruco_param)
            
            img = aruco.drawDetectedMarkers(img, corners)
                
            # Draw custom text
            if ids is not None:
                for i in range(len(ids)):
                    corner = corners[i][0]
                    top_left = (int(corner[0][0]), int(corner[0][1]))
                    bottom_right = (int(corner[2][0]), int(corner[2][1]))
                    
                    marker_id = ids[i][0]
                    if marker_id < len(self.marker_chars):
                        cv2.putText(img, self.marker_chars[marker_id], top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            resized = cv2.resize(img, (int(self.img_size_x/2), int(self.img_size_y)), interpolation=cv2.INTER_AREA)
            self.publisher.publish(self.cvbrid.cv2_to_imgmsg(resized))
        
        cv2.imshow('frame', img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetection()
    try :
        rclpy.spin(node)
    except KeyboardInterrupt :
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally :
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

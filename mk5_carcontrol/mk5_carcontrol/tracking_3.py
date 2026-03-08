

#Camera & IMU => Tracking => odrive
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
import time

class tracking(Node):
    def __init__(self):
        super().__init__('tracking')
        qos_profile = QoSProfile(depth=10)


        #motor encoder feedback
        #self.encoder_subscriber = self.create_subscription(Float32MultiArray, 'Odrive_encoder', self.encoder_clear, qos_profile)

        self.drive_command = Float32MultiArray()
        self.control_publisher = self.create_publisher(
            Float32MultiArray,
            'Odrive_control',  
            qos_profile
        )



################################################# ODRIVE MODE#################################################
        self.odrive_mode = 1.0

################################################# DIRECTION #################################################
        #Direction Subscrier
        self.direction_subscriber = self.create_subscription(
            String,
            'tracking3',
            self.track_control,
            qos_profile)

################################################# CAR CONTROL #################################################       
        self.car_left = 0.0
        self.car_right = 0.0

################################################# Function START #################################################


    def track_control(self, msg):
        self.get_logger().info('Received data : {}'.format(msg.data))
        trackdata = msg.data
        print(trackdata)

        if trackdata == "STOP":
          self.car_left = 0.0
          self.car_right = 0.0          


        #오른쪽으로 90도 회전
        elif trackdata == "TURNRIGHTN":
            self.turn_right_90()

        elif trackdata == "TURNRIGHTTH":
            self.turn_right_360()

        elif trackdata == "TURNLEFTN":
            self.turn_left_90()
                
        elif trackdata == "FRONT":
            self.normal_front()
        
        




        self.drive_command.data = [self.odrive_mode,self.car_right,self.car_left]
        self.control_publisher.publish(self.drive_command)

########################### CAR DIRECTION ###########################

    #NORMAL
    def normal_front(self) :
        self.get_logger().info("NORMAL GO FRONT")
        self.car_left = 5.0
        self.car_right = 5.0

    def normal_left(self) :
        self.get_logger().info("NORMAL GO LEFT")
        self.car_left = -5.0
        self.car_right = 5.0


    def normal_right(self) :
        self.get_logger().info("NORMAL GO RIGHT")
        self.car_left = 5.0
        self.car_right = -5.0


 
#---------------- 회전 function ------------------
    def turn_right_90(self):
        self.get_logger().info("TURN RIGHT 90°")

        # 회전 시작
        self.car_left = 5.0
        self.car_right = -5.0
        time.sleep(3)
        self.car_left = 0.0
        self.car_right = 0.0

    def turn_right_360(self):
        self.get_logger().info("TURN RIGHT 360°")
        self.car_left = 5.0
        self.car_right = -5.0

        time.sleep(20)

        self.car_left = 0.0
        self.car_right = 0.0
   
    def turn_left_90(self):
        self.get_logger().info("TURN LEFT 90°")
        self.car_left = -5.0
        self.car_right = 5.0
        time.sleep(2)
        self.car_left = 0.0
        self.car_right = 0.0
   

def main(args=None):
    rclpy.init(args=args)
    node = tracking()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()      
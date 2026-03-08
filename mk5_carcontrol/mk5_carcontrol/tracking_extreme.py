

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


        #Camera IMU Data Sampling
        self.imu_subscriber = self.create_subscription(Float32MultiArray, 'imu_processing', self.imu_msg_sampling, qos_profile)
        self.robot_state = 0

 
        #ODrive Command
        self.car_control_publisher = self.create_publisher(Float32MultiArray, 'Odrive_car_control', qos_profile)
        self.flip_control_publisher = self.create_publisher(Float32MultiArray, 'Odrive_flip_control', qos_profile)
        self.drive_command = Float32MultiArray()


        #motor encoder feedback
        #self.encoder_subscriber = self.create_subscription(Float32MultiArray, 'Odrive_encoder', self.encoder_clear, qos_profile)


################################################# ODRIVE MODE#################################################
        self.odrive_mode = 1.0

################################################# DIRECTION #################################################
        #Direction Subscrier
        self.direction_subscriber = self.create_subscription(
            String,
            'tracking',
            self.track_control,
            qos_profile)
        
        #박스 잡히는 영역에 따라 천천히 움직이고, 정지하고, 재개하는 플래그 설정.
        self.boxslow_sub = self.create_subscription(
            String,
            'boxmove',
            self.box_slow,
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


        # Robot STATE : NORMAL 

        elif self.robot_state == 0 :
            if trackdata == "FRONT":
                 self.normal_front()

            elif trackdata == "LEFT":
                self.normal_left()

            elif trackdata == "RIGHT":
                self.normal_right()


        # Robot STATE 1 : 오른쪽으로 살짝 기운 상태

        elif self.robot_state == 1 :

            if trackdata == "FRONT":
                self.state1_front()


            elif trackdata == "LEFT":
                self.state1_left()


            elif trackdata == "RIGHT":
                self.state1_right()

        # Robot STATE 2 : 오른쪽으로 많이 기운 상태
        elif self.robot_state == 2 :

            if trackdata == "FRONT":
                self.state2_front()

            elif trackdata == "LEFT":
                self.state2_left()


            elif trackdata == "RIGHT":
                self.state2_right()


        # Robot STATE 3 : Week Light : 왼쪽으로 살짝 기운 상태
        elif self.robot_state == -1 :

            if trackdata == "FRONT":
                self.state3_front()


            elif trackdata == "LEFT":
                self.state3_left()


            elif trackdata == "RIGHT":
                self.state3_right()

        # Robot STATE 4 : STRONG  Leftt : 왼쪽으로 많이 기운 상태
        elif self.robot_state == -2 :

            if trackdata == "FRONT":
                self.state4_front()

            elif trackdata == "LEFT":
                self.state4_left()


            elif trackdata == "RIGHT":
                self.state4_right()

        self.drive_command.data = [self.odrive_mode, self.car_left,self.car_right]
        self.car_control_publisher.publish(self.drive_command)




########################### CAR DIRECTION ###########################

    #NORMAL
    def normal_front(self) :
        self.get_logger().info("NORMAL GO FRONT")
        self.car_left = 2.0
        self.car_right = 2.4

    def normal_left(self) :
        self.get_logger().info("NORMAL GO LEFT")
        self.car_left = -1.0
        self.car_right = 2.0


    def normal_right(self) :
        self.get_logger().info("NORMAL GO RIGHT")
        self.car_left = 3.0
        self.car_right = -2.0

    #STATE 1 : Robot => Weeak RIGHT => GO LEFT WEAKLY

    def state1_front(self) :

        self.get_logger().info("STATE 1- GO FRONT")
        self.car_left = 2.0
        self.car_right = 2.7
        time.sleep(1)

        self.car_left = 2.0
        self.car_right = 2.0
        time.sleep(1)

    def state1_left(self) :
        self.get_logger().info("STATE 1- GO LEFT")
        self.car_left = -1.0
        self.car_right = 2.0


    def state1_right(self) :
        self.get_logger().info("STATE 1- GO RIGHT")
        self.car_left = 1.3
        self.car_right = -1.0

    #STATE 2 : Robot => STRONG RIGHT => GO LEFT STRONG

    def state2_front(self) :

        self.get_logger().info("STATE 2 - GO FRONT")
        self.car_left = 1.2 #이전 1.0,3.0
        self.car_right = 3.35
        time.sleep(1)

        self.car_left = 2.0
        self.car_right = 2.0
        time.sleep(1)

    def state2_left(self) :
        self.get_logger().info("STATE 2- GO LEFT")
        self.car_left = -2.0
        self.car_right = 3.0

    def state2_right(self) :
        self.get_logger().info("STATE 2- GO RIGHT")
        self.car_left = 2.0
        self.car_right = -3.0
        time.sleep(0.5)

        self.car_left = 2.7
        self.car_right = 2.0
        time.sleep(0.5)        


    #STATE 3 : Robot => weak LEFT => GO RIGHT Weakly

    def state3_front(self) :
        self.get_logger().info("STATE 3 - GO FRONT")

        self.car_left = 2.3
        self.car_right = 2.0
        time.sleep(1)

        self.car_left = 2.0
        self.car_right = 2.0
        time.sleep(1)       


    def state3_left(self) :
        self.get_logger().info("STATE 3- GO LEFT")
        self.car_left = -1.0
        self.car_right = 1.3


    def state3_right(self) :
        self.get_logger().info("STATE 3- GO RIGHT")
        self.car_left = 2.0
        self.car_right = -1.0


    #STATE 4 : Robot => STRONG LEFT => GO RIGHT STRONG

    def state4_front(self) :

        self.get_logger().info("STATE 4 - GO FRONT")
        self.car_left = 2.5
        self.car_right = 2.0
        time.sleep(1)

        self.car_left = 2.1
        self.car_right = 2.0
        time.sleep(1)

    def state4_left(self) :
        self.get_logger().info("STATE 4- GO LEFT")
        self.car_left = -2.0
        self.car_right = 3.0
        time.sleep(0.5)     
        self.car_left = 3.5
        self.car_right = 2.0
        time.sleep(0.5)   

    def state4_right(self) :
        self.get_logger().info("STATE 4- GO RIGHT")
        self.car_left = 3.0
        self.car_right = -2.0
        time.sleep(0.5)  
        self.car_left = 2.5
        self.car_right = 2.0
        time.sleep(0.5)   

 

##############################
######## IMU SAMPLING ########
##############################
    def imu_msg_sampling(self, msg):
        robot_state = msg.data


        if robot_state == 0.0:
            self.get_logger().info("NORMAL")
            self.robot_state = 0


        if robot_state == -3.0:
            self.get_logger().info("뒤로 살짝 기운 상태")
            self.robot_state = -3

        elif robot_state == 3.0:
            self.get_logger().info("앞으로 살짝 기운 상태")
            self.robot_state = 3

        elif robot_state == 1.0:
            self.get_logger().info("오른쪽으로 살짝 기운 상태")
            self.robot_state = 1

        elif robot_state == 2.0:
            self.get_logger().info("오른쪽으로 많이 기운 상태")
            self.robot_state = 2

        elif robot_state == -1.0:
            self.get_logger().info("왼쪽으로 살짝 기운 상태")
            self.robot_state = -1

        elif robot_state == -2.0:
            self.get_logger().info("왼쪽으로 많이 기운 상태")
            self.robot_state = -2


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
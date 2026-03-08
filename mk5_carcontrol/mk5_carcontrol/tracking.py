

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


################################################# IMU #################################################

        #Camera IMU Data 
        self.imu_subscriber = self.create_subscription(Float32MultiArray, 'imu_data', self.imu_msg_sampling, qos_profile)

 
        # IMU Data Prcossing
        self.control_publisher = self.create_publisher(Float32MultiArray, 'imu_processing', qos_profile)
        self.drive_command = Float32MultiArray()


        #IMU Threshold
        self.normal_max = 107
        self.normal_min = 77

        self.right_weak_max = 124
        self.left_weak_max = 65


        #IMU - DIRECTION FLAG
        self.robot_roll = 0  #IMU Pitch 
        self.robot_pitch = 0 # IMU Pitch 


################################################# ODRIVE MODE#################################################
        self.odrive_mode = 1.0

################################################# DIRECTION #################################################
        #Direction Subscrier
        self.direction_subscriber = self.create_subscription(
            String,
            'tracking',
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
        elif trackdata == "TURNTORIGHT90":
            self.turn_to_right_90()

        elif trackdata == "TURNTORIGHT360":
            self.turn_to_right_360()

        # Robot STATE : NORMAL => Color based tracking

        elif trackdata != "STOP" and self.robot_roll == 0 :
            if trackdata == "FRONT":
                 self.normal_front()

            elif trackdata == "LEFT":
                self.normal_left()

            elif trackdata == "RIGHT":
                self.normal_right()


        # Robot STATE 1 : Week Right => you have to go left weakly !!!

        elif trackdata != "STOP" and self.robot_roll == 1 :

            if trackdata == "FRONT":
                self.state1_front()


            elif trackdata == "LEFT":
                self.state1_left()


            elif trackdata == "RIGHT":
                self.state1_right()

        # Robot STATE 2 : STRONG  Right => you have to go left Strong!!!
        elif trackdata != "STOP" and self.robot_roll == 2 :

            if trackdata == "FRONT":
                self.state2_front()

            elif trackdata == "LEFT":
                self.state2_left()


            elif trackdata == "RIGHT":
                self.state2_right()


        # Robot STATE 3 : Week Light => you have to go right weakly !!!

        elif trackdata != "STOP" and self.robot_roll == -1 :

            if trackdata == "FRONT":
                self.state3_front()


            elif trackdata == "LEFT":
                self.state3_left()


            elif trackdata == "RIGHT":
                self.state3_right()

        # Robot STATE 4 : STRONG  Leftt => you have to go right Stronlg!!!
        elif trackdata != "STOP" and self.robot_roll == -2 :

            if trackdata == "FRONT":
                self.state4_front()

            elif trackdata == "LEFT":
                self.state4_left()


            elif trackdata == "RIGHT":
                self.state4_right()

        self.drive_command.data = [self.odrive_mode, self.car_left,self.car_right]
        self.control_publisher.publish(self.drive_command)

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

 
#---------------- 회전 function ------------------
    def turn_to_right_90(self):
        self.get_logger().info("TURN RIGHT 90°")

        # 회전 시작
        self.car_left = 2.0
        self.car_right = -2.0
        self.control_publisher.publish(Float32MultiArray(data=[self.odrive_mode, self.car_left, self.car_right]))

        # 목표 객체 인식할 때 까지 확인
        # while abs(self.robot_yaw - target_yaw) > 2:  # 오차 ±2도 허용
        #     rclpy.spin_once(self, timeout_sec=0.05)

        time.sleep(2)

        # 정지
        self.car_left = 0.0
        self.car_right = 0.0
        self.control_publisher.publish(Float32MultiArray(data=[self.odrive_mode, self.car_left, self.car_right]))

    def turn_to_right_360(self):
        self.get_logger().info("TURN RIGHT 360°")

        # 회전 시작
        self.car_left = 2.0
        self.car_right = -2.0
        self.control_publisher.publish(Float32MultiArray(data=[self.odrive_mode, self.car_left, self.car_right]))

        # 목표 객체 인식할 때 까지 확인
        # while abs(self.robot_yaw - target_yaw) > 2:  # 오차 ±2도 허용
        #     rclpy.spin_once(self, timeout_sec=0.05)
        time.sleep(5)

        # 정지
        self.car_left = 0.0
        self.car_right = 0.0
        self.control_publisher.publish(Float32MultiArray(data=[self.odrive_mode, self.car_left, self.car_right]))



##############################
######## IMU SAMPLING ########
##############################
    def imu_msg_sampling(self, msg):

        roll_data = msg.data[0]
        pitch_data = msg.data[1]


        #WEEK RIGHT STATE => GO TO WEAK LEFT 

        if self.normal_max <= roll_data <= self.right_weak_max:
            self.robot_roll = 1
    

        #STRONG RIGHT STATE => GO TO WEAK LEFT 

        elif roll_data > self.right_weak_max:
            self.robot_roll = 2


        #WEEK LEFT STATE => GO TO WEAK RIGHT 

        elif self.left_weak_max <= roll_data < self.normal_min:
            self.robot_roll = -1


        #STRONG LEFT STATE => GO TO WEAK LEFT      
          
        elif roll_data < self.left_weak_max:
            self.robot_roll = -2
    

        #NORMAL

        else:
            if self.normal_min<= roll_data <=self.normal_max and 20<= pitch_data<=30:
                self.robot_roll = 0



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
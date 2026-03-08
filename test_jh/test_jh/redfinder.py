import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker

class DaePublisher(Node):
    def __init__(self):
        super().__init__('dae_publisher')
        self.publisher = self.create_publisher(Marker, 'visualization_marker', 10)
        timer_period = 1.0
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        marker = Marker()
        marker.header.frame_id = "map"   # rviz2 좌표계 (TF와 맞추세요)
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "dae_mesh"
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # DAE 파일 경로
        marker.mesh_resource = "file:///home/ljh/Downloads/apt.dae"
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.a = 1.0
        marker.color.r = 0.8
        marker.color.g = 0.8
        marker.color.b = 0.8

        marker.mesh_use_embedded_materials = True #이게 맵에 색을 입히는 코드

        self.publisher.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = DaePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
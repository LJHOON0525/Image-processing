import rclpy
from rclpy.node import Node
from roboflow import Roboflow
import os

class RoboflowDownloader(Node):
    def __init__(self):
        super().__init__('roboflow_downloader')
        self.get_logger().info("Roboflow 데이터셋 다운로드 노드 시작됨")

        # Roboflow API 키 입력 (너의 API 키로 변경 필요)
        API_KEY = "CpMTVxLHgwQTzhNOUN5U"
        rf = Roboflow(api_key=API_KEY)

        # 프로젝트 & 버전 설정 (Roboflow에서 확인한 값으로 변경 필요)
        workspace = "firsttest-appcq"
        project_name = "my-first-project-fi4eg"
        version_number = 1

        try:
            project = rf.workspace(workspace).project(project_name)
            version = project.version(version_number)
            
            # YOLOv8 형식으로 다운로드
            dataset_path = version.download("yolov8")

            self.get_logger().info(f"✅ 데이터셋 다운로드 완료: {dataset_path}")

        except Exception as e:
            self.get_logger().error(f"❌ 데이터셋 다운로드 실패: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = RoboflowDownloader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

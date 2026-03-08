from roboflow import Roboflow

rf = Roboflow(api_key="CpMTVxLHgwQTzhNOUN5U")
project = rf.workspace("firsttest-appcq").project("my-first-project-vrpvt")
version = project.version(2)  # 너가 방금 만든 2번째 버전
dataset = version.download("yolov8")  # YOLOv8 형식으로 다운로드

# YOLOv8 학습
from ultralytics import YOLO

model = YOLO("yolov8n.pt") 

model.train(
    data=dataset.location + "/data.yaml",  # 자동 다운로드된 data.yaml 경로 사용
    epochs=50,
    imgsz=640,
    batch=16,
    name="train_goodbox",
    project="runs/detect",
    workers=8
)

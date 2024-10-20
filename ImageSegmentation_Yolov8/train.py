from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  # load a pretrained model

model.train(data='config.yaml', epochs=100, imgsz=640)

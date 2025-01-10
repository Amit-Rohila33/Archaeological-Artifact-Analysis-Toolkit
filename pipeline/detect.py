from models.yolov8_model import YOLOv8Model

def detect_artifacts(image_path):
    yolo_model = YOLOv8Model()
    detected_objects = yolo_model.detect(image_path)
    return detected_objects

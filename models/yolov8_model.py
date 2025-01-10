import torch

class YOLOv8Model:
    def __init__(self, model_path='yolov8.pt'):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov8', pretrained=True)
    
    def detect(self, image):
        results = self.model(image)
        return results.xywh[0]  # Get detections in (x, y, width, height) format

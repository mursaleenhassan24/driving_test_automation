import cv2
import math
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

class VisionEyeDistanceCalculator:
    def __init__(self, model_path, pixel_per_meter=10):
        self.model = YOLO(model_path)
        self.pixel_per_meter = pixel_per_meter
        self.txt_color, self.txt_background, self.bbox_clr = ((0, 0, 0), (255, 255, 255), (255, 0, 255))

    def calculate_distance(self, im0, polygons):
        
        self.h, self.w = im0.shape[:2]
        self.center_point = (0, self.h)
        annotator = Annotator(im0, line_width=2)
        results = self.model.track(im0, persist=True)
        boxes = results[0].boxes.xyxy.cpu()

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, class_ids):
                if cls == 2 or cls == 7:
                    
                    nearest_point = None
                    min_distance = float('inf')

                    for polygon in polygons:
                        for point in polygon:
                            distance = math.sqrt((point[0] - self.center_point[0]) ** 2 + (point[1] - self.center_point[1]) ** 2)
                            if distance < min_distance:
                                min_distance = distance
                                nearest_point = point

                    self.center_point = nearest_point

                    annotator.box_label(box, label=str(track_id), color=self.bbox_clr)
                    annotator.visioneye(box, self.center_point)

                    x1, y1 = int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)    # Bounding box centroid

                    distance = (math.sqrt((x1 - self.center_point[0]) ** 2 + (y1 - self.center_point[1]) ** 2))/self.pixel_per_meter

                    text_size, _ = cv2.getTextSize(f"Distance: {distance:.2f} m", cv2.FONT_HERSHEY_SIMPLEX,1.2, 3)
                    cv2.rectangle(im0, (x1, y1 - text_size[1] - 10),(x1 + text_size[0] + 10, y1), self.txt_background, -1)
                    cv2.putText(im0, f"Distance: {distance:.2f} m",(x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2,self.txt_color, 3)
        return im0
        

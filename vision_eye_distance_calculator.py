import cv2
import math
from constants import *
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

class VisionEyeDistanceCalculator:
    def __init__(self, model_path, pixel_per_meter=10):
        self.model = YOLO(model_path)
        self.pixel_per_meter = pixel_per_meter

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
                    x1, y1, x2, y2 = map(int, box)
                    bounding_box_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

                    nearest_point = None
                    min_distance = float('inf')

                    for polygon in polygons:
                        for point in polygon:
                            if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
                                distance = math.sqrt((point[0] - self.center_point[0]) ** 2 + (point[1] - self.center_point[1]) ** 2)
                                if distance < min_distance:
                                    min_distance = distance
                                    nearest_point = point

                    if nearest_point is not None:
                        self.center_point = nearest_point

                    annotator.box_label(box, label=str(track_id), color=BOUNDING_BOX_COLOR)
                    annotator.visioneye(box, self.center_point)

                    box_centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                    distance = math.sqrt((box_centroid[0] - self.center_point[0]) ** 2 + (box_centroid[1] - self.center_point[1]) ** 2) / self.pixel_per_meter
                    
        return im0, distance

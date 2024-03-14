import os
import cv2
import math
import numpy as np
from constants import *
from shapely.geometry import Polygon, Point
from ultralytics import YOLO, YOLOWorld
from ultralytics.utils.plotting import Annotator

class VisionEyeDistanceCalculator:
    def __init__(self, model_path, pixel_per_meter=10):
        if not os.path.exists('models'):
            os.makedirs('models')

        self.model = YOLOWorld(model_path)
        self.pixel_per_meter = pixel_per_meter

    def calculate_distance(self, im0, polygons):
        self.h, self.w = im0.shape[:2]
        self.center_point = (0, self.h)
        annotator = Annotator(im0, line_width=2)
        results = self.model.track(im0, persist=True)
        boxes = results[0].boxes.xyxy.cpu()

        distance = 0
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, class_ids):
                if cls == 2 or cls == 7:
                    x1, y1, x2, y2 = map(int, box)
                    bounding_box_coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    box_centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

                    nearest_point = None

                    nearest_voting = {}
                    for polygon in polygons:
                        polygon = Polygon(polygon)
                        bounding_box_centroid = Point(np.mean([coord[0] for coord in bounding_box_coords]), np.mean([coord[1] for coord in bounding_box_coords]))
                        nearest_point = polygon.exterior.interpolate(polygon.exterior.project(bounding_box_centroid))

                        if nearest_point is not None:
                            self.center_point = (int(nearest_point.x), int(nearest_point.y))
                            distance = math.sqrt((box_centroid[0] - self.center_point[0]) ** 2 + (box_centroid[1] - self.center_point[1]) ** 2) / self.pixel_per_meter
                            nearest_voting[distance] = self.center_point 

                    if nearest_voting:
                        self.center_point = nearest_voting[min(nearest_voting.keys())]
                        distance = min(nearest_voting.keys())
                    
                    annotator.box_label(box, label=str(track_id), color=BOUNDING_BOX_COLOR)
                    annotator.visioneye(box, self.center_point)

        return im0, distance

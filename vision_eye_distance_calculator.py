import os
import cv2
import math
import numpy as np

from shapely.geometry import Polygon, Point
from ultralytics import YOLO, YOLOWorld
from ultralytics.utils.plotting import Annotator

from helper_functions.helper_functions import *
from constants import *

class VisionEyeDistanceCalculator:
    def __init__(self, model_path):
        if not os.path.exists('models'):
            os.makedirs('models')

        self.load_model(model_path)

    def load_model(self, model_path):
        if model_path.endswith('world.pt'):
            self.model = YOLOWorld(model_path)
        else:
            self.model = YOLOWorld(model_path)

        self.classes = list(self.model.names.values())

    def calculate_distance(self, im0, polygons):
        self.h, self.w = im0.shape[:2]
        self.center_point = (0, self.h)
        annotator = Annotator(im0, line_width=2)
        results = self.model.track(im0, persist=True)
        boxes = results[0].boxes.xyxy.cpu()

        distance = 0
        pixel_per_meter = 0
        box_centroid = (0, 0)
        
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, class_ids):
                if self.classes[cls] in VEHICLE_CLASSES:
                    x1, y1, x2, y2 = map(int, box)
                    vehicle_height_pixels, vehicle_width_pixels = y2 - y1, x2 - x1
                    pixel_per_meter = calculate_pixel_per_meter(vehicle_width_pixels, vehicle_height_pixels)
                    
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
                            distance = math.sqrt((box_centroid[0] - self.center_point[0]) ** 2 + (box_centroid[1] - self.center_point[1]) ** 2) / pixel_per_meter
                            nearest_voting[distance] = self.center_point 

                    if nearest_voting:
                        self.center_point = nearest_voting[min(nearest_voting.keys())]
                        distance = min(nearest_voting.keys())
                    
                    annotator.box_label(box, label=str(track_id), color=BOUNDING_BOX_COLOR)
                    annotator.visioneye(box, self.center_point)

        track_id = 1 if len(track_ids) == 0 else track_ids[0]
        
        return im0, distance, pixel_per_meter, track_id, box_centroid

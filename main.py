import os
import cv2
import numpy as np

from constants import *
from database.database import Database
from helper_functions.polygon_drawer import PolygonDrawer
from vision_eye_distance_calculator import VisionEyeDistanceCalculator

class Main:
    def __init__(self):
        self.database = Database()
        self.vision_eye_distance_calculator = VisionEyeDistanceCalculator(model_path=path_model_yolov8m)
        
        if path_output:
            if not os.path.exists(path_output):
                os.makedirs(path_output)

        self.cap = cv2.VideoCapture(path_stream)
        self.w, self.h, self.fps = (int(self.cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        self.polygons = []
                
    def interpolate_points(self, array):
        interpolated_points = []
        for i in range(len(array) - 1):
            start_point = array[i]
            end_point = array[i + 1]
            
            distance = np.linalg.norm(end_point - start_point)
            num_steps = int(distance) + 1
            
            for j in range(num_steps):
                interpolated_point = (start_point * (num_steps - j - 1) + end_point * (j + 1)) / num_steps
                interpolated_points.append(interpolated_point.astype(int))

        interpolated_points.append(array[-1])
        return np.array(interpolated_points)

    def draw_polygons(self, canvas):
        for points in self.polygons:
            if len(points) > 0:
                cv2.fillPoly(canvas, [points], FINAL_LINE_COLOR)

    def put_text(self, im0, distance, pixels_per_meter):
        text = f"Distance: {distance:.2f} m"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(im0, (self.w - (text_size[0] + 12), 10), (self.w - 2, text_size[1] + 20), TEXT_BACKGROUND, -1)
        cv2.putText(im0, text, (self.w - (text_size[0] + 10), 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, TEXT_COLOR, 3)

        cv2.rectangle(im0, (self.w - (text_size[0] + 12), 40), (self.w - 2, text_size[1] + 20), TEXT_BACKGROUND, -1)
        cv2.putText(im0, f"Pixels: {pixels_per_meter}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, TEXT_COLOR, 3)

    def run(self):
        if path_output:
            self.out = cv2.VideoWriter(path_output, cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (self.w, self.h))

        while True:
            ret, im0 = self.cap.read()
            if not ret:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            im0, distance, pixels_per_meter, track_id, box_centroid  = self.vision_eye_distance_calculator.calculate_distance(im0, self.polygons)
            self.database.insert(track_id, f'{box_centroid}', distance, 0)

            if self.polygons == []:
                self.polygons = PolygonDrawer("visioneye-distance-calculation", image=im0).run()
                self.polygons = [self.interpolate_points(poly) for poly in self.polygons]

            self.draw_polygons(im0)

            self.put_text(im0, distance, pixels_per_meter)

            cv2.imshow("visioneye-distance-calculation", im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if path_output:
                self.out.write(im0)

        if path_output:
            self.out.release()

        self.cap.release()
        cv2.destroyAllWindows()



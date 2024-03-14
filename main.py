import cv2
import numpy as np
from constants import *
from helper_functions.polygon_drawer import PolygonDrawer
from vision_eye_distance_calculator import VisionEyeDistanceCalculator

class Main:
    def __init__(self):
        self.vision_eye_distance_calculator = VisionEyeDistanceCalculator(model_path=path_model_yolov8m)
        
        self.cap = cv2.VideoCapture(path_stream)
        self.w, self.h, self.fps = (int(self.cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        self.polygons = [np.array([[569, 289],
                        [825, 271],
                        [951, 594],
                        [493, 641],
                        [568, 289]]), np.array([[606, 109],
                        [592, 180],
                        [779, 170],
                        [749, 100]]), np.array([[ 935,   90],
                        [1013,  145],
                        [1101,  222],
                        [1246,  356],
                        [1275,  392],
                        [1275,  231],
                        [1066,   94],
                        [ 939,   86]]), np.array([[417, 122],
                        [301, 252],
                        [173, 395],
                        [ 10, 619],
                        [ 14, 316],
                        [260, 137],
                        [418, 120]]), np.array([[ 300,  103],
                        [ 567,   69],
                        [ 701,   54],
                        [ 906,   50],
                        [1166,   55],
                        [1122,    2],
                        [ 684,    3],
                        [ 389,   28],
                        [ 309,   59],
                        [ 300,   99]])]
        self.polygons = [self.interpolate_points(poly) for poly in self.polygons]
        
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

    def put_text(self, im0, distance):
        text = f"Distance: {distance:.2f} m"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(im0, (self.w - (text_size[0] + 12), 10), (self.w - 2, text_size[1] + 20), TEXT_BACKGROUND, -1)
        cv2.putText(im0, text, (self.w - (text_size[0] + 10), 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, TEXT_COLOR, 3)

    def run(self):
        if path_output:
            self.out = cv2.VideoWriter(path_output, cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (self.w, self.h))

        while True:
            ret, im0 = self.cap.read()
            if not ret:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            im0, distance = self.vision_eye_distance_calculator.calculate_distance(im0, self.polygons)

            if self.polygons == []:
                self.polygons = PolygonDrawer("visioneye-distance-calculation", image=im0).run()
            self.draw_polygons(im0)

            self.put_text(im0, distance)

            cv2.imshow("visioneye-distance-calculation", im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if path_output:
                self.out.write(im0)

        if path_output:
            self.out.release()

        self.cap.release()
        cv2.destroyAllWindows()


main = Main()
main.run()
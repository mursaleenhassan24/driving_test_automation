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

    def draw_polygons(self, canvas):
        for points in self.polygons:
            if len(points) > 0:
                cv2.fillPoly(canvas, [points], FINAL_LINE_COLOR)

    def run(self):
        if path_output:
            self.out = cv2.VideoWriter(path_output, cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (self.w, self.h))

        while True:
            ret, im0 = self.cap.read()
            if not ret:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            if self.polygons == []:
                self.polygons = PolygonDrawer("visioneye-distance-calculation", image=im0).run()
                print(self.polygons)
            self.draw_polygons(im0)

            im0 = self.vision_eye_distance_calculator.calculate_distance(im0, self.polygons)
            
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
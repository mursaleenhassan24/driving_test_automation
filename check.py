import cv2
import numpy as np
from constants import *
from vision_eye_distance_calculator import VisionEyeDistanceCalculator

points = []  # List to store points of the current polygon
polygons = []  # List to store all polygons

def draw_polygon(event, x, y, flags, param):
    global points
    global polygons
    img = param  # Access the image from the parameter
    
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        
        # Draw lines connecting points
        if len(points) > 1:
            cv2.line(img, points[-2], points[-1], (0, 255, 0), 2)
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points) > 2:
            polygons.append(np.array(points))
            points = []  # Clear points for the next polygon
            cv2.polylines(img, polygons, isClosed=True, color=(255, 0, 0), thickness=2)

class Main:
    def __init__(self):
        self.vision_eye_distance_calculator = VisionEyeDistanceCalculator(model_path=path_model_yolov8m)
        self.cap = cv2.VideoCapture(path_stream)
        self.w, self.h, self.fps = (int(self.cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    def run(self):
        if path_output:
            self.out = cv2.VideoWriter(path_output, cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (self.w, self.h))

        while True:
            ret, im0 = self.cap.read()
            if not ret:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            
            cv2.namedWindow('DrivingTestAutomation')
            cv2.setMouseCallback('DrivingTestAutomation', draw_polygon, im0)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):
                points.clear()
                polygons.clear()
                polygon_drawn = False
            elif key == 27:  # Press Esc to exit
                print(polygons)

            im0 = self.vision_eye_distance_calculator.calculate_distance(im0)

            cv2.imshow("DrivingTestAutomation", im0)
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

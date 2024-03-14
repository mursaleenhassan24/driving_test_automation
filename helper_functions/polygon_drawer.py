import cv2
import numpy as np
from constants import *

class PolygonDrawer:
    def __init__(self, window_name, image):
        self.window_name = window_name 
        self.done = False 
        self.current = (0, 0) 
        self.points = [] 
        self.polygons = []
        self.image = image

    def on_mouse(self, event, x, y, buttons, user_param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.polygons.append(np.array(self.points))
            self.done = True

    def draw_polygons(self, canvas):
        for points in self.polygons:
            if len(points) > 0:
                cv2.fillPoly(canvas, [points], FINAL_LINE_COLOR)

    def draw_working_lines(self, canvas):
        if len(self.points) > 0:
            cv2.polylines(canvas, [np.array(self.points)], False, FINAL_LINE_COLOR, 3)
            cv2.line(canvas, self.points[-1], self.current, WORKING_LINE_COLOR, 3)

    def run(self):
        while True:
            canvas = self.image.copy()
            self.draw_polygons(canvas)
            self.draw_working_lines(canvas)
            cv2.imshow(self.window_name, canvas)
            cv2.setMouseCallback(self.window_name, self.on_mouse)

            key = cv2.waitKey(50) & 0xFF

            if key == ord('n'):
                self.done = False 
                self.current = (0, 0) 
                self.points = []
                continue     
            elif key == 27:
                break

        cv2.destroyWindow(self.window_name)
        return self.polygons



import numpy as np
import cv2

FINAL_LINE_COLOR = (0, 0, 255)
WORKING_LINE_COLOR = (0, 0, 255)


class PolygonDrawer(object):
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
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            print("Completing polygon with %d points." % len(self.points))
            self.polygons.append(np.array(self.points))
            self.done = True

    def run(self):
        while True:
            if len(self.polygons) > 0:
                for points in self.polygons:
                    if (len(points) > 0):
                        cv2.fillPoly(self.image, np.array([points]), FINAL_LINE_COLOR)
            cv2.namedWindow(self.window_name)
            cv2.imshow(self.window_name, self.image)
            cv2.waitKey(1)
            cv2.setMouseCallback(self.window_name, self.on_mouse)

            while(not self.done):
                canvas = self.image.copy()
                if (len(self.points) > 0):
                    cv2.polylines(canvas, np.array([self.points]), False, FINAL_LINE_COLOR, 3)
                    cv2.line(canvas, self.points[-1], self.current, WORKING_LINE_COLOR, 3)
           
                cv2.imshow(self.window_name, canvas)

                canvas = self.image.copy()
                if (len(self.points) > 0):
                    cv2.fillPoly(canvas, np.array([self.points]), FINAL_LINE_COLOR)

                key = cv2.waitKey(50) & 0xFF
            
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(50) & 0xFF
            
            if key==ord('n'):
                self.done = False 
                self.current = (0, 0) 
                self.points = []
                continue     
            elif key == 27:
                print("All Polygons = %s" % self.polygons)
                print("Closing without finishing.")
                break

        cv2.destroyWindow(self.window_name)
        return canvas


if __name__ == "__main__":
    image = cv2.imread('image.jpg')
    pd = PolygonDrawer("Polygon", image)
    image = pd.run()
    cv2.imwrite("polygon.png", image)
    print("Polygon = %s" % pd.points)
import cv2
import numpy as np

# Global variables
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


# Create a black image
height, width = 500, 500
img = np.zeros((height, width, 3), dtype=np.uint8)

# Create a window and set mouse callback
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw_polygon, img)

while True:
    cv2.imshow('Image', img)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('n'):
        points = []
        polygons = []  # Clear existing polygons
        img = np.zeros((height, width, 3), dtype=np.uint8)  # Clear image
    elif key == 27:  # Press Esc to exit
        print(polygons)
        break

cv2.destroyAllWindows()

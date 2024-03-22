# from main import Main   

# main = Main()
# main.run()


from ultralytics import YOLO
from helper_functions import speed_estimation
import cv2

model = YOLO("models/yolov8l-world.pt")
names = model.model.names

cap = cv2.VideoCapture("videos/test (1).mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


# line_pts = [[(20, 200), (1260, 200)], [(20, 400), (1260, 400)], [(20, 600), (1260, 600)]]

# Init speed-estimation obj
speed_obj = speed_estimation.SpeedEstimator()
speed_obj.set_args(reg_pts=None,
                   names=names,
                   view_img=True)

count = 1
while cap.isOpened():

    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.track(im0, persist=True, show=False)

    im0 = speed_obj.estimate_speed(im0, tracks)

cap.release()
cv2.destroyAllWindows()
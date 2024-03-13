import cv2
from constants import *
from vision_eye_distance_calculator import VisionEyeDistanceCalculator

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
            cv2.imwrite('image.jpg', im0)
            break

            im0 = self.vision_eye_distance_calculator.calculate_distance(im0)

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
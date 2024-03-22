# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict
from time import time

import cv2
import numpy as np

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors
# from ultralytics.solutions import speed_estimation


class SpeedEstimator:
    """A class to estimation speed of objects in real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the speed-estimator class with default values for Visual, Image, track and speed parameters."""

        # Visual & im0 information
        self.im0 = None
        self.annotator = None
        self.view_img = False

        # Region information
        self.reg_pts = None
        self.region_thickness = 3
        self.reg_pts_gap_size = 100

        # Predict/track information
        self.clss = None
        self.names = None
        self.boxes = None
        self.trk_ids = None
        self.trk_pts = None
        self.line_thickness = 1
        self.trk_history = defaultdict(list)

        # Speed estimator information
        self.current_time = 0
        self.dist_data = {}
        self.trk_idsdict = {}
        self.spdl_dist_thresh = 10
        self.trk_previous_times = {}
        self.trk_previous_points = {}

        # Check if environment support imshow
        self.env_check = check_imshow(warn=True)

    def set_args(
        self,
        reg_pts,
        names,
        view_img=False,
        line_thickness=1,
        region_thickness=1,
        spdl_dist_thresh=10,
        reg_pts_gap_size=20
    ):
        """
        Configures the speed estimation and display parameters.

        Args:
            reg_pts (list): Initial list of points defining the speed calculation region.
            names (dict): object detection classes names
            view_img (bool): Flag indicating frame display
            line_thickness (int): Line thickness for bounding boxes.
            region_thickness (int): Speed estimation region thickness
            spdl_dist_thresh (int): Euclidean distance threshold for speed line
            reg_pts_gap_size (int): Size of the gap between lines.
        """
        if reg_pts is None:
            print("Region points not provided, using full frame")
        else:
            self.reg_pts = reg_pts
        self.names = names
        self.view_img = view_img
        self.line_thickness = line_thickness
        self.region_thickness = region_thickness
        self.spdl_dist_thresh = spdl_dist_thresh
        self.reg_pts_gap_size = reg_pts_gap_size

    def set_region(self):

        if self.reg_pts is None:
            height, width = self.im0.shape[:2]
            
            # Calculate the number of lines and spacing between them
            num_lines_horizontal = height // (2 * self.reg_pts_gap_size)  # Gap + Line
            num_lines_vertical = width // (2 * self.reg_pts_gap_size)  # Gap + Line
            spacing_horizontal = height // (num_lines_horizontal + 1)
            spacing_vertical = width // (num_lines_vertical + 1)
            
            # Initialize reg_pts
            self.reg_pts = []

            # Generate horizontal lines
            for i in range(1, num_lines_horizontal + 1):
                y = i * spacing_horizontal
                line_start = (0, y)
                line_end = (width, y)
                self.reg_pts.append((line_start, line_end))

            # Generate vertical lines
            for i in range(1, num_lines_vertical + 1):
                x = i * spacing_vertical
                line_start = (x, 0)
                line_end = (x, height)
                self.reg_pts.append((line_start, line_end))


    def extract_tracks(self, tracks):
        """
        Extracts results from the provided data.

        Args:
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.boxes = tracks[0].boxes.xyxy.cpu()
        self.clss = tracks[0].boxes.cls.cpu().tolist()
        self.trk_ids = tracks[0].boxes.id.int().cpu().tolist()

    def store_track_info(self, track_id, box):
        """
        Store track data.

        Args:
            track_id (int): object track id.
            box (list): object bounding box data
        """
        track = self.trk_history[track_id]
        bbox_center = (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))
        track.append(bbox_center)

        if len(track) > 30:
            track.pop(0)

        self.trk_pts = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        return track

    def plot_box_and_track(self, track_id, box, cls, track):
        """
        Plot track and bounding box.

        Args:
            track_id (int): object track id.
            box (list): object bounding box data
            cls (str): object class name
            track (list): tracking history for tracks path drawing
        """
        speed_label = f"{int(self.dist_data[track_id])}km/ph" if track_id in self.dist_data else self.names[int(cls)]
        bbox_color = colors(int(track_id)) if track_id in self.dist_data else (255, 0, 255)

        self.annotator.box_label(box, speed_label, bbox_color)

        cv2.polylines(self.im0, [self.trk_pts], isClosed=False, color=(0, 255, 0), thickness=1)
        cv2.circle(self.im0, (int(track[-1][0]), int(track[-1][1])), 5, bbox_color, -1)

    def calculate_speed(self, trk_id, track):
        """
        Calculation of object speed.

        Args:
            trk_id (int): object track id.
            track (list): tracking history for tracks path drawing
        """
        for reg_pts in self.reg_pts:
            
            if np.abs(reg_pts[1][0] - reg_pts[0][0]) > self.spdl_dist_thresh:
                if reg_pts[1][1] - self.spdl_dist_thresh < track[-1][1] < reg_pts[1][1] + self.spdl_dist_thresh:
                    direction = "backward" #Backward

                elif reg_pts[0][1] - self.spdl_dist_thresh < track[-1][1] < reg_pts[0][1] + self.spdl_dist_thresh:
                    direction = "forward" #Forward

                else:
                    direction = "unknown"
            else:
                if reg_pts[0][0] - self.spdl_dist_thresh < track[-1][1] < reg_pts[0][0] + self.spdl_dist_thresh:
                    direction = "right" #Right

                elif reg_pts[1][0] - self.spdl_dist_thresh < track[-1][1] < reg_pts[1][0] + self.spdl_dist_thresh:
                    direction = "left" #Left

                else:
                    direction = "unknown"
              

            if self.trk_previous_times[trk_id] != 0 and direction != "unknown" and trk_id not in self.trk_idsdict:
                self.trk_idsdict[trk_id] = (direction, tuple(int(round(x)) for x in track[-1]))

                time_difference = time() - self.trk_previous_times[trk_id]
                if time_difference > 0:
                    dist_difference = np.abs(track[-1][1] - self.trk_previous_points[trk_id][1])
                    speed = dist_difference / time_difference
                    self.dist_data[trk_id] = speed

        self.trk_previous_times[trk_id] = time()
        self.trk_previous_points[trk_id] = track[-1]
        
        if trk_id in self.trk_idsdict:
            if int(track[-1][0]) > (self.trk_idsdict[trk_id][1][0] + self.reg_pts_gap_size) - self.spdl_dist_thresh:
                self.trk_idsdict.pop(trk_id)
            elif int(track[-1][1]) > (self.trk_idsdict[trk_id][1][1] + self.reg_pts_gap_size) - self.spdl_dist_thresh:
                self.trk_idsdict.pop(trk_id)

            elif int(track[-1][0]) < (self.trk_idsdict[trk_id][1][0] - self.reg_pts_gap_size) + self.spdl_dist_thresh:
                self.trk_idsdict.pop(trk_id)
            elif int(track[-1][1]) < (self.trk_idsdict[trk_id][1][1] - self.reg_pts_gap_size) + self.spdl_dist_thresh:
                self.trk_idsdict.pop(trk_id)
            

    def estimate_speed(self, im0, tracks, region_color=(255, 0, 0)):
        """
        Calculate object based on tracking data.

        Args:
            im0 (nd array): Image
            tracks (list): List of tracks obtained from the object tracking process.
            region_color (tuple): Color to use when drawing regions.
        """
        self.im0 = im0
        if tracks[0].boxes.id is None:
            if self.view_img and self.env_check:
                self.display_frames()
            return im0
        self.extract_tracks(tracks)
        self.set_region()

        self.annotator = Annotator(self.im0, line_width=2)
        for reg in self.reg_pts:
            self.annotator.draw_region(reg_pts=reg, color=region_color, thickness=self.region_thickness)

        self.direction = {}
        for box, trk_id, cls in zip(self.boxes, self.trk_ids, self.clss):
            track = self.store_track_info(trk_id, box)

            if trk_id not in self.trk_previous_times:
                self.trk_previous_times[trk_id] = 0

            self.plot_box_and_track(trk_id, box, cls, track)
            self.calculate_speed(trk_id, track)

        if self.view_img and self.env_check:
            self.display_frames()

        return im0
    
    def get_direction(self):
        """Return the direction of the object."""
        return self.direction

    def display_frames(self):
        """Display frame."""
        cv2.imshow("Ultralytics Speed Estimation", self.im0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return


if __name__ == "__main__":
    SpeedEstimator()

from tracker import Tracker, Track
import numpy as np
import cv2
from random import random


class OpticalFlow(Tracker):
    """ Uses cv2.calcOpticalFlowFarneback to implement OpticalFlow tracking of YOLO-detected objects """

    def __init__(self):
        self.last_frame = None
        self.curr_frame = None
        self.id = -1
        self.tracks = []
        self.last_trackers = np.array([])

    def get_id(self):
        """ Auto increments id while accessing """
        self.id += 1
        return self.id

    def update(self, bboxes, scores, frame):
        """ Updates bboxed by comparison with last-detected bboxes and YOLO detected bboxes"""
        if self.last_frame is not None:
            self.curr_frame = self.convert_frame_to_gray(frame.copy())

            # get new bboxes
            flow = self.calculate_flow()
            new_bboxes = self.get_bbox_from_flow(flow)
            self.compare_bboxes(new_bboxes, bboxes)

            self.last_frame = self.curr_frame.copy()
        else:
            self.last_frame = self.convert_frame_to_gray(frame.copy())  # first frame

    def convert_frame_to_gray(self, frame):
        """ Converts frame from RGB to GRAY color """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def calculate_flow(self):
        """ Calculates optical flow based on two GRAY frames """
        return cv2.calcOpticalFlowFarneback(self.last_frame, self.curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    def get_bbox_from_flow(self, flow_map):
        """ Return xywh bbox from flow based on magnitude """
        magnitude, _ = cv2.cartToPolar(flow_map[..., 0], flow_map[..., 1])
        filtered_magnitude = magnitude > 2

        contours, _ = cv2.findContours(filtered_magnitude.astype(
            np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = np.array([cv2.boundingRect(contour) for contour in contours])

        # filtering smaller movements
        if bounding_boxes.ndim == 2:
            bounding_boxes = bounding_boxes[np.all(bounding_boxes > 6, axis=1)]

        return np.array(bounding_boxes)

    def compare_bboxes(self, bboxes_found, yolo_bboxes=None):
        """
            Compares bounding boxes found by tracker and YOLO with existing ones.
            First bbox can be initialized only by YOLO detector.
            When tracks are not empty, bboxes found are compared using IoU.
            Only first found match is added to tracks.
            All other matches found need to be really close to YOLO detection to be passed further.

        Args:
            bboxes_found (np.ndarray): xywh detections
            yolo_bboxes (list): xywh detections
        """
        # no tracks
        if self.tracks == [] and yolo_bboxes not in [None, []]:
            self.tracks = [Track(self.get_id(), box)
                           for box in self.xywh_to_xyxy(np.array(yolo_bboxes))]
        # objects found by optical flow
        elif self.tracks != [] and bboxes_found.ndim == 2:
            current_bboxes = [track.get_xywh() for track in self.tracks]
            new_tracks = []
            track_id_found = set()

            # assign new bboxes
            for bbox, track_bbox in zip(bboxes_found, self.xywh_to_xyxy(bboxes_found)):
                bbox_found_idx = self.bbox_over_IOU_threshold(
                    bbox, current_bboxes, threshold=.4, get_idx=True)
                if bbox_found_idx is not None and self.tracks[bbox_found_idx].track_id not in track_id_found:
                    # Update Track with new bbox
                    track_found = self.tracks[bbox_found_idx]
                    new_tracks.append(Track(track_found.track_id, track_bbox))
                    track_id_found.add(track_found.track_id)
                elif yolo_bboxes not in [None, []] and self.bbox_over_IOU_threshold(bbox, yolo_bboxes, threshold=0.7):
                    # Create new Track
                    new_tracks.append(Track(self.get_id(), track_bbox))


            new_tracks.extend(self.check_and_increment_lifetime(track_id_found))
            self.tracks = sorted(new_tracks, key=lambda t: t.track_id)

    def check_and_increment_lifetime(self, ids_found):
        """ 
            Increment track lifetime if it was not found.
            Returns trackers that are still viable.
        """
        tracks_not_found = []
        for track in self.tracks:
            if track.track_id not in ids_found:
                track.last_updated += 1
                if track.last_updated < 5:
                    tracks_not_found.append(track)
        return tracks_not_found
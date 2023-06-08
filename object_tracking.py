from Trackers.opencv_trackers import OpenCVTracker
from Trackers.optical_flow import OpticalFlow
from Trackers.deep_sort_tracker import DeepSortTracker, SortTracker
from ultralytics import YOLO
from yolo_model import MODEL_PATH
import torch
from io import BytesIO
import sys
import cv2
import random


class ObjectTracking:
    """ Combine YOLO with 3 Trackers """

    def __init__(self, name='', threshold=.45, yolo_path=f'{MODEL_PATH}'):
        self.yolo = YOLO(yolo_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo.to(device)

        self.tracker = self.get_tracker(name)
        self.adnotations = []
        self.frame_counter = 0
        self.object_counter = 0

        if 0 > threshold > 1:
            raise ValueError(f'Threshold {threshold} not in [0,1]')
        self.threshold = threshold

    def get_tracker(self, name):
        """ Return proper tracker based on name. Invalid name defaults to OpenCV CSRT tracker. """
        tracker_dict = {
            'deepsort': DeepSortTracker(), 'sort': SortTracker(),
            'kcf': OpenCVTracker('KCF'), 'medianflow': OpenCVTracker('MEDIANFLOW'), 'csrt': OpenCVTracker(),
            'opticalflow': OpticalFlow()
        }
        return tracker_dict.get(name.lower(), OpenCVTracker())

    def get_video(self, video_path_in, video_path_out='out.mp4'):
        """ Load video from video_path_in.

        Args:
            video_path_in: video path to file with .mp4 extension
            video_path_out: output file path
        """
        self.video_in = cv2.VideoCapture(video_path_in)
        self.next_frame()

        self.cap_out = cv2.VideoWriter(video_path_out,
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       self.video_in.get(cv2.CAP_PROP_FPS),
                                       (self.frame.shape[1],
                                        self.frame.shape[0])
                                       )
        self.fps = self.video_in.get(cv2.CAP_PROP_FPS)

        self.colors = [(random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255)) for _ in range(10)]

    def write_video(self):
        """ Write detections as colored boxes based on track_id """
        for track in self.tracker.tracks:
            x1, y1, x2, y2 = track.bbox
            self.adnotations.append(
                [self.frame_counter, track.track_id, x1, y1, x2, y2])
            cv2.rectangle(self.frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (self.colors[track.track_id % len(self.colors)]), 3)
            cv2.putText(self.frame, 'Drone', (int(x1), int(y1)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (self.colors[track.track_id % len(self.colors)]), 2)
            self.object_counter = track.track_id

        self.cap_out.write(self.frame)

    def detect(self):
        """ Detect object on current frame

        Returns:
            tuple(list, list, numpy.ndarray): detection values 
        """
        current_frame = self.frame
        [results] = self.yolo(current_frame.copy())
        bboxes = []
        scores = []
        for bbox, score in zip(results.boxes.xyxy, results.boxes.conf):
            score = score.item()
            bbox = [int(item) for item in bbox]
            if score > self.threshold:
                bboxes.append(self.yolo_bbox_to_xywh(bbox))
                scores.append(score)

        return bboxes, scores, current_frame

    def yolo_bbox_to_xywh(self, bbox):
        return bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]

    def update_tracker(self, bboxes, scores, frame):
        self.tracker.update(bboxes, scores, frame)

    def next_frame(self):
        """ Load next frame """
        self.frame_returned, self.frame = self.video_in.read()
        self.frame_counter += 1

    def save_adnotations(self):
        """ Save adnotations in proper format """
        self.text_file = BytesIO()
        for item in self.adnotations:
            self.text_file.write((', '.join(str(round(num, 2))
                                 for num in item) + '\n').encode())
        self.text_file.seek(0)

    def get_adnotations(self):
        return self.text_file

    def run(self):
        """ Start Object Tracking

        Returns:
            io.StringIO: return file-like object
        """
        while self.frame_returned:
            detect_tuple = self.detect()
            self.update_tracker(*detect_tuple)
            self.write_video()
            self.next_frame()

        self.save_adnotations()
        self.video_in.release()
        self.cap_out.release()

        return self.text_file


if __name__ == "__main__":
    if len(sys.argv) > 3:
        name = str(sys.argv[2]).lower()
        threshold = float(sys.argv[3])
        ot = ObjectTracking(name=name, threshold=threshold)
        ot.get_video(sys.argv[1])
    else:
        ot = ObjectTracking(name='csrt', threshold=0.45)
        ot.get_video('./drone.mp4')

    print(ot.run().read())

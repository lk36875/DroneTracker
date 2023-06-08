from deep_sort.deep_sort.tracker import Tracker as DeepSort
from sort.sort import Sort

from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection

from tracker import Tracker, Track
from Trackers import encoder_dir
import numpy as np


class DeepSortTracker(Tracker):
    """ Utilize DeepSort library with it's box encoder and metric to track objects. """

    def __init__(self, metric=None, encoder_filename=f'{encoder_dir}/mars-small128.pb'):
        if metric is None:
            metric = nn_matching.NearestNeighborDistanceMetric("cosine", 2)
        self.deep_sort = DeepSort(metric, 2, n_init=2)

        self.encoder = gdet.create_box_encoder(encoder_filename, batch_size=1)
        self.tracks = []

    def update(self, bboxes, scores, frame):
        """ Update tracker using bboxes with standard xywh format (NOT YOLO xywh format)"""
        features = self.encoder(frame, bboxes)

        detections_scores_features = []
        for bbox_id, bbox in enumerate(bboxes):
            detections_scores_features.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        self.deep_sort.predict()
        self.deep_sort.update(detections_scores_features)
        self.update_tracks()

    def update_tracks(self):
        """ Create Track objects from confirmed tracks  """
        tracks = []
        for track in self.deep_sort.tracks:
            if not track.is_confirmed() or track.time_since_update > 2:
                continue
            bbox = track.to_tlbr()
            id = track.track_id

            tracks.append(Track(id, bbox))

        self.tracks = tracks


class SortTracker(Tracker):
    """ Utilize Sort library to track objects """

    def __init__(self):
        self.sort = Sort(max_age=2, min_hits=2, iou_threshold=3)
        self.tracks = []

    def update(self, bboxes, scores, frame=None):
        """ Convert bboxes and scores to right format and update tracks """
        bboxes = self.xywh_to_xyxy(np.array(bboxes).reshape(-1, 4))
        scores = np.array(scores).reshape(-1, 1)
        detections = np.c_[bboxes, scores]

        tracked = self.sort.update(detections)

        self.tracks.clear()
        for *bbox, box_id in tracked:
            box_id = int(box_id)
            self.tracks.append(Track(box_id, bbox))

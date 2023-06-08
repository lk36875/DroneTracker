from abc import ABC, abstractmethod
import numpy as np


class Tracker(ABC):
    """ Abstract Tracker class implements update method, 2 concrete methods of bbox conversion and IoU check """
    tracks: list

    @abstractmethod
    def update(self, bboxes, scores, frame):
        """
            Return an updated Track list based on the provided parameters.

            Args:
                bboxes (list[tuple]): bounding boxes in xywh format.
                scores (list[list]): A 2D list of scores.
                frame (numpy.ndarray): An image frame.

            After calling update(), tracker contains updated attribute tracks.
        """
        pass

    def xywh_to_xyxy(self, bboxes):
        """ Convert bboxes from xywh to xyxy format """
        return np.hstack((bboxes[:, 0:2], bboxes[:, 0:2] + bboxes[:, 2:4]))

    def xyxy_to_xywh(self, bboxes):
        """ Convert bboxes from xyxy to xywh format """
        return np.hstack((bboxes[:, 0:2], bboxes[:, 2:4] - bboxes[:, 0:2]))

    def bbox_over_IOU_threshold(self, bbox, bboxes, threshold=0, get_idx=False):
        """ 
            Check IOU score for bbox comparing with each bbox in bboxes.
            If get_idx is set to True, return id of first matching bbox in passed list. 
        """
        x, y, w, h = bbox
        bboxes = np.array(bboxes)
        x_left = np.maximum(bboxes[:, 0], x)
        y_top = np.maximum(bboxes[:, 1], y)
        x_right = np.minimum(x + w, bboxes[:, 0] + bboxes[:, 2])
        y_bottom = np.minimum(y + h, bboxes[:, 1] + bboxes[:, 3])

        intersection_area = np.maximum(0, x_right - x_left) * np.maximum(0, y_bottom - y_top)

        # intersecting bboxes
        possible_idx = intersection_area > 0
        intersection_area = intersection_area[possible_idx]

        union_area = w*h + bboxes[possible_idx, 2] * bboxes[possible_idx, 3] - intersection_area
        iou = intersection_area / union_area

        if get_idx:
            indices = np.where(possible_idx)[0]
            idx_found = np.where(iou > threshold)[0]
            if idx_found.size:
                return indices[idx_found].min()
            else:
                return None
        else:
            return np.any(iou > threshold)


class Track:
    """ Track object stores track_id and bbox in xyxy format """

    def __init__(self, id, bbox, last_updated=0):
        self.track_id = id
        self.bbox = bbox
        self.last_updated = last_updated

    def __repr__(self):
        return f"<Track ID: {self.track_id}, Bounding Box: {self.bbox}>"
    
    def __eq__(self, other):
        bbox_equal = all([x == y for x, y in zip(self.bbox, other.bbox)])
        return self.track_id == other.track_id and bbox_equal

    def get_xywh(self):
        return (self.bbox[0], self.bbox[1], self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])


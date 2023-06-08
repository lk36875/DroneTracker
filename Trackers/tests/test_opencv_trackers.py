import unittest
import numpy as np
from tracker import Track
from opencv_trackers import OpenCVTracker
import cv2
import os
from .files import DRONE_PATH

class OpenCVTrackerTests(unittest.TestCase):

    def setUp(self):
        self.opencv_tracker = OpenCVTracker()
        self.video_in = cv2.VideoCapture(f"{DRONE_PATH}")
        _, self.frame = self.video_in.read()

    def test_get_bboxes(self):
        self.opencv_tracker.trackers.add(cv2.legacy.TrackerKCF_create(
        ), np.zeros((100, 100, 3), dtype=np.uint8), (10, 10, 20, 20))
        self.opencv_tracker.tracks_id = [0]
        bboxes = self.opencv_tracker.get_bboxes()
        expected_bboxes = [Track(0, [10, 10, 30, 30])]
        self.assertEqual(bboxes, expected_bboxes)

    def test_update_trackers_detection(self):
        self.opencv_tracker.trackers.add(
            cv2.legacy.TrackerKCF_create(), self.frame.copy(), (10, 10, 20, 20))

        bboxes = [[112, 112, 112, 112], [10, 10, 20, 20]]
        self.opencv_tracker.update_trackers(bboxes, self.frame.copy())
        np.testing.assert_equal(self.opencv_tracker.trackers.getObjects()[1], [112, 112, 112, 112])

    def test_update_trackers_empty(self):
        self.opencv_tracker.trackers.add(
            cv2.legacy.TrackerKCF_create(), self.frame.copy(), (10, 10, 20, 20))

        bboxes = []
        self.opencv_tracker.update_trackers(bboxes, self.frame.copy())
        np.testing.assert_equal(len(self.opencv_tracker.trackers.getObjects()), 1)

    def test_remove_unused(self):
        self.opencv_tracker.trackers.add(
            cv2.legacy.TrackerKCF_create(), self.frame.copy(), (10, 10, 20, 20))

        self.opencv_tracker.last_trackers = np.array([[10, 10, 20, 20]])
        self.opencv_tracker.remove_unused(self.frame.copy())
        np.testing.assert_equal(len(self.opencv_tracker.trackers.getObjects()), 0)

    def test_remove_no_matches(self):
        self.opencv_tracker.trackers.add(
            cv2.legacy.TrackerKCF_create(), self.frame.copy(), (10, 10, 20, 20))

        self.opencv_tracker.last_trackers = np.array([[12, 12, 22, 22]])
        self.opencv_tracker.remove_unused(self.frame.copy())
        np.testing.assert_equal(len(self.opencv_tracker.trackers.getObjects()), 1)

    def tearDown(self):
        self.video_in.release()


if __name__ == '__main__':
    unittest.main()

from object_tracking import ObjectTracking
from Trackers.deep_sort_tracker import DeepSortTracker, SortTracker
from Trackers.opencv_trackers import OpenCVTracker
from Trackers.tracker import Track
from .files import DRONE_PATH
import unittest
import cv2
import numpy as np

from Trackers.optical_flow import OpticalFlow

class ObjectTrackingTests(unittest.TestCase):
    def setUp(self):
        self.tracking = ObjectTracking(name='csrt', threshold=0.45)
        self.tracking.get_video(f'{DRONE_PATH}')

    def tearDown(self):
        self.tracking.video_in.release()
        self.tracking.cap_out.release()

    def test_init(self):
        self.assertIsNotNone(self.tracking.yolo)
        self.assertIsNotNone(self.tracking.tracker)
        self.assertEqual(len(self.tracking.adnotations), 0)
        self.assertEqual(self.tracking.frame_counter, 1)
        self.assertEqual(self.tracking.object_counter, 0)
        self.assertEqual(self.tracking.threshold, 0.45)

    def test_get_tracker(self):
        tracker1 = self.tracking.get_tracker('deepsort')
        tracker2 = self.tracking.get_tracker('sort')
        tracker3 = self.tracking.get_tracker('kcf')
        tracker4 = self.tracking.get_tracker('medianflow')
        tracker5 = self.tracking.get_tracker('csrt')
        tracker6 = self.tracking.get_tracker('opticalflow')
        tracker7 = self.tracking.get_tracker('invalid')

        self.assertIsInstance(tracker1, DeepSortTracker)
        self.assertIsInstance(tracker2, SortTracker)
        self.assertIsInstance(tracker3, OpenCVTracker)
        self.assertIsInstance(tracker4, OpenCVTracker)
        self.assertIsInstance(tracker5, OpenCVTracker)
        self.assertIsInstance(tracker6, OpticalFlow)
        self.assertIsInstance(tracker7, OpenCVTracker)

    def test_get_video(self):
        self.assertAlmostEqual(self.tracking.video_in.get(cv2.CAP_PROP_FPS), self.tracking.fps)

    def test_write_video(self):
        self.tracking.tracker.tracks = [
            Track(bbox=[10, 20, 30, 40], id=1),
            Track(bbox=[50, 60, 70, 80], id=2)
        ]

        self.tracking.write_video()

        self.assertEqual(self.tracking.object_counter, 2)
        self.assertEqual(self.tracking.adnotations[0][1], 1)
        self.assertEqual(self.tracking.adnotations[1][1], 2)

    def test_detect(self):
        bboxes, scores, frame = self.tracking.detect()

        self.assertIsInstance(bboxes, list)
        self.assertIsInstance(scores, list)
        self.assertIsInstance(frame, np.ndarray)

    def test_yolo_bbox_to_xywh(self):
        bbox = [10, 20, 30, 40]
        result = self.tracking.yolo_bbox_to_xywh(bbox)
        self.assertEqual(result, (10, 20, 20, 20))

    def test_update_tracker(self):
        bboxes = [[10, 20, 30, 40]]
        scores = [0.8]
        frame = self.tracking.frame

        self.tracking.update_tracker(bboxes, scores, frame)

        track = Track(0, [9, 19, 39, 59])
        self.assertEqual(len(self.tracking.tracker.tracks), 1)
        self.assertTrue(self.tracking.tracker.tracks[0] == track)

    def test_next_frame(self):
        self.tracking.next_frame()

        self.assertEqual(self.tracking.frame_counter, 2)
        self.assertTrue(self.tracking.frame_returned)

    def test_save_adnotations(self):
        self.tracking.adnotations = [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12]
        ]

        self.tracking.save_adnotations()

        expected_text = '1, 2, 3, 4, 5, 6\n7, 8, 9, 10, 11, 12\n'
        self.assertEqual(self.tracking.text_file.getvalue(), expected_text)


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
from tracker import Track
from deep_sort_tracker import DeepSortTracker, SortTracker
import unittest


class DeepSortTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = DeepSortTracker()

    def test_init(self):
        self.assertIsNotNone(self.tracker.deep_sort)
        self.assertIsNotNone(self.tracker.encoder)
        self.assertEqual(len(self.tracker.tracks), 0)

    def test_update(self):
        bboxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
        scores = np.array([0.8, 0.9])
        frame = np.zeros((100, 100, 3))

        self.tracker.update(bboxes, scores, frame)
        self.tracker.update(bboxes, scores, frame)

        self.assertNotEqual(len(self.tracker.tracks), 0)

    def test_update_tracks(self):
        dummy_tracks = [
            DummyTrack(confirmed=True, time_since_update=1, tlbr=[10, 20, 30, 40], track_id=1),
            DummyTrack(confirmed=False, time_since_update=3, tlbr=[50, 60, 70, 80], track_id=2)
        ]
        self.tracker.deep_sort.tracks = dummy_tracks

        self.tracker.update_tracks()

        self.assertEqual(len(self.tracker.tracks), 1)
        self.assertEqual(self.tracker.tracks[0].track_id, 1)
        self.assertEqual(self.tracker.tracks[0].bbox, [10, 20, 30, 40])


class DummyTrack:
    def __init__(self, confirmed, time_since_update, tlbr, track_id):
        self._confirmed = confirmed
        self.time_since_update = time_since_update
        self._tlbr = tlbr
        self.track_id = track_id

    def is_confirmed(self):
        return self._confirmed

    def to_tlbr(self):
        return self._tlbr


class SortTests(unittest.TestCase):

    def setUp(self):
        self.tracker = SortTracker()

    def test_update(self):
        bboxes = [[1, 1, 2, 2], [1, 2, 3, 4]]
        scores = [1, 0.1]
        self.tracker.update(bboxes, scores)
        correct_res = [Track(2, [1.0, 1.0, 4.0, 6.0]), Track(1, [1.0, 1.0, 3.0, 3.0])]
        self.assertTrue(self.tracker.tracks[0], correct_res[0])
        self.assertTrue(self.tracker.tracks[1], correct_res[1])


if __name__ == '__main__':
    unittest.main()

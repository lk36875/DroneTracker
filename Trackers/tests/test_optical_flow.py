import unittest
import numpy as np
from tracker import Track
from optical_flow import OpticalFlow


class OpticalFlowTests(unittest.TestCase):

    def setUp(self):
        self.optical_flow = OpticalFlow()

    def test_initialization(self):
        self.assertIsNone(self.optical_flow.last_frame)
        self.assertIsNone(self.optical_flow.curr_frame)
        self.assertEqual(self.optical_flow.id, -1)
        self.assertEqual(len(self.optical_flow.tracks), 0)
        self.assertIsInstance(self.optical_flow.last_trackers, np.ndarray)
        self.assertEqual(len(self.optical_flow.last_trackers), 0)

    def test_get_id(self):
        self.assertEqual(self.optical_flow.get_id(), 0)
        self.assertEqual(self.optical_flow.get_id(), 1)
        self.assertEqual(self.optical_flow.get_id(), 2)

    def test_convert_frame_to_gray(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        gray_frame = self.optical_flow.convert_frame_to_gray(frame)
        self.assertEqual(gray_frame.shape, (10, 10))
        self.assertEqual(gray_frame.dtype, np.uint8)

    def test_calculate_flow(self):
        self.optical_flow.last_frame = np.zeros((10, 10), dtype=np.uint8)
        self.optical_flow.curr_frame = np.ones((10, 10), dtype=np.uint8)
        flow = self.optical_flow.calculate_flow()
        self.assertIsInstance(flow, np.ndarray)
        self.assertEqual(flow.shape, (10, 10, 2))

    def test_get_bbox_from_flow_multiple(self):
        flow_map = np.zeros((1000, 1000, 2), dtype=np.float32)
        flow_map[20:50, 20:50, :] = 2.0
        flow_map[70:90, 70:90, :] = 2.0
        bounding_boxes = self.optical_flow.get_bbox_from_flow(flow_map)
        self.assertEqual(len(bounding_boxes), 2)
        self.assertIsInstance(bounding_boxes, np.ndarray)

    def test_get_bbox_from_flow_no_movement(self):
        flow_map = np.zeros((1000, 1000, 2), dtype=np.float32)
        bounding_boxes = self.optical_flow.get_bbox_from_flow(flow_map)
        self.assertEqual(len(bounding_boxes), 0)
        self.assertIsInstance(bounding_boxes, np.ndarray)

    def test_get_bbox_from_flow_small_movement(self):
        flow_map = np.zeros((1000, 1000, 2), dtype=np.float32)
        flow_map[2:5, 20:5, :] = 4.0
        flow_map[7:9, 70:9, :] = 4.0
        bounding_boxes = self.optical_flow.get_bbox_from_flow(flow_map)
        self.assertEqual(len(bounding_boxes), 0)

    def test_get_bbox_from_flow_single_bbox(self):
        flow_map = np.zeros((1000, 1000, 2), dtype=np.float32)
        flow_map[20:70, 30:90, :] = 3.0
        bounding_boxes = self.optical_flow.get_bbox_from_flow(flow_map)
        self.assertEqual(len(bounding_boxes), 1)
        self.assertIsInstance(bounding_boxes, np.ndarray)
        self.assertTupleEqual(tuple(bounding_boxes[0]), (30, 20, 60, 50))

    def test_check_and_increment_lifetime_no_tracks(self):
        tracks_not_found = self.optical_flow.check_and_increment_lifetime([])
        self.assertEqual(len(tracks_not_found), 0)

    def test_check_and_increment_lifetime_tracks_found_all_ids(self):
        track1 = Track(1, last_updated=2, bbox=[])
        track2 = Track(2, last_updated=3, bbox=[])
        self.optical_flow.tracks = [track1, track2]
        tracks_not_found = self.optical_flow.check_and_increment_lifetime([1, 2])
        self.assertEqual(len(tracks_not_found), 0)
        self.assertEqual(track1.last_updated, 2)
        self.assertEqual(track2.last_updated, 3)

    def test_check_and_increment_lifetime_tracks_not_found(self):
        track1 = Track(1, last_updated=2, bbox=[])  # not found
        track2 = Track(2, last_updated=3, bbox=[])  # found
        track3 = Track(3, last_updated=5, bbox=[])  # expired, not added
        track4 = Track(4, last_updated=4, bbox=[])  # found
        self.optical_flow.tracks = [track1, track2, track3, track4]
        ids_found = [2, 4]
        tracks_not_found = self.optical_flow.check_and_increment_lifetime(ids_found)

        self.assertEqual(len(tracks_not_found), 1)

        self.assertEqual(track1.last_updated, 3)
        self.assertEqual(track2.last_updated, 3)
        self.assertEqual(track3.last_updated, 6)
        self.assertEqual(track4.last_updated, 4)

    def test_update_no_last_frame(self):
        bboxes = np.array([[10, 10, 20, 20], [30, 30, 40, 40]])
        scores = np.array([0.9, 0.8])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.optical_flow.update(bboxes, scores, frame)
        np.testing.assert_equal(self.optical_flow.last_frame,
                                self.optical_flow.convert_frame_to_gray(frame))

    def test_update_with_last_frame(self):
        self.optical_flow.last_frame = np.zeros((100, 100), dtype=np.uint8)
        self.optical_flow.curr_frame = np.ones((100, 100), dtype=np.uint8)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.optical_flow.update([], [], frame)
        self.assertIsNotNone(self.optical_flow.curr_frame)

    def test_compare_bboxes_yolo_found(self):
        bboxes_found = np.array([])
        yolo_bboxes = [[1, 2, 3, 4]]
        self.optical_flow.compare_bboxes(bboxes_found, yolo_bboxes)
        t = Track(0, np.array([1, 2, 4, 6]))
        self.assertEqual([t], self.optical_flow.tracks)

    def test_compare_bboxes_yolo_empty(self):
        bboxes_found = np.array([])
        yolo_bboxes = []
        self.optical_flow.compare_bboxes(bboxes_found, yolo_bboxes)
        self.assertEqual([], self.optical_flow.tracks)

    def test_compare_bboxes_yolo_empty(self):
        self.optical_flow.tracks = [Track(0, [1, 2, 4, 6])]
        bboxes_found = np.array([[1, 2, 3, 4.1]])
        yolo_bboxes = []
        self.optical_flow.compare_bboxes(bboxes_found, yolo_bboxes)
        self.assertEqual([Track(0, [1, 2, 4, 6.1])], self.optical_flow.tracks)

    def test_compare_bboxes_first_found(self):
        self.optical_flow.tracks = [Track(0, [1, 2, 4, 6]), Track(1, [1, 2, 4, 6])]
        bboxes_found = np.array([[1, 2, 3, 4.1]])
        yolo_bboxes = []
        self.optical_flow.compare_bboxes(bboxes_found, yolo_bboxes)
        self.assertEqual(Track(0, [1, 2, 4, 6.1]), self.optical_flow.tracks[0])
        self.assertEqual(Track(1, [1, 2, 4, 6]), self.optical_flow.tracks[1])

    def test_compare_bboxes_first_and_yolo(self):
        self.optical_flow.tracks = [Track(12, [1, 2, 4, 6])]
        bboxes_found = np.array([[100, 100, 100, 100], [12, 32, 12, 12]])
        yolo_bboxes = [[100, 100, 100, 100]]
        self.optical_flow.compare_bboxes(bboxes_found, yolo_bboxes)
        self.assertEqual(2, len(self.optical_flow.tracks))
        self.assertEqual(Track(0, [100, 100, 200, 200]), self.optical_flow.tracks[0])


if __name__ == '__main__':
    unittest.main()

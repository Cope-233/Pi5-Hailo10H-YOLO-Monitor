"""Host-only tests. Hailo accelerator and Telegram credentials are not required."""

import unittest
from dataclasses import replace
from unittest.mock import patch, MagicMock
from datetime import datetime

import cv2
import numpy as np

from monitor import MonitorState, Settings, UVCCamera, decode_scale, letterbox, nms_xyxy


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.settings = replace(Settings(), empty_timeout=10, rearm_seconds=5,
                                telegram_retry=60, snapshot_lookback=20,
                                require_prior_presence=False)

    def test_letterbox(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        output, ratio, x, y = letterbox(frame, 640, 640)
        self.assertEqual(output.shape, (640, 640, 3))
        self.assertEqual((ratio, x, y), (1, 0, 140))

    def test_nms_rejects_overlapping_person_boxes(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
        scores = np.array([0.95, 0.8, 0.9], dtype=np.float32)
        self.assertEqual(nms_xyxy(boxes, scores, .45).tolist(), [0, 2])

    def test_yolo26_reg_max_one_decode(self):
        boxes = np.array([[[1., 2., 3., 4.]]], np.float32)
        classes = np.full((1, 1, 80), -10., dtype=np.float32)
        classes[0, 0, 0] = 2.1972246  # sigmoid -> 0.9
        coords, scores = decode_scale(boxes, classes, 8)
        np.testing.assert_allclose(coords, [[-4, -12, 28, 36]])
        self.assertAlmostEqual(float(scores[0]), .9, places=5)

    def test_empty_alert_once_then_rearm(self):
        s = MonitorState()
        self.assertFalse(s.observe_empty(100, self.settings))
        self.assertFalse(s.observe_empty(109, self.settings))
        self.assertTrue(s.observe_empty(110, self.settings))
        s.alert_sent = True
        self.assertFalse(s.observe_empty(120, self.settings))
        s.record_person(125, datetime.now(), b'jpg', 1.0, self.settings)
        self.assertTrue(s.alert_sent)
        s.record_person(130, datetime.now(), b'jpg2', 1.0, self.settings)
        self.assertFalse(s.alert_sent)
        self.assertIsNone(s.empty_since)

    def test_camera_error_resets_empty_countdown(self):
        s = MonitorState()
        s.observe_empty(10, self.settings)
        s.reset_on_error()
        self.assertFalse(s.observe_empty(30, self.settings))
        self.assertTrue(s.observe_empty(40, self.settings))

    def test_telegram_failure_throttling(self):
        s = MonitorState()
        s.observe_empty(0, self.settings)
        self.assertTrue(s.observe_empty(10, self.settings))
        s.last_alert_attempt = 10  # Failed Telegram request
        self.assertFalse(s.observe_empty(30, self.settings))
        self.assertTrue(s.observe_empty(70, self.settings))

    def test_prior_presence_gate(self):
        s = MonitorState()
        settings = replace(self.settings, require_prior_presence=True)
        s.observe_empty(0, settings)
        self.assertFalse(s.observe_empty(20, settings))
        s.record_person(21, datetime.now(), None, None, settings)
        self.assertFalse(s.observe_empty(22, settings))
        self.assertTrue(s.observe_empty(32, settings))

    def test_uvc_camera_opens_reads_and_closes(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, frame)
        with patch('monitor.cv2.VideoCapture', return_value=capture) as open_camera:
            camera = UVCCamera(self.settings)
            self.assertIs(camera.read(), frame)
            open_camera.assert_called_once_with('/dev/video0', cv2.CAP_V4L2)
            self.assertTrue(capture.set.called)
            camera.close()
            capture.release.assert_called_once()
            self.assertIsNone(camera.capture)

    def test_uvc_failure_releases_device_and_reconnects(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        first = MagicMock()
        first.isOpened.return_value = True
        first.read.return_value = (False, None)
        second = MagicMock()
        second.isOpened.return_value = True
        second.read.return_value = (True, frame)
        with patch('monitor.cv2.VideoCapture', side_effect=[first, second]):
            camera = UVCCamera(self.settings)
            with self.assertRaises(OSError):
                camera.read()
            first.release.assert_called_once()
            self.assertIs(camera.read(), frame)
            camera.close()
            second.release.assert_called_once()

    def test_uvc_unavailable_camera(self):
        capture = MagicMock()
        capture.isOpened.return_value = False
        with patch('monitor.cv2.VideoCapture', return_value=capture):
            with self.assertRaises(OSError):
                UVCCamera(self.settings).read()
            capture.release.assert_called_once()

    def test_snapshot_quality_and_encoding(self):
        from monitor import annotated_jpeg, snapshot_quality
        frame = np.zeros((100, 100, 3), np.uint8)
        boxes = np.array([[5, 5, 40, 40]], np.float32)
        scores = np.array([.9], np.float32)
        result = annotated_jpeg(frame, boxes, scores, 90)
        self.assertEqual(cv2.imdecode(np.frombuffer(result, np.uint8), 1).shape, frame.shape)
        self.assertGreater(snapshot_quality(frame, boxes[0], .9), 0)


if __name__ == '__main__':
    unittest.main()
#!/usr/bin/env python3
"""Raspberry Pi 5 + Hailo-10H YOLO26 person/empty-room Telegram monitor.

Uses HailoRT 5.4 InferModel directly. No Ultralytics runtime monkey patch,
Torch inference, or fake/dummy image batches are required.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests

LOG = logging.getLogger("hailo-monitor")
BASE_DIR = Path(__file__).resolve().parent
STOP = False


def env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    camera_device: str = field(default_factory=lambda: os.getenv("CAMERA_DEVICE", "/dev/video0"))
    camera_width: int = field(default_factory=lambda: int(os.getenv("CAMERA_WIDTH", "640")))
    camera_height: int = field(default_factory=lambda: int(os.getenv("CAMERA_HEIGHT", "480")))
    camera_fps: int = field(default_factory=lambda: int(os.getenv("CAMERA_FPS", "15")))
    camera_fourcc: str = field(default_factory=lambda: os.getenv("CAMERA_FOURCC", "MJPG").upper())
    hef_path: Path = field(default_factory=lambda: Path(os.getenv("HEF_PATH", str(BASE_DIR / "models/yolo26n.hef"))))
    conf_threshold: float = field(default_factory=lambda: float(os.getenv("CONF_THRESHOLD", "0.35")))
    nms_iou_threshold: float = field(default_factory=lambda: float(os.getenv("NMS_IOU_THRESHOLD", "0.45")))
    inference_interval: float = field(default_factory=lambda: float(os.getenv("INFERENCE_INTERVAL_SECONDS", "0.5")))
    empty_timeout: float = field(default_factory=lambda: float(os.getenv("EMPTY_TIMEOUT_SECONDS", "300")))
    rearm_seconds: float = field(default_factory=lambda: float(os.getenv("REARM_PERSON_SECONDS", "10")))
    snapshot_lookback: float = field(default_factory=lambda: float(os.getenv("SNAPSHOT_LOOKBACK_SECONDS", "20")))
    snapshot_interval: float = field(default_factory=lambda: float(os.getenv("SNAPSHOT_INTERVAL_SECONDS", "0.5")))
    max_candidates: int = field(default_factory=lambda: int(os.getenv("MAX_SNAPSHOT_CANDIDATES", "60")))
    jpeg_quality: int = field(default_factory=lambda: int(os.getenv("JPEG_QUALITY", "92")))
    status_interval: float = field(default_factory=lambda: float(os.getenv("STATUS_LOG_INTERVAL_SECONDS", "10")))
    telegram_enabled: bool = field(default_factory=lambda: env_bool("TELEGRAM_ENABLED", True))
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    telegram_retry: float = field(default_factory=lambda: float(os.getenv("TELEGRAM_RETRY_SECONDS", "60")))
    require_prior_presence: bool = field(default_factory=lambda: env_bool("REQUIRE_PRIOR_PRESENCE", False))
    save_snapshots: bool = field(default_factory=lambda: env_bool("SAVE_EVENT_SNAPSHOTS", False))
    event_dir: Path = field(default_factory=lambda: Path(os.getenv("EVENT_DIR", str(BASE_DIR / "events"))))

    def validate(self) -> None:
        if self.telegram_enabled and (not self.telegram_bot_token or not self.telegram_chat_id):
            raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required when Telegram is enabled")
        if not self.hef_path.is_file():
            raise FileNotFoundError(f"Hailo-10H HEF not found: {self.hef_path}")
        if not self.camera_device or min(self.camera_width, self.camera_height, self.camera_fps) <= 0:
            raise ValueError("A camera device, positive dimensions, and FPS are required")
        if self.camera_fourcc and len(self.camera_fourcc) != 4:
            raise ValueError("CAMERA_FOURCC must be four characters or empty")
        if not (0 < self.conf_threshold <= 1 and 0 < self.nms_iou_threshold <= 1):
            raise ValueError("Confidence and NMS IoU thresholds must be within (0, 1]")
        if min(self.inference_interval, self.empty_timeout, self.rearm_seconds,
               self.snapshot_lookback, self.snapshot_interval,
               self.status_interval, self.telegram_retry) <= 0:
            raise ValueError("All timing settings must be greater than zero")
        if self.max_candidates < 1 or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("MAX_SNAPSHOT_CANDIDATES or JPEG_QUALITY is out of range")


def letterbox(frame: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, int, int]:
    src_h, src_w = frame.shape[:2]
    if src_h == 0 or src_w == 0:
        raise ValueError("Empty camera image")
    ratio = min(width / src_w, height / src_h)
    new_w, new_h = round(src_w * ratio), round(src_h * ratio)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x, y = (width - new_w) // 2, (height - new_h) // 2
    padded = np.full((height, width, 3), 114, dtype=np.uint8)
    padded[y:y + new_h, x:x + new_w] = resized
    return padded, ratio, x, y


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Class-0 (person) non-maximum suppression."""
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int32)
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        lower = np.maximum(boxes[i, :2], boxes[rest, :2])
        upper = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        size = np.maximum(0, upper - lower)
        inter = size[:, 0] * size[:, 1]
        union = areas[i] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int32)


def decode_scale(box_tensor: np.ndarray, class_tensor: np.ndarray, stride: float) -> tuple[np.ndarray, np.ndarray]:
    """Decode raw YOLO26 reg_max=1 LTRB offsets and class-0 logits.

    Only compatible with a particular non-NMS YOLO26 Hailo export: NHWC
    (H,W,4) regression and (H,W,80) raw class logits at each scale.
    """
    h, w, channels = box_tensor.shape
    if channels != 4 or class_tensor.shape[:2] != (h, w) or class_tensor.shape[-1] < 1:
        raise ValueError("Unexpected YOLO26 detection head dimensions")
    distances = box_tensor.reshape(-1, 4)
    logits = class_tensor[..., 0].reshape(-1)
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cx, cy = gx.reshape(-1) + 0.5, gy.reshape(-1) + 0.5
    boxes = np.column_stack((
        cx - distances[:, 0], cy - distances[:, 1],
        cx + distances[:, 2], cy + distances[:, 3]
    )).astype(np.float32)
    return boxes * stride, scores


class HailoYOLO26PersonDetector:
    """Single-frame HailoRT 5.4 InferModel implementation; no dummy batches."""

    def __init__(self, hef_path: Path, confidence: float, nms_iou: float):
        # Import only on a Pi with the matching HailoRT Python wheel installed.
        from hailo_platform import FormatType, VDevice

        self.confidence = confidence
        self.nms_iou = nms_iou
        self.device = VDevice()
        self.context = None
        try:
            self.model = self.device.create_infer_model(str(hef_path))
            self.model.set_batch_size(1)  # Real single-image model configuration, not a batch workaround.
            input_info = self.model.input()
            input_info.set_format_type(FormatType.UINT8)
            shape = tuple(input_info.shape)
            if len(shape) != 3 or shape[-1] != 3:
                raise ValueError(f"Expected HWC RGB HEF input, got {shape}")
            self.height, self.width, _ = shape

            self.box_heads, self.class_heads = {}, {}
            self.outputs = list(self.model.outputs)
            for output in self.outputs:
                output.set_format_type(FormatType.FLOAT32)
                shape = tuple(output.shape)
                if len(shape) != 3:
                    continue
                key = shape[:2]
                if shape[-1] == 4:
                    self.box_heads[key] = output.name
                elif shape[-1] == 80:
                    self.class_heads[key] = output.name

            self.scales = sorted(set(self.box_heads) & set(self.class_heads), reverse=True)
            if len(self.scales) != 3:
                raise ValueError("Expected 3 YOLO26 raw head pairs (box=4/class=80); check HEF export")
            self.context = self.model.configure()
            self.configured = self.context.__enter__()
            self.input_buffer = np.empty((self.height, self.width, 3), dtype=np.uint8)
            self.output_buffers = {
                o.name: np.empty(tuple(o.shape), dtype=np.float32) for o in self.outputs
            }
            self.bindings = self.configured.create_bindings()
            self.bindings.input().set_buffer(self.input_buffer)
            for name, buf in self.output_buffers.items():
                self.bindings.output(name).set_buffer(buf)
            LOG.info("Hailo-10H InferModel ready: %sx%s, heads=%s", self.width, self.height, self.scales)
        except Exception:
            self.close()
            raise

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        pre, ratio, pad_x, pad_y = letterbox(frame, self.width, self.height)
        np.copyto(self.input_buffer, cv2.cvtColor(pre, cv2.COLOR_BGR2RGB))
        start = time.perf_counter()
        self.configured.run([self.bindings], 10000)
        infer_ms = (time.perf_counter() - start) * 1000
        boxes_list, scores_list = [], []
        for h, w in self.scales:
            stride = self.width / w
            boxes, scores = decode_scale(
                self.output_buffers[self.box_heads[(h, w)]],
                self.output_buffers[self.class_heads[(h, w)]], stride
            )
            mask = scores >= self.confidence
            if np.any(mask):
                boxes_list.append(boxes[mask])
                scores_list.append(scores[mask])
        if not boxes_list:
            return np.empty((0, 4), np.float32), np.empty(0, np.float32), infer_ms

        boxes = np.concatenate(boxes_list)
        scores = np.concatenate(scores_list)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.height)
        keep = nms_xyxy(boxes, scores, self.nms_iou)[:300]
        boxes, scores = boxes[keep], scores[keep]
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
        frame_h, frame_w = frame.shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame_h)
        return boxes, scores, infer_ms

    def close(self) -> None:
        if getattr(self, "context", None) is not None:
            try:
                self.context.__exit__(None, None, None)
            except Exception:
                LOG.exception("Error closing configured model")
            self.context = None
        if getattr(self, "device", None) is not None:
            try:
                self.device.release()
            except Exception:
                LOG.exception("Error releasing Hailo device")
            self.device = None


class UVCCamera:
    """Read a directly attached, driverless UVC webcam through Linux V4L2.

    A failed read releases the device, allowing the next iteration to reopen it.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        device = self.settings.camera_device
        source: str | int = int(device) if device.isdecimal() else device
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            raise OSError(f"Cannot open UVC camera: {device}")
        if self.settings.camera_fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.settings.camera_fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.camera_height)
        cap.set(cv2.CAP_PROP_FPS, self.settings.camera_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture = cap
        LOG.info("UVC camera opened: %s", device)

    def read(self) -> np.ndarray:
        if self.capture is None:
            self.open()
        ok, frame = self.capture.read()
        if not ok or frame is None or frame.size == 0:
            self.close()
            raise OSError("UVC camera read failed")
        return frame

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


def annotated_jpeg(frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray, quality: int) -> bytes | None:
    image = frame.copy()
    for coords, score in zip(boxes, scores):
        x1, y1, x2, y2 = (int(v) for v in coords)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"person {score:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return encoded.tobytes() if ok else None


def snapshot_quality(frame: np.ndarray, box: np.ndarray, confidence: float) -> float:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = frame[y1:y2, x1:x2]
    sharpness = float(cv2.Laplacian(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    area_score = min((x2 - x1) * (y2 - y1) / (w * h) * 5.0, 1.0)
    return confidence * 1.5 + area_score + min(sharpness / 500.0, 1.0)


class Telegram:
    def __init__(self, session: requests.Session, settings: Settings):
        self.session, self.settings = session, settings

    def alert(self, jpeg: bytes | None, last_seen: datetime | None, seconds: float) -> bool:
        if not self.settings.telegram_enabled:
            LOG.info("Telegram disabled; empty-room alert simulated")
            return True
        seen = last_seen.strftime("%Y-%m-%d %H:%M:%S") if last_seen else "Unknown"
        caption = ("Study room empty-room alert\n"
                   f"No person detected for {seconds:.0f} seconds.\n"
                   f"Last person seen: {seen}\n"
                   f"Alert time: {datetime.now():%Y-%m-%d %H:%M:%S}")
        operation = "sendPhoto" if jpeg else "sendMessage"
        url = f"https://api.telegram.org/bot{self.settings.telegram_bot_token}/{operation}"
        try:
            payload = {"chat_id": self.settings.telegram_chat_id}
            if jpeg:
                payload["caption"] = caption
                response = self.session.post(url, data=payload,
                                             files={"photo": ("last_person.jpg", jpeg, "image/jpeg")}, timeout=30)
            else:
                payload["text"] = caption
                response = self.session.post(url, data=payload, timeout=15)
            response.raise_for_status()
            # Telegram can return a JSON error; do not expose the bot token in logs.
            if response.json().get("ok") is not True:
                LOG.warning("Telegram reported an unsuccessful response")
                return False
            return True
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            LOG.warning("Telegram HTTP error: status=%s", status)
        except (requests.exceptions.RequestException, ValueError):
            LOG.warning("Telegram request failed (no credentials logged)")
        return False


@dataclass
class MonitorState:
    """Tracks consecutive *successful* detections; camera errors never mean empty."""
    empty_since: float | None = None
    alert_sent: bool = False
    rearm_since: float | None = None
    last_seen: datetime | None = None
    last_jpeg: bytes | None = None
    candidates: deque = field(default_factory=deque)
    last_candidate_at: float = float("-inf")
    last_alert_attempt: float = float("-inf")

    def reset_on_error(self) -> None:
        self.empty_since = None
        self.rearm_since = None

    def record_person(self, now: float, wall: datetime, jpg: bytes | None,
                      candidate_score: float | None, settings: Settings) -> None:
        self.empty_since = None
        self.last_seen = wall
        if self.alert_sent:
            if self.rearm_since is None:
                self.rearm_since = now
            elif now - self.rearm_since >= settings.rearm_seconds:
                self.alert_sent = False
                self.rearm_since = None
                self.last_jpeg = None
                self.candidates.clear()
                LOG.info("Monitor rearmed after sustained person presence")
        else:
            self.rearm_since = None
        if jpg:
            self.last_jpeg = jpg
            if candidate_score is not None and now - self.last_candidate_at >= settings.snapshot_interval:
                self.candidates.append((now, candidate_score, jpg))
                self.last_candidate_at = now
        while self.candidates and now - self.candidates[0][0] > settings.snapshot_lookback:
            self.candidates.popleft()

    def observe_empty(self, now: float, settings: Settings) -> bool:
        self.rearm_since = None
        if self.empty_since is None:
            self.empty_since = now
            LOG.info("Empty-room timer started")
        if self.alert_sent or now - self.empty_since < settings.empty_timeout:
            return False
        if settings.require_prior_presence and self.last_seen is None:
            return False
        return now - self.last_alert_attempt >= settings.telegram_retry

    def best_jpeg(self) -> bytes | None:
        return max(self.candidates, key=lambda x: x[1])[2] if self.candidates else self.last_jpeg


def save_event(jpg: bytes, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = path / f"empty_{datetime.now():%Y%m%d_%H%M%S}.jpg"
    # Limit access even if global umask is permissive.
    fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(jpg)
    LOG.info("Saved event snapshot: %s", target)


def handle_signal(signum: int, _frame: object) -> None:
    global STOP
    STOP = True
    LOG.info("Received signal %s", signum)


def run(settings: Settings) -> None:
    global STOP
    STOP = False
    settings.validate()
    detector = HailoYOLO26PersonDetector(settings.hef_path, settings.conf_threshold,
                                         settings.nms_iou_threshold)
    state = MonitorState(candidates=deque(maxlen=settings.max_candidates))
    camera = UVCCamera(settings)
    with requests.Session() as tg:
        messenger = Telegram(tg, settings)
        last_log = float("-inf")
        last_camera_error = float("-inf")
        try:
            LOG.info("Monitoring started; camera=%s", settings.camera_device)
            while not STOP:
                loop_start = time.monotonic()
                try:
                    frame = camera.read()
                    boxes, scores, infer_ms = detector.detect(frame)
                except Exception as exc:  # Includes HailoRT device errors; do not treat failure as absence.
                    state.reset_on_error()
                    if time.monotonic() - last_camera_error >= 60:
                        LOG.warning("Camera/inference error (%s); empty timer reset", type(exc).__name__)
                        last_camera_error = time.monotonic()
                    time.sleep(settings.inference_interval)
                    continue

                now, wall = time.monotonic(), datetime.now()
                if len(boxes):
                    jpeg = annotated_jpeg(frame, boxes, scores, settings.jpeg_quality)
                    quality = (snapshot_quality(frame, boxes[0], float(scores[0]))
                               if len(boxes) == 1 else None)
                    state.record_person(now, wall, jpeg, quality, settings)
                elif state.observe_empty(now, settings):
                    state.last_alert_attempt = now
                    elapsed = now - state.empty_since
                    jpeg = state.best_jpeg()
                    if messenger.alert(jpeg, state.last_seen, elapsed):
                        state.alert_sent = True
                        LOG.info("Empty-room alert delivered")
                        if jpeg and settings.save_snapshots:
                            try:
                                save_event(jpeg, settings.event_dir)
                            except OSError:
                                LOG.exception("Failed to save local event snapshot")
                    else:
                        LOG.warning("Alert delivery failed; will retry in %.0fs", settings.telegram_retry)

                if now - last_log >= settings.status_interval:
                    duration = now - state.empty_since if state.empty_since is not None else 0.0
                    LOG.info("persons=%d infer=%.1fms empty=%.0fs alerted=%s",
                             len(boxes), infer_ms, duration, state.alert_sent)
                    last_log = now
                wait = settings.inference_interval - (time.monotonic() - loop_start)
                if wait > 0:
                    time.sleep(wait)
        finally:
            camera.close()
            detector.close()
            LOG.info("Monitoring stopped")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    run(Settings())


if __name__ == "__main__":
    main()
"""
Robust video capture.

Wraps cv2.VideoCapture with automatic reconnection, which matters for RTSP /
CCTV streams that drop frames or disconnect. For finite video files, set
STREAM_MAX_RECONNECTS = 0 so end-of-file is treated as a clean stop.
"""

import time

import cv2

import config
from logging_setup import get_logger

log = get_logger("video")


class VideoSource:
    def __init__(self, source, max_reconnects=0, reconnect_delay=3.0):
        self.source = source
        self.max_reconnects = max_reconnects
        self.reconnect_delay = reconnect_delay
        self.cap = None
        self._reconnects = 0

    def open(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.cap = None
            return False
        log.info("Opened video source: %r", self.source)
        return True

    @property
    def width(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap else 0

    @property
    def height(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap else 0

    @property
    def fps(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 0
        return fps if fps and fps > 0 else 20.0

    def read(self):
        """
        Read the next frame.

        Returns (ok, frame). ok=False means the stream has ended and no further
        reconnect will be attempted.
        """
        if self.cap is None:
            return False, None

        ok, frame = self.cap.read()
        if ok:
            return True, frame

        # Read failed. Attempt reconnect if configured.
        if self._reconnects >= self.max_reconnects:
            if self.max_reconnects > 0:
                log.error("Stream read failed; reconnect limit reached. Stopping.")
            return False, None

        self._reconnects += 1
        log.warning(
            "Stream read failed. Reconnect attempt %d/%d in %.1fs...",
            self._reconnects, self.max_reconnects, self.reconnect_delay,
        )
        self.release()
        time.sleep(self.reconnect_delay)
        if self.open():
            ok, frame = self.cap.read()
            if ok:
                log.info("Reconnected successfully.")
                return True, frame
        log.warning("Reconnect attempt %d did not yield a frame.", self._reconnects)
        return self.read()  # recurse until limit reached

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def from_config():
    src = config.VIDEO_SOURCE
    # Files should not auto-reconnect on EOF; live streams should.
    is_stream = isinstance(src, int) or (
        isinstance(src, str) and src.lower().startswith(("rtsp://", "http://", "https://"))
    )
    max_rc = config.STREAM_MAX_RECONNECTS if is_stream else 0
    return VideoSource(src, max_reconnects=max_rc, reconnect_delay=config.STREAM_RECONNECT_DELAY)

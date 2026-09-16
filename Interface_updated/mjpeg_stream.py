# mjpeg_stream.py
#
# Minimal MJPEG-over-HTTP streamer for testing the Pepper's Cone wireless
# pipeline: laptop (running live_view.py) -> Wi-Fi -> Raspberry Pi -> TV.
#
# Wired into live_view.py's fullscreen tick: whatever warped frame is being
# shown locally is also pushed here and re-served as a live MJPEG stream that
# any browser can open directly -- including the Pi's kiosk-mode Chromium.
#
# No new dependencies: uses only cv2 (already required by this app) and the
# standard library's http.server.

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BOUNDARY = "ppcone_frame_boundary"

# Fullscreen, letterboxed viewer page. object-fit: contain is important here --
# the warped output is a square canvas (CANVAS_SIZE x CANVAS_SIZE) and must
# NOT be stretched to fill a 16:9 screen, or the reflection will misalign
# with the physical cone.
_VIEWER_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  html,body {{ margin:0; height:100%; background:#000; overflow:hidden; }}
  img {{ width:100vw; height:100vh; object-fit:contain; display:block; }}
</style></head>
<body><img src="/stream.mjpg"></body></html>
"""


class MJPEGStreamer:
    """Holds the latest BGR frame and serves it as an MJPEG stream on its own port."""

    def __init__(self, port=8081, jpeg_quality=80):
        self._port = port
        self._jpeg_quality = jpeg_quality
        self._lock = threading.Lock()
        self._latest_jpeg = None
        self._server = None
        self._thread = None

    def update_frame(self, frame_bgr):
        """Call with the latest warped BGR frame (e.g. from _fullscreen_tick)."""
        ok, buf = cv2.imencode(
            ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
        )
        if ok:
            with self._lock:
                self._latest_jpeg = buf.tobytes()

    def get_latest_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    def start(self):
        if self._server is not None:
            return  # already running
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass  # keep the console quiet during a live show

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = _VIEWER_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=%s" % BOUNDARY,
                    )
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    try:
                        while True:
                            jpg = streamer.get_latest_jpeg()
                            if jpg is not None:
                                self.wfile.write(("--%s\r\n" % BOUNDARY).encode())
                                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                                self.wfile.write(
                                    ("Content-Length: %d\r\n\r\n" % len(jpg)).encode()
                                )
                                self.wfile.write(jpg)
                                self.wfile.write(b"\r\n")
                            time.sleep(1 / 30)  # cap at ~30 fps over the wire
                    except (BrokenPipeError, ConnectionResetError):
                        pass  # viewer closed/reloaded -- normal
                    return

                self.send_response(404)
                self.end_headers()

        self._server = ThreadingHTTPServer(("0.0.0.0", self._port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print("[mjpeg_stream] serving on http://0.0.0.0:%d/  (viewer at '/')" % self._port)

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server = None
            self._thread = None

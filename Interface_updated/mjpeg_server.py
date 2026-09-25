# mjpeg_server.py
#
# Minimal MJPEG-over-HTTP server for streaming the finished cone-warp
# canvas to a browser-based receiver (e.g. a Raspberry Pi in kiosk mode).
# Holds only the single latest encoded frame -- no backlog/queue, so a
# slow network can never cause the stream to fall further and further
# behind; it just skips frames.

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

_BOUNDARY = "frame"


class _MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path != "/stream":
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header(
            "Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        server = self.server.mjpeg_server
        last_sent = None
        try:
            while True:
                jpeg = server.wait_for_frame(last_sent, timeout=1.0)
                if jpeg is None:
                    continue
                last_sent = jpeg
                self.wfile.write(f"--{_BOUNDARY}\r\n".encode())
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


class MJPEGServer:
    """Background HTTP server that streams the latest JPEG frame pushed
    to it via update_frame() as a multipart/x-mixed-replace MJPEG stream.
    Any browser can view it by loading '<local_url()>' directly."""

    def __init__(self, host="0.0.0.0", port=8554):
        self._host = host
        self._port = port
        self._httpd = None
        self._thread = None
        self._cond = threading.Condition()
        self._jpeg = None

    def start(self):
        if self._thread is not None:
            return
        self._httpd = ThreadingHTTPServer((self._host, self._port), _MJPEGHandler)
        self._httpd.mjpeg_server = self
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        self._httpd = None
        self._thread = None
        with self._cond:
            self._jpeg = None

    def update_frame(self, bgr_frame, jpeg_quality=85):
        ok, buf = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            return
        with self._cond:
            self._jpeg = buf.tobytes()
            self._cond.notify_all()

    def wait_for_frame(self, last_seen, timeout=1.0):
        with self._cond:
            self._cond.wait_for(lambda: self._jpeg is not last_seen, timeout=timeout)
            return self._jpeg

    @property
    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def local_url(self, path="/stream"):
        host = self._host
        if host in ("0.0.0.0", ""):
            host = _get_active_local_ip()
        return f"http://{host}:{self._port}{path}"


def _get_active_local_ip():
    """Return the local IP for whichever network is actually routing traffic
    right now. socket.gethostbyname(socket.gethostname()) is unreliable on
    machines with multiple adapters (VPN/virtual NICs) -- it can keep
    returning a stale adapter's IP even after the active network changes.
    Opening a UDP "connection" (no packets are actually sent) forces the OS
    to pick the outbound-route interface, which tracks the active network."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"
    finally:
        s.close()

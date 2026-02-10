"""
Fragmented MP4 (fMP4) encoder for smooth video streaming via MSE + WebSocket.

Spawns an ffmpeg subprocess that encodes raw BGR24 frames to H.264 fMP4
and broadcasts the resulting init segment + media segments to connected
WebSocket clients via per-client queues.
"""

import logging
import queue
import struct
import subprocess
import threading

logger = logging.getLogger(__name__)


class FMP4Encoder:
    """Encodes raw BGR24 frames to H.264 fMP4 and broadcasts to clients."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps

        self._proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()

        # Broadcast state
        self._clients: list[queue.Queue] = []
        self._clients_lock = threading.Lock()

        # The ftyp+moov boxes emitted once at the start
        self.init_segment: bytes | None = None
        self._init_ready = threading.Event()

    def start(self) -> None:
        """Spawn ffmpeg and start the reader thread."""
        if self._running:
            return

        self._running = True
        self.init_segment = None
        self._init_ready.clear()

        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}",
                "-r", str(self.fps),
                "-i", "pipe:0",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-profile:v", "baseline",
                "-level", "3.0",
                "-g", str(self.fps),
                "-keyint_min", str(self.fps),
                "-pix_fmt", "yuv420p",
                "-f", "mp4",
                "-movflags", "+frag_every_frame+empty_moov+default_base_moof",
                "-flush_packets", "1",
                "-loglevel", "error",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        logger.info("FMP4Encoder started (%dx%d @ %dfps)", self.width, self.height, self.fps)

    def stop(self) -> None:
        """Stop ffmpeg and reader thread."""
        self._running = False

        if self._proc:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

        if self._reader_thread:
            self._reader_thread.join(timeout=5)
            self._reader_thread = None

        # Drain all client queues with sentinel
        with self._clients_lock:
            for q in self._clients:
                try:
                    q.put_nowait(None)
                except queue.Full:
                    pass
            self._clients.clear()

        self._init_ready.set()  # unblock anyone waiting
        logger.info("FMP4Encoder stopped")

    def feed_frame(self, frame) -> None:
        """Write a raw BGR24 frame to ffmpeg stdin."""
        if not self._running or self._proc is None or self._proc.stdin is None:
            return
        try:
            self._proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            logger.warning("FMP4Encoder: ffmpeg stdin pipe broken")
            self._running = False

    def register_client(self) -> queue.Queue:
        """Register a new client and return its segment queue."""
        q = queue.Queue(maxsize=120)
        with self._clients_lock:
            self._clients.append(q)
        logger.debug("FMP4Encoder: client registered (%d total)", len(self._clients))
        return q

    def unregister_client(self, q: queue.Queue) -> None:
        """Remove a client queue."""
        with self._clients_lock:
            try:
                self._clients.remove(q)
            except ValueError:
                pass
        logger.debug("FMP4Encoder: client unregistered (%d total)", len(self._clients))

    def wait_for_init(self, timeout: float = 10.0) -> bool:
        """Block until the init segment is available."""
        return self._init_ready.wait(timeout=timeout)

    # ------------------------------------------------------------------
    # Internal: read ffmpeg stdout, parse MP4 boxes, broadcast segments
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        """Read ffmpeg stdout, parse MP4 boxes, broadcast to clients."""
        stdout = self._proc.stdout
        init_parts = bytearray()
        moof_buf = bytearray()
        collecting_moof = False

        while self._running and self._proc and self._proc.poll() is None:
            box = self._read_box(stdout)
            if box is None:
                break

            box_type, box_data = box

            # Init segment: ftyp + moov (emitted once at the start)
            if box_type in (b"ftyp", b"moov"):
                init_parts.extend(box_data)
                if box_type == b"moov":
                    self.init_segment = bytes(init_parts)
                    self._init_ready.set()
                    logger.info(
                        "FMP4Encoder: init segment captured (%d bytes)",
                        len(self.init_segment),
                    )
                continue

            # Media segments: moof + mdat pair
            if box_type == b"moof":
                moof_buf = bytearray(box_data)
                collecting_moof = True
                continue

            if box_type == b"mdat" and collecting_moof:
                moof_buf.extend(box_data)
                segment = bytes(moof_buf)
                moof_buf = bytearray()
                collecting_moof = False
                self._broadcast(segment)
                continue

            # Other boxes: skip (shouldn't happen with fMP4 but just in case)

        logger.debug("FMP4Encoder reader loop exiting")

    @staticmethod
    def _read_box(stream) -> tuple[bytes, bytes] | None:
        """
        Read one MP4 box from the stream.

        Returns (box_type, full_box_bytes) or None on EOF.
        Full box bytes include the 8-byte header.
        """
        header = b""
        while len(header) < 8:
            chunk = stream.read(8 - len(header))
            if not chunk:
                return None
            header += chunk

        size = struct.unpack(">I", header[:4])[0]
        box_type = header[4:8]

        if size == 1:
            # 64-bit extended size
            ext = b""
            while len(ext) < 8:
                chunk = stream.read(8 - len(ext))
                if not chunk:
                    return None
                ext += chunk
            size = struct.unpack(">Q", ext)[0]
            remaining = size - 16
            body = header + ext
        elif size == 0:
            # Box extends to EOF — read everything
            body = header + stream.read()
            return (box_type, body)
        else:
            remaining = size - 8
            body = header

        data = bytearray()
        while len(data) < remaining:
            chunk = stream.read(remaining - len(data))
            if not chunk:
                return None
            data += chunk

        return (box_type, body + bytes(data))

    def _broadcast(self, segment: bytes) -> None:
        """Send a media segment to all registered client queues."""
        with self._clients_lock:
            for q in self._clients:
                try:
                    q.put_nowait(segment)
                except queue.Full:
                    # Client is too slow, drop oldest segment and push new one
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        q.put_nowait(segment)
                    except queue.Full:
                        pass

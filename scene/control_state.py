import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class IngestRequest:
    snapshot_dir: str
    request_id: str
    session_id: Optional[str] = None
    image_name: Optional[str] = None


class ControlState:
    """Thread-safe shared state between the trainer loop and the HTTP server."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

        # Ingest queue: list[IngestRequest]
        self._ingest_queue: list = []

        # Per-request lifecycle: request_id -> state string
        self._request_states: Dict[str, str] = {}

        # Idempotency ledger: (session_id, image_name) -> request_id.
        # The upstream server may retry an ingest POST; a duplicate must be
        # a no-op that reports the original request's id.
        self._seen_images: Dict[Tuple[str, str], str] = {}

        # Session pinning: this trainer process serves one capture session.
        # Set on the first session-tagged ingest; later mismatches are
        # rejected by the HTTP layer with 409.
        self.session_id: Optional[str] = None

        # Pending checkpoint flag; trainer sets _last_snapshot_path and clears this
        self._checkpoint_requested: bool = False
        self._checkpoint_done: bool = False
        self._last_snapshot_path: Optional[str] = None

        # GPU-serialization handshake (home-server pause/resume). The HTTP
        # thread requests a pause; the trainer loop moves its model to CPU at
        # a safe point, marks itself paused, parks until resume is requested,
        # then moves back to GPU. `_vram_mb_paused` is the post-move VRAM the
        # /pause reply reports.
        self._pause_requested: bool = False
        self._resume_requested: bool = False
        self._paused: bool = False
        self._vram_mb_paused: Optional[float] = None

        # Status mirror updated by the trainer each iteration
        self.gaussian_count: int = 0
        self.monitor_state: str = "initializing"
        self.queue_depth: int = 0
        self.n_images: int = 0
        self.bootstrap_complete: bool = False
        self.iter: int = 0
        self.last_ingest: Optional[float] = None  # epoch seconds of last accepted ingest

    # ------------------------------------------------------------------
    # Ingest queue
    # ------------------------------------------------------------------

    def enqueue_ingest(self, snapshot_dir: str, request_id: Optional[str] = None,
                       session_id: Optional[str] = None,
                       image_name: Optional[str] = None) -> Tuple[str, bool]:
        """Enqueue a snapshot for integration.

        Returns (request_id, duplicate). When (session_id, image_name) has
        been seen before, nothing is enqueued and the original request_id is
        returned with duplicate=True.
        """
        with self._cond:
            if session_id and image_name:
                key = (session_id, image_name)
                existing = self._seen_images.get(key)
                if existing is not None:
                    return existing, True

            if not request_id:
                request_id = str(uuid.uuid4())
            if session_id and image_name:
                self._seen_images[(session_id, image_name)] = request_id
            if session_id and self.session_id is None:
                self.session_id = session_id

            self._ingest_queue.append(
                IngestRequest(snapshot_dir, request_id, session_id, image_name))
            self._request_states[request_id] = "queued"
            self.queue_depth = len(self._ingest_queue)
            self.last_ingest = time.time()
            self._cond.notify_all()
        return request_id, False

    def ledger_state(self) -> dict:
        """Serializable idempotency ledger for checkpoint/resume, so a
        retried ingest POST is still recognised after a process restart."""
        with self._lock:
            return {
                "session_id": self.session_id,
                "seen_images": [
                    [sid, name, rid]
                    for (sid, name), rid in self._seen_images.items()
                ],
            }

    def restore_ledger(self, state: dict):
        with self._cond:
            self.session_id = state.get("session_id")
            self._seen_images = {
                (sid, name): rid
                for sid, name, rid in state.get("seen_images", [])
            }

    def drain_ingest_queue(self) -> list:
        with self._cond:
            items = list(self._ingest_queue)
            self._ingest_queue.clear()
            self.queue_depth = 0
            return items

    def set_request_state(self, request_id: str, state: str):
        with self._cond:
            self._request_states[request_id] = state
            self._cond.notify_all()

    def get_request_state(self, request_id: str) -> str:
        with self._lock:
            return self._request_states.get(request_id, "unknown")

    # ------------------------------------------------------------------
    # Checkpoint handshake
    # ------------------------------------------------------------------

    def request_checkpoint(self):
        with self._cond:
            self._checkpoint_requested = True
            self._checkpoint_done = False
            self._cond.notify_all()

    def checkpoint_pending(self) -> bool:
        with self._lock:
            return self._checkpoint_requested

    def complete_checkpoint(self, path: str):
        with self._cond:
            self._last_snapshot_path = path
            self._checkpoint_requested = False
            self._checkpoint_done = True
            self._cond.notify_all()

    def wait_for_checkpoint(self, timeout: float = 30.0) -> Optional[str]:
        with self._cond:
            self._cond.wait_for(lambda: self._checkpoint_done, timeout=timeout)
            if self._checkpoint_done:
                self._checkpoint_done = False
                return self._last_snapshot_path
            return None

    # ------------------------------------------------------------------
    # GPU-serialization handshake (pause / resume)
    # ------------------------------------------------------------------

    def request_pause(self):
        """HTTP thread: ask the trainer loop to move its model to CPU."""
        with self._cond:
            self._pause_requested = True
            self._resume_requested = False
            self._cond.notify_all()

    def request_resume(self):
        """HTTP thread: ask the trainer loop to move its model back to GPU."""
        with self._cond:
            self._resume_requested = True
            self._pause_requested = False
            self._cond.notify_all()

    def pause_requested(self) -> bool:
        """Trainer loop: a pause is pending and we're not already paused."""
        with self._lock:
            return self._pause_requested and not self._paused

    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def current_vram_mb(self) -> Optional[float]:
        with self._lock:
            return self._vram_mb_paused

    def mark_paused(self, vram_mb: Optional[float]):
        """Trainer loop: the model is now on CPU; VRAM freed. Wakes /pause."""
        with self._cond:
            self._paused = True
            self._pause_requested = False
            self._vram_mb_paused = vram_mb
            self._cond.notify_all()

    def wait_until_paused(self, timeout: float = 30.0) -> bool:
        """HTTP /pause: block until the trainer acks the model is on CPU."""
        with self._cond:
            self._cond.wait_for(lambda: self._paused, timeout=timeout)
            return self._paused

    def wait_for_resume(self, timeout: float = 1.0) -> bool:
        """Trainer loop (while paused): block until resume is requested.

        Polled with a finite timeout so a SIGINT still lands promptly on the
        main thread. Returns True once resume has been requested."""
        with self._cond:
            self._cond.wait_for(lambda: self._resume_requested, timeout=timeout)
            return self._resume_requested

    def mark_resumed(self):
        """Trainer loop: the model is back on GPU. Wakes /resume."""
        with self._cond:
            self._paused = False
            self._resume_requested = False
            self._vram_mb_paused = None
            self._cond.notify_all()

    def wait_until_resumed(self, timeout: float = 30.0) -> bool:
        """HTTP /resume: block until the trainer has left the paused state."""
        with self._cond:
            self._cond.wait_for(lambda: not self._paused, timeout=timeout)
            return not self._paused

    # ------------------------------------------------------------------
    # Idle / wake
    # ------------------------------------------------------------------

    def wait_for_work(self, timeout: float = 1.0):
        """Block until ingest arrives, checkpoint, or a pause is requested."""
        with self._cond:
            self._cond.wait_for(
                lambda: (bool(self._ingest_queue) or self._checkpoint_requested
                         or self._pause_requested),
                timeout=timeout,
            )

    # ------------------------------------------------------------------
    # Status snapshot (called by trainer each iter, no lock needed for reads)
    # ------------------------------------------------------------------

    def update_status(self, gaussian_count: int, monitor_state: str,
                      n_images: int, bootstrap_complete: bool,
                      iteration: Optional[int] = None):
        with self._lock:
            self.gaussian_count = gaussian_count
            self.monitor_state = monitor_state
            self.n_images = n_images
            self.bootstrap_complete = bootstrap_complete
            self.queue_depth = len(self._ingest_queue)
            if iteration is not None:
                self.iter = iteration

    def status_snapshot(self) -> dict:
        with self._lock:
            return {
                "gaussian_count": self.gaussian_count,
                "monitor_state": self.monitor_state,
                "last_snapshot": self._last_snapshot_path,
                "queue_depth": self.queue_depth,
                "n_images": self.n_images,
                "bootstrap_complete": self.bootstrap_complete,
                "session_id": self.session_id,
                "iter": self.iter,
                "last_ingest": self.last_ingest,
                "paused": self._paused,
            }

    def health_snapshot(self) -> dict:
        """Health payload per the home-server contract (app/api/trainer.py):
        keys state, iter, splat_count, queue_depth, last_ingest, vram_mb,
        session_id, n_images, bootstrap_complete, paused."""
        with self._lock:
            return {
                "state": self.monitor_state,
                "iter": self.iter,
                "splat_count": self.gaussian_count,
                "queue_depth": len(self._ingest_queue),
                "last_ingest": self.last_ingest,
                "vram_mb": _vram_mb(),
                "session_id": self.session_id,
                "n_images": self.n_images,
                "bootstrap_complete": self.bootstrap_complete,
                "paused": self._paused,
            }


def _vram_mb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated() / (1024 ** 2), 1)
    except Exception:
        pass
    return None

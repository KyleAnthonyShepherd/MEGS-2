import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class IngestRequest:
    snapshot_dir: str
    request_id: str


class ControlState:
    """Thread-safe shared state between the trainer loop and the HTTP server."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

        # Ingest queue: list[IngestRequest]
        self._ingest_queue: list = []

        # Per-request lifecycle: request_id -> state string
        self._request_states: Dict[str, str] = {}

        # Pending checkpoint flag; trainer sets _last_snapshot_path and clears this
        self._checkpoint_requested: bool = False
        self._checkpoint_done: bool = False
        self._last_snapshot_path: Optional[str] = None

        # Status mirror updated by the trainer each iteration
        self.gaussian_count: int = 0
        self.monitor_state: str = "initializing"
        self.queue_depth: int = 0
        self.n_images: int = 0
        self.bootstrap_complete: bool = False

    # ------------------------------------------------------------------
    # Ingest queue
    # ------------------------------------------------------------------

    def enqueue_ingest(self, snapshot_dir: str, request_id: Optional[str] = None) -> str:
        if not request_id:
            request_id = str(uuid.uuid4())
        with self._cond:
            self._ingest_queue.append(IngestRequest(snapshot_dir, request_id))
            self._request_states[request_id] = "queued"
            self.queue_depth = len(self._ingest_queue)
            self._cond.notify_all()
        return request_id

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
    # Idle / wake
    # ------------------------------------------------------------------

    def wait_for_work(self, timeout: float = 1.0):
        """Block until ingest arrives or checkpoint is requested."""
        with self._cond:
            self._cond.wait_for(
                lambda: bool(self._ingest_queue) or self._checkpoint_requested,
                timeout=timeout,
            )

    # ------------------------------------------------------------------
    # Status snapshot (called by trainer each iter, no lock needed for reads)
    # ------------------------------------------------------------------

    def update_status(self, gaussian_count: int, monitor_state: str,
                      n_images: int, bootstrap_complete: bool):
        with self._lock:
            self.gaussian_count = gaussian_count
            self.monitor_state = monitor_state
            self.n_images = n_images
            self.bootstrap_complete = bootstrap_complete
            self.queue_depth = len(self._ingest_queue)

    def status_snapshot(self) -> dict:
        with self._lock:
            return {
                "gaussian_count": self.gaussian_count,
                "monitor_state": self.monitor_state,
                "last_snapshot": self._last_snapshot_path,
                "queue_depth": self.queue_depth,
                "n_images": self.n_images,
                "bootstrap_complete": self.bootstrap_complete,
            }

"""Structured JSONL event log for the continuous trainer.

One JSON object per line, written append-only and flushed per event so the
file is readable while training runs (and after a crash). This is both the
operator's debugging record for misbehaving field sessions and the input to
the regression harness's trigger-sequence golden comparison.

Event names used by continuous_train.py:
    ingest, densify, fast_prune, lightweight_prune, cull_sg_axes,
    converged, checkpoint, resume, snapshot
"""

import json
import time
from pathlib import Path
from typing import List


class EventLog:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a")

    def emit(self, event: str, **fields):
        rec = {"t": round(time.time(), 3), "event": event}
        rec.update(fields)
        self._fh.write(json.dumps(rec, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_events(path: str) -> List[dict]:
    """Read a JSONL event log, skipping any torn trailing line."""
    out = []
    p = Path(path)
    if not p.exists():
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn write at crash — ignore
    return out


def trigger_sequence(events: List[dict],
                     kinds=("densify", "fast_prune", "lightweight_prune",
                            "cull_sg_axes")) -> List[str]:
    """Reduce an event stream to the ordered list of trigger firings.

    Returns entries like "densify@stalled" — the trigger name plus the
    monitor state it fired in. Iteration numbers are deliberately excluded:
    they shift with harmless timing changes, while the ordered kind+state
    sequence is the scheduler's behavioural fingerprint.
    """
    seq = []
    for e in events:
        if e.get("event") in kinds:
            seq.append(f"{e['event']}@{e.get('state', '?')}")
    return seq

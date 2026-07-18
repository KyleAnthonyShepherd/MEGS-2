"""Tests for scene/event_log.py."""

from tests._direct_import import load_scene_module

event_log = load_scene_module("event_log")
EventLog = event_log.EventLog
read_events = event_log.read_events
trigger_sequence = event_log.trigger_sequence


def test_emit_read_round_trip(tmp_path):
    p = tmp_path / "events.jsonl"
    with EventLog(str(p)) as log:
        log.emit("ingest", iter=0, n_images=1)
        log.emit("densify", iter=10, state="stalled", n_before=5, n_after=9)
    events = read_events(str(p))
    assert [e["event"] for e in events] == ["ingest", "densify"]
    assert events[1]["state"] == "stalled"
    assert all("t" in e for e in events)


def test_append_across_instances(tmp_path):
    """Reopening (as after a crash/resume) appends rather than truncates."""
    p = tmp_path / "events.jsonl"
    with EventLog(str(p)) as log:
        log.emit("ingest", iter=0)
    with EventLog(str(p)) as log:
        log.emit("resume", iter=100)
    assert [e["event"] for e in read_events(str(p))] == ["ingest", "resume"]


def test_torn_trailing_line_skipped(tmp_path):
    p = tmp_path / "events.jsonl"
    with EventLog(str(p)) as log:
        log.emit("ingest", iter=0)
    with open(p, "a") as f:
        f.write('{"t": 1, "event": "densi')  # crash mid-write
    events = read_events(str(p))
    assert len(events) == 1


def test_read_missing_file_is_empty(tmp_path):
    assert read_events(str(tmp_path / "nope.jsonl")) == []


def test_trigger_sequence_extraction(tmp_path):
    p = tmp_path / "events.jsonl"
    with EventLog(str(p)) as log:
        log.emit("ingest", iter=0)
        log.emit("densify", iter=10, state="wants_capacity")
        log.emit("train_state", iter=500)
        log.emit("fast_prune", iter=600, state="stalled")
        log.emit("converged", iter=700)
        log.emit("cull_sg_axes", iter=700, state="converged")
    seq = trigger_sequence(read_events(str(p)))
    assert seq == ["densify@wants_capacity", "fast_prune@stalled",
                   "cull_sg_axes@converged"]

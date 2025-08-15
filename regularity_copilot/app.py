from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from flask import Flask, render_template, request, jsonify


app = Flask(__name__)


@dataclass
class Segment:
    name: str
    distance_km: float
    speed_kmh: float


@dataclass
class EventState:
    start_timestamp_ms: Optional[int]
    segments: List[Segment]
    odo_factor: float
    correction_horizon_km: float

    def total_distance_km(self) -> float:
        return sum(s.distance_km for s in self.segments)


# In-memory state. In production, replace with persistent storage.
event_state = EventState(
    start_timestamp_ms=None,
    segments=[
        Segment(name="S1", distance_km=5.0, speed_kmh=48.0),
        Segment(name="S2", distance_km=7.2, speed_kmh=52.0),
        Segment(name="S3", distance_km=3.8, speed_kmh=45.0),
    ],
    odo_factor=1.0,  # calibration factor (official km = recorded km * factor)
    correction_horizon_km=1.0,  # over how many km to smooth the correction
)


def compute_elapsed_hours_for_distance(distance_km: float, segments: List[Segment]) -> float:
    """
    Returns ideal elapsed time in hours from start to the given official distance.
    If distance exceeds total, clamps at total.
    """
    remaining = max(0.0, distance_km)
    elapsed_h = 0.0
    for seg in segments:
        if remaining <= 0:
            break
        take_km = min(remaining, seg.distance_km)
        if seg.speed_kmh <= 0:
            # Avoid division by zero; treat as infinite time
            return float("inf")
        elapsed_h += take_km / seg.speed_kmh
        remaining -= take_km
    return elapsed_h


def compute_segment_cursor(distance_km: float, segments: List[Segment]) -> Dict[str, Any]:
    """
    Returns details of where the given distance sits within the segment list.
    """
    cumulative = 0.0
    for idx, seg in enumerate(segments):
        next_cum = cumulative + seg.distance_km
        if distance_km <= next_cum:
            return {
                "index": idx,
                "name": seg.name,
                "distance_into_segment_km": max(0.0, distance_km - cumulative),
                "segment": seg,
                "remaining_in_segment_km": max(0.0, next_cum - distance_km),
                "cumulative_before_km": cumulative,
            }
        cumulative = next_cum
    # Beyond last segment
    if segments:
        last_idx = len(segments) - 1
        last = segments[last_idx]
        return {
            "index": last_idx,
            "name": last.name,
            "distance_into_segment_km": last.distance_km,
            "segment": last,
            "remaining_in_segment_km": 0.0,
            "cumulative_before_km": cumulative - last.distance_km,
        }
    return {
        "index": -1,
        "name": "",
        "distance_into_segment_km": 0.0,
        "segment": Segment(name="", distance_km=0.0, speed_kmh=0.0),
        "remaining_in_segment_km": 0.0,
        "cumulative_before_km": 0.0,
    }


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def compute_recommended_speed(
    delta_seconds: float,
    current_segment_speed_kmh: float,
    correction_horizon_km: float,
    min_speed_kmh: float = 0.0,
    max_speed_kmh: float = 150.0,
) -> float:
    """
    Uses a first-order sensitivity model: t = D/v, dt/dv = -D/v^2.
    For desired time change delta_t (seconds), dV ≈ -(v^2/D) * delta_t_hours.
    """
    v = max(0.1, current_segment_speed_kmh)
    D = max(0.05, correction_horizon_km)
    delta_h = delta_seconds / 3600.0
    dV = -(v * v / D) * delta_h
    v_prime = clamp(v + dV, min_speed_kmh, max_speed_kmh)
    return v_prime


@app.route("/")
def index():
    state = {
        "start_timestamp_ms": event_state.start_timestamp_ms,
        "segments": [asdict(s) for s in event_state.segments],
        "odo_factor": event_state.odo_factor,
        "correction_horizon_km": event_state.correction_horizon_km,
        "total_distance_km": event_state.total_distance_km(),
    }
    return render_template("index.html", state=state)


@app.route("/configure", methods=["POST"])
def configure():
    data = request.get_json(silent=True) or {}
    try:
        start_timestamp_ms = data.get("start_timestamp_ms")
        segments_payload = data.get("segments") or []
        odo_factor = float(data.get("odo_factor", 1.0))
        correction_horizon_km = float(data.get("correction_horizon_km", 1.0))

        segments: List[Segment] = []
        for i, s in enumerate(segments_payload):
            name = str(s.get("name") or f"S{i+1}")
            distance_km = float(s.get("distance_km", 0.0))
            speed_kmh = float(s.get("speed_kmh", s.get("speed", 0.0)))
            if distance_km < 0 or speed_kmh <= 0:
                return jsonify({
                    "ok": False,
                    "error": f"Segment invalid: name={name}, distance_km={distance_km}, speed_kmh={speed_kmh}",
                }), 400
            segments.append(Segment(name=name, distance_km=distance_km, speed_kmh=speed_kmh))

        event_state.start_timestamp_ms = int(start_timestamp_ms) if start_timestamp_ms is not None else None
        event_state.segments = segments
        event_state.odo_factor = max(0.1, float(odo_factor))
        event_state.correction_horizon_km = max(0.1, float(correction_horizon_km))

        return jsonify({
            "ok": True,
            "state": {
                "start_timestamp_ms": event_state.start_timestamp_ms,
                "segments": [asdict(s) for s in event_state.segments],
                "odo_factor": event_state.odo_factor,
                "correction_horizon_km": event_state.correction_horizon_km,
                "total_distance_km": event_state.total_distance_km(),
            }
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/state", methods=["GET"])  # Useful for debugging / initial load
def get_state():
    return jsonify({
        "start_timestamp_ms": event_state.start_timestamp_ms,
        "segments": [asdict(s) for s in event_state.segments],
        "odo_factor": event_state.odo_factor,
        "correction_horizon_km": event_state.correction_horizon_km,
        "total_distance_km": event_state.total_distance_km(),
    })


@app.route("/api/status", methods=["GET"])  # core computation
def status():
    try:
        now_ms = float(request.args.get("now_ms", "0"))
        odo_km_raw = float(request.args.get("odo_km", "0"))
        start_timestamp_ms = event_state.start_timestamp_ms

        if start_timestamp_ms is None:
            return jsonify({"ok": False, "error": "Start time not set"}), 400

        official_km = odo_km_raw * event_state.odo_factor

        elapsed_ideal_hours = compute_elapsed_hours_for_distance(official_km, event_state.segments)
        if math.isinf(elapsed_ideal_hours):
            return jsonify({"ok": False, "error": "Invalid segment speed (0 km/h)"}), 400

        ideal_ms = start_timestamp_ms + int(elapsed_ideal_hours * 3600_000)
        delta_seconds = (now_ms - ideal_ms) / 1000.0

        cursor = compute_segment_cursor(official_km, event_state.segments)
        seg = cursor["segment"]
        remaining_total_km = max(0.0, event_state.total_distance_km() - official_km)
        horizon_km = min(event_state.correction_horizon_km, max(0.05, remaining_total_km))

        recommended_speed_kmh = compute_recommended_speed(
            delta_seconds=delta_seconds,
            current_segment_speed_kmh=seg.speed_kmh if seg.speed_kmh > 0 else 1.0,
            correction_horizon_km=horizon_km,
        )

        return jsonify({
            "ok": True,
            "now_ms": int(now_ms),
            "start_timestamp_ms": start_timestamp_ms,
            "official_km": round(official_km, 3),
            "ideal_timestamp_ms": int(ideal_ms),
            "delta_seconds": round(delta_seconds, 3),
            "current_segment_index": cursor["index"],
            "current_segment_name": cursor["name"],
            "distance_into_segment_km": round(cursor["distance_into_segment_km"], 3),
            "ideal_segment_speed_kmh": seg.speed_kmh,
            "correction_horizon_km": round(horizon_km, 3),
            "recommended_speed_kmh": round(recommended_speed_kmh, 2),
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
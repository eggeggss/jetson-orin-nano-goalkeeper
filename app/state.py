"""
state.py — 跨模組共享狀態（camera frame、enrollment 進度、即時監控）

main.py 和 routes.py 都從這裡 import，確保讀寫同一份物件。
"""
from typing import Optional
import asyncio
import numpy as np

# 最新一幀（供 /api/snapshot 和 /api/video 使用）
current_frame: Optional[np.ndarray] = None

# 帶辨識框的標注幀（供 MJPEG 串流使用）
annotated_frame: Optional[np.ndarray] = None

# Enrollment 進行中的狀態
enrollment_state: dict = {
    "active": False,
    "person_id": None,
    "person_name": "",
    "samples_needed": 5,
    "samples_captured": 0,
    "last_capture_time": 0.0,
    "valid_since": 0.0,
    "hold_progress": 0,
    "face_size_percent": 0,
    "face_fit_status": "idle",
    "last_status": "",
    "completed": False,
}

# 即時監控狀態（供 SSE 推送）
live_status: dict = {
    "phase": "idle",          # idle | recognition_off | enrollment | enrollment_done | detecting | conversation | completed | warning
    "person_name": "",
    "department": "",
    "confidence": 0.0,
    "transcript": [],         # list of {"role": "ai"|"user", "text": str}
    "event_id": None,
    "faces": [],              # list of current detected faces (bbox + name)
    "purpose_summary": "",
    "completed_at": "",
    "cooldown_remaining": 0,
    "completion_status": "",
    "reset_generation": 0,
    "camera_off": False,
    "recognition_enabled": True,
}

# Incremented by the kiosk reset button to cancel any active conversation and
# clear recognition cooldowns in the camera loop.
reset_generation: int = 0
recognition_paused_until: float = 0.0
recognition_enabled: bool = True
camera_available: bool = False

# SSE 廣播 queue（routes 訂閱，state 寫入）
_sse_queues: list[asyncio.Queue] = []


def push_event(payload: dict):
    """Push a live_status snapshot to all connected SSE clients."""
    import copy, json
    # Serialize through JSON to strip any numpy types before queuing
    snapshot = json.loads(json.dumps(copy.deepcopy(payload), default=lambda o: float(o) if hasattr(o, '__float__') else str(o)))
    for q in _sse_queues:
        try:
            q.put_nowait(snapshot)
        except asyncio.QueueFull:
            pass


def subscribe() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=20)
    _sse_queues.append(q)
    return q


def unsubscribe(q: asyncio.Queue):
    try:
        _sse_queues.remove(q)
    except ValueError:
        pass

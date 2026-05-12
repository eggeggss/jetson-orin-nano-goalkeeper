from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from typing import Optional, AsyncGenerator
import os
import cv2
import json
import asyncio
import numpy as np

from database import get_db
from face.matcher import matcher
import state


def _safe_json(obj):
    """Serialize live_status safely, converting numpy types to Python natives."""
    return json.loads(json.dumps(obj, default=lambda o: float(o) if hasattr(o, '__float__') else str(o)))

router = APIRouter()
_TMPL = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


# ── Dashboard ─────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    conn = get_db()
    events = conn.execute("""
        SELECT id, person_name, confidence, status, summary, created_at
        FROM events ORDER BY created_at DESC LIMIT 100
    """).fetchall()
    persons = conn.execute(
        "SELECT id, name, department, is_active FROM persons ORDER BY name"
    ).fetchall()
    conn.close()
    return _TMPL.TemplateResponse("dashboard.html", {
        "request": request,
        "events": [dict(e) for e in events],
        "persons": [dict(p) for p in persons],
    })


# ── Events API ────────────────────────────────────────────────────────────────

@router.get("/api/events")
async def list_events(limit: int = 50, person_id: Optional[int] = None):
    conn = get_db()
    if person_id:
        rows = conn.execute(
            "SELECT * FROM events WHERE person_id=? ORDER BY created_at DESC LIMIT ?",
            (person_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM events ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.get("/api/events/{event_id}")
async def get_event(event_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Event not found")
    return dict(row)


# ── Persons API ───────────────────────────────────────────────────────────────

@router.get("/api/persons")
async def list_persons():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, name, department, employee_id, is_active, created_at FROM persons ORDER BY name"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.post("/api/persons", status_code=201)
async def create_person(data: dict):
    name = data.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    conn = get_db()
    conn.execute(
        "INSERT INTO persons (name, department, employee_id) VALUES (?, ?, ?)",
        (name, data.get("department", ""), data.get("employee_id", "")),
    )
    conn.commit()
    pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    matcher.reload()
    return {"id": pid, "message": "Person created"}


@router.delete("/api/persons/{person_id}")
async def deactivate_person(person_id: int):
    conn = get_db()
    conn.execute("UPDATE persons SET is_active=0 WHERE id=?", (person_id,))
    conn.commit()
    conn.close()
    matcher.reload()
    return {"message": "Person deactivated"}


@router.put("/api/persons/{person_id}")
async def update_person(person_id: int, data: dict):
    name = data.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    conn = get_db()
    row = conn.execute("SELECT id FROM persons WHERE id=?", (person_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Person not found")
    conn.execute(
        "UPDATE persons SET name=?, department=?, employee_id=?, is_active=? WHERE id=?",
        (name, data.get("department", ""), data.get("employee_id", ""),
         1 if data.get("is_active", True) else 0, person_id),
    )
    conn.commit()
    conn.close()
    matcher.reload()
    return {"message": "Person updated"}


@router.delete("/api/persons/{person_id}/hard")
async def delete_person(person_id: int):
    """Permanently delete person and all their face embeddings."""
    conn = get_db()
    conn.execute("DELETE FROM face_embeddings WHERE person_id=?", (person_id,))
    conn.execute("DELETE FROM persons WHERE id=?", (person_id,))
    conn.commit()
    conn.close()
    matcher.reload()
    return {"message": "Person permanently deleted"}


# ── System API ────────────────────────────────────────────────────────────────

@router.get("/api/status")
async def system_status():
    from face.detector import detector
    conn = get_db()
    event_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    conn.close()
    return {
        "model_loaded": detector.app is not None,
        "persons_loaded": len(matcher.persons),
        "event_count": event_count,
        "recognition_enabled": state.recognition_enabled,
    }


@router.post("/api/reload-embeddings")
async def reload_embeddings():
    matcher.reload()
    return {"persons_loaded": len(matcher.persons)}


# ── Camera snapshot ───────────────────────────────────────────────────────────

@router.get("/api/live/status")
async def live_status():
    return JSONResponse(content=_safe_json(state.live_status))


@router.post("/api/live/reset")
async def reset_live_flow():
    state.reset_generation += 1
    state.enrollment_state["active"] = False
    state.enrollment_state["completed"] = False
    state.enrollment_state["valid_since"] = 0.0
    state.enrollment_state["hold_progress"] = 0
    state.enrollment_state["face_fit_status"] = "idle"
    state.camera_available = False
    state.current_frame = None
    state.annotated_frame = None
    state.live_status.update({
        "phase": "idle" if state.recognition_enabled else "recognition_off",
        "person_name": "",
        "department": "",
        "confidence": 0.0,
        "transcript": [],
        "event_id": None,
        "faces": [],
        "purpose_summary": "",
        "completed_at": "",
        "cooldown_remaining": 0,
        "completion_status": "",
        "reset_generation": state.reset_generation,
        "camera_off": True if state.recognition_enabled else False,
        "recognition_enabled": state.recognition_enabled,
    })
    state.push_event(state.live_status)
    return {"message": "Flow reset", "reset_generation": state.reset_generation}


@router.post("/api/recognition/start")
async def start_recognition():
    state.recognition_enabled = True
    state.reset_generation += 1
    state.live_status.update({
        "phase": "idle" if state.recognition_enabled else "recognition_off",
        "person_name": "",
        "department": "",
        "confidence": 0.0,
        "transcript": [],
        "event_id": None,
        "faces": [],
        "purpose_summary": "",
        "completed_at": "",
        "cooldown_remaining": 0,
        "completion_status": "",
        "reset_generation": state.reset_generation,
        "camera_off": False,
        "recognition_enabled": True,
    })
    state.push_event(state.live_status)
    return {"message": "Recognition started", "reset_generation": state.reset_generation}


@router.post("/api/recognition/stop")
async def stop_recognition():
    state.recognition_enabled = False
    state.reset_generation += 1
    state.enrollment_state["active"] = False
    state.enrollment_state["completed"] = False
    state.live_status.update({
        "phase": "recognition_off",
        "person_name": "",
        "department": "",
        "confidence": 0.0,
        "transcript": [],
        "event_id": None,
        "faces": [],
        "purpose_summary": "",
        "completed_at": "",
        "cooldown_remaining": 0,
        "completion_status": "",
        "reset_generation": state.reset_generation,
        "camera_off": False,
        "recognition_enabled": False,
    })
    state.push_event(state.live_status)
    return {"message": "Recognition stopped", "reset_generation": state.reset_generation}


@router.get("/api/live/stream")
async def live_stream(request: Request):
    """Server-Sent Events stream for real-time face recognition & conversation updates."""
    q = state.subscribe()

    async def generator() -> AsyncGenerator[str, None]:
        try:
            # Send current state immediately on connect
            yield f"data: {json.dumps(_safe_json(state.live_status))}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"  # prevent proxy timeout
        finally:
            state.unsubscribe(q)

    return StreamingResponse(generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/api/snapshot")
async def snapshot():
    frame = state.current_frame
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not ready")
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


@router.get("/api/video")
async def video_feed():
    """MJPEG stream with face-recognition bounding box overlays."""
    async def generate():
        while True:
            if state.live_status.get("camera_off"):
                frame = None
            else:
                frame = state.annotated_frame if state.annotated_frame is not None \
                        else state.current_frame
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            if frame is not None:
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       jpeg.tobytes() + b"\r\n")
            await asyncio.sleep(0.05)  # ~20 fps max

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/kiosk", response_class=HTMLResponse)
async def kiosk(request: Request):
    response = _TMPL.TemplateResponse("kiosk.html", {"request": request})
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


# ── Enrollment API ────────────────────────────────────────────────────────────

@router.post("/api/enroll/start", status_code=201)
async def start_enrollment(data: dict):
    name = data.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    if state.enrollment_state.get("active"):
        raise HTTPException(status_code=409, detail="Enrollment already in progress")

    conn = get_db()
    conn.execute(
        "INSERT INTO persons (name, department, employee_id) VALUES (?, ?, ?)",
        (name, data.get("department", ""), data.get("employee_id", "")),
    )
    conn.commit()
    pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()

    state.enrollment_state.update({
        "active": True,
        "person_id": pid,
        "person_name": name,
        "samples_needed": int(data.get("samples", 5)),
        "samples_captured": 0,
        "last_capture_time": 0.0,
        "valid_since": 0.0,
        "hold_progress": 0,
        "face_size_percent": 0,
        "face_fit_status": "waiting",
        "last_status": "準備擷取，請將臉靠近鏡頭並對準方框 …",
        "completed": False,
    })
    state.live_status.update({
        "phase": "enrollment",
        "person_name": "",
        "department": "",
        "confidence": 0.0,
        "transcript": [],
        "event_id": None,
        "faces": [],
        "purpose_summary": "",
        "completed_at": "",
        "cooldown_remaining": 0,
        "completion_status": "",
        "reset_generation": state.reset_generation,
        "camera_off": False,
        "recognition_enabled": state.recognition_enabled,
    })
    state.push_event(state.live_status)
    return {"person_id": pid, "message": "Enrollment started"}


@router.get("/api/enroll/status")
async def enrollment_status():
    return state.enrollment_state


@router.post("/api/enroll/cancel")
async def cancel_enrollment():
    state.enrollment_state["active"] = False
    state.enrollment_state["last_status"] = "已取消"
    state.enrollment_state["completed"] = False
    state.live_status.update({
        "phase": "idle" if state.recognition_enabled else "recognition_off",
        "person_name": "",
        "department": "",
        "confidence": 0.0,
        "transcript": [],
        "event_id": None,
        "faces": [],
        "purpose_summary": "",
        "completed_at": "",
        "cooldown_remaining": 0,
        "completion_status": "",
        "reset_generation": state.reset_generation,
        "camera_off": False,
        "recognition_enabled": state.recognition_enabled,
    })
    state.push_event(state.live_status)
    return {"message": "Cancelled"}


@router.post("/api/enroll/finish")
async def finish_enrollment():
    state.enrollment_state.update({
        "active": False,
        "completed": False,
        "valid_since": 0.0,
        "hold_progress": 0,
        "face_fit_status": "idle",
        "last_status": "已存檔",
    })
    state.live_status.update({
        "phase": "idle",
        "person_name": "",
        "department": "",
        "confidence": 0.0,
        "transcript": [],
        "event_id": None,
        "faces": [],
        "purpose_summary": "",
        "completed_at": "",
        "cooldown_remaining": 0,
        "completion_status": "",
        "reset_generation": state.reset_generation,
        "camera_off": False,
        "recognition_enabled": state.recognition_enabled,
    })
    state.push_event(state.live_status)
    return {"message": "Enrollment saved"}

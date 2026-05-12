"""
main.py — 備品室門禁系統入口

啟動三個並行 asyncio 任務：
  1. FastAPI dashboard (port 8000)
  2. Camera loop — 偵測人臉 → 比對身份 → 觸發語音對話
  3. 背景清理任務 — 每小時清理過期事件
"""

import asyncio
import cv2
import httpx
import logging
import os
import sounddevice as sd
import time
from datetime import datetime

import numpy as np
import uvicorn
from PIL import Image, ImageDraw, ImageFont

from config import settings
from database import cleanup_old_events, get_db, init_db
from face.detector import detector
from face.matcher import matcher
import state

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")

# ── Global state ──────────────────────────────────────────────────────────────
_conversation_lock = asyncio.Lock()
_last_event: dict[str | None, float] = {}   # person_id (or None) → last event timestamp


def _clear_camera_frames():
    state.current_frame = None
    state.annotated_frame = None
    state.live_status["faces"] = []


def _mark_camera_unavailable(reason: str):
    was_available = state.camera_available
    state.camera_available = False
    _clear_camera_frames()
    if was_available or not state.live_status.get("camera_off"):
        logger.warning("Camera unavailable: %s", reason)
        state.live_status.update({
            "camera_off": True,
            "reset_generation": state.reset_generation,
        })
        state.push_event(state.live_status)


def _mark_camera_ready():
    was_available = state.camera_available
    state.camera_available = True
    if not was_available or state.live_status.get("camera_off"):
        logger.info("Camera stream has fresh frames again.")
        if state.live_status.get("phase") not in {"completed", "warning"}:
            state.live_status.update({
                "camera_off": False,
                "reset_generation": state.reset_generation,
            })
            state.push_event(state.live_status)

# ── CJK-capable bounding box drawing ─────────────────────────────────────────

_cjk_font = None

def _get_cjk_font(size: int = 20) -> ImageFont.FreeTypeFont:
    global _cjk_font
    if _cjk_font is None:
        for path in [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]:
            if os.path.exists(path):
                _cjk_font = ImageFont.truetype(path, size)
                break
        else:
            _cjk_font = ImageFont.load_default()
    return _cjk_font


def _draw_boxes(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw face bounding boxes with CJK-compatible labels using Pillow."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = _get_cjk_font(20)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        is_known = det["is_known"]
        label = f"{det['name']} {det['confidence']:.0%}" if is_known else "未知"
        box_color = (0, 230, 0) if is_known else (255, 100, 0)  # green / orange (RGB)

        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)

        # Label background
        try:
            bbox_t = font.getbbox(label)
            tw, th = bbox_t[2] - bbox_t[0], bbox_t[3] - bbox_t[1]
        except AttributeError:
            tw, th = len(label) * 8, 14  # fallback for default font
        pad = 4
        draw.rectangle([x1, max(y1 - th - pad * 2, 0), x1 + tw + pad * 2, y1],
                       fill=box_color)
        draw.text((x1 + pad, max(y1 - th - pad, 0)), label,
                  fill=(0, 0, 0), font=font)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ── Enrollment capture (web UI) ───────────────────────────────────────────────

async def _do_enrollment_capture(frame: np.ndarray):
    """Called from camera_loop when enrollment is active. Auto-captures samples."""
    loop = asyncio.get_event_loop()
    face = await loop.run_in_executor(None, detector.best_face, frame)
    now = time.monotonic()
    hold_seconds = 2.0

    if face is None or face.det_score < 0.85:
        state.enrollment_state.update({
            "valid_since": 0.0,
            "hold_progress": 0,
            "face_size_percent": 0,
            "face_fit_status": "waiting",
            "last_status": "等待人臉進入方框，請靠近鏡頭並正對攝影機 …",
        })
        return

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    face_h_ratio = max(0.0, min(1.0, (y2 - y1) / max(h, 1)))
    face_size_percent = int(round(face_h_ratio * 100))
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    off_center = abs(cx - w / 2) > w * 0.18 or abs(cy - h / 2) > h * 0.20

    if face_h_ratio < 0.32:
        state.enrollment_state.update({
            "valid_since": 0.0,
            "hold_progress": 0,
            "face_size_percent": face_size_percent,
            "face_fit_status": "too_far",
            "last_status": f"臉部太小（{face_size_percent}%），請再靠近鏡頭並讓臉填滿方框。",
        })
        return
    if face_h_ratio > 0.72:
        state.enrollment_state.update({
            "valid_since": 0.0,
            "hold_progress": 0,
            "face_size_percent": face_size_percent,
            "face_fit_status": "too_close",
            "last_status": f"臉部太近（{face_size_percent}%），請稍微後退到方框內。",
        })
        return
    if off_center:
        state.enrollment_state.update({
            "valid_since": 0.0,
            "hold_progress": 0,
            "face_size_percent": face_size_percent,
            "face_fit_status": "off_center",
            "last_status": "請將臉移到方框中央，保持正面面向鏡頭。",
        })
        return

    valid_since = state.enrollment_state.get("valid_since") or now
    elapsed = now - valid_since
    hold_progress = int(min(100, round(elapsed / hold_seconds * 100)))
    state.enrollment_state.update({
        "valid_since": valid_since,
        "hold_progress": hold_progress,
        "face_size_percent": face_size_percent,
        "face_fit_status": "holding",
        "last_status": f"位置正確，請保持不動：{hold_progress}%",
    })

    if elapsed < hold_seconds:
        return

    # Save embedding
    person_id = state.enrollment_state["person_id"]
    emb = face.embedding.astype(np.float32)
    conn = get_db()
    conn.execute(
        "INSERT INTO face_embeddings (person_id, embedding) VALUES (?, ?)",
        (person_id, emb.tobytes()),
    )
    conn.commit()
    conn.close()

    state.enrollment_state["samples_captured"] += 1
    state.enrollment_state["last_capture_time"] = now
    state.enrollment_state["valid_since"] = 0.0
    state.enrollment_state["hold_progress"] = 0
    state.enrollment_state["face_size_percent"] = face_size_percent
    state.enrollment_state["face_fit_status"] = "captured"
    captured = state.enrollment_state["samples_captured"]
    needed = state.enrollment_state["samples_needed"]
    state.enrollment_state["last_status"] = f"✓ 樣本 {captured}/{needed} 擷取成功，請繼續保持臉在方框內。"
    logger.info("Enrollment %s: sample %d/%d captured", state.enrollment_state["person_name"], captured, needed)

    if captured >= needed:
        state.enrollment_state["active"] = False
        state.enrollment_state["completed"] = True
        state.enrollment_state["hold_progress"] = 100
        state.enrollment_state["face_fit_status"] = "completed"
        state.enrollment_state["last_status"] = f"✓ 錄製完成，已取得 {needed} 個樣本，可以存檔。"
        state.live_status.update({
            "phase": "enrollment_done",
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
        matcher.reload()
        logger.info("Enrollment completed for %s (ID=%d)", state.enrollment_state["person_name"], person_id)


# ── Database helpers ──────────────────────────────────────────────────────────

def _create_event(person_id, person_name: str, confidence: float,
                  image_path: str, status: str = "detected") -> int:
    conn = get_db()
    conn.execute(
        "INSERT INTO events (person_id, person_name, confidence, status, event_image_path) "
        "VALUES (?, ?, ?, ?, ?)",
        (person_id, person_name, confidence, status, image_path),
    )
    conn.commit()
    eid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return eid


def _update_event(event_id: int, transcript: str, summary: str, status: str):
    conn = get_db()
    conn.execute(
        "UPDATE events SET transcript=?, summary=?, status=? WHERE id=?",
        (transcript, summary, status, event_id),
    )
    conn.commit()
    conn.close()


def _save_snapshot(frame: np.ndarray, label: str) -> str:
    os.makedirs(settings.events_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(settings.events_dir, f"{label}_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path


# ── Frame processing ──────────────────────────────────────────────────────────

async def _handle_matched(frame: np.ndarray, person_id: int, name: str,
                           department: str, confidence: float):
    """Matched identity: save snapshot, start conversation."""
    from conversation.realtime import RealtimeConversation

    now = time.monotonic()
    if now - _last_event.get(person_id, 0) < settings.cooldown_seconds:
        return  # still in cooldown

    logger.info("✓ 身份確認：%s (conf=%.2f)", name, confidence)
    reset_generation = state.reset_generation

    image_path = _save_snapshot(frame, f"person{person_id}")
    event_id = _create_event(person_id, name, confidence, image_path,
                             status="conversation_started")

    # Broadcast: conversation starting
    state.live_status.update({
        "phase": "conversation",
        "person_name": name,
        "department": department,
        "confidence": float(round(confidence, 3)),
        "transcript": [],
        "event_id": event_id,
        "purpose_summary": "",
        "completed_at": "",
        "cooldown_remaining": 0,
        "completion_status": "",
        "reset_generation": state.reset_generation,
        "camera_off": False,
        "recognition_enabled": state.recognition_enabled,
    })
    state.push_event(state.live_status)

    convo = RealtimeConversation(name, department)
    convo_task = asyncio.create_task(convo.run())
    reset_requested = False
    purpose_summary = "（未提供理由）"
    completion_status = "completed"
    try:
        deadline = time.monotonic() + settings.conversation_max_seconds + 15
        while True:
            if state.reset_generation != reset_generation:
                reset_requested = True
                convo_task.cancel()
                try:
                    await convo_task
                except asyncio.CancelledError:
                    pass
                _update_event(event_id, "", "（使用者手動重啟流程）", status="reset")
                logger.info("事件 #%d 已由前台手動重啟流程", event_id)
                return

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                convo_task.cancel()
                try:
                    await convo_task
                except asyncio.CancelledError:
                    pass
                raise asyncio.TimeoutError

            done, _ = await asyncio.wait({convo_task}, timeout=min(0.2, remaining))
            if done:
                result = convo_task.result()
                break

        _update_event(
            event_id,
            transcript=result["transcript"],
            summary=result["summary"],
            status="completed" if result["completed"] else "timeout",
        )
        purpose_summary = result["summary"]
        completion_status = "completed" if result["completed"] else "timeout"
        _last_event[person_id] = time.monotonic()
        logger.info("事件 #%d 對話完成：%s", event_id, result["summary"])
    except asyncio.TimeoutError:
        purpose_summary = "（對話逾時）"
        completion_status = "timeout"
        _update_event(event_id, "", purpose_summary, status="timeout")
        _last_event[person_id] = time.monotonic()
        logger.warning("事件 #%d 對話逾時", event_id)
    except Exception as exc:
        purpose_summary = f"（錯誤：{exc}）"
        completion_status = "error"
        _update_event(event_id, "", purpose_summary, status="error")
        _last_event[person_id] = time.monotonic()
        logger.exception("事件 #%d 對話錯誤", event_id)
    finally:
        if reset_requested:
            return
        completed_at = datetime.now()
        state.camera_available = False
        _clear_camera_frames()
        for remaining in range(settings.cooldown_seconds, -1, -1):
            if state.reset_generation != reset_generation:
                return
            state.live_status.update({
                "phase": "completed",
                "person_name": name,
                "department": department,
                "confidence": float(round(confidence, 3)),
                "event_id": event_id,
                "purpose_summary": purpose_summary,
                "completed_at": completed_at.isoformat(timespec="seconds"),
                "cooldown_remaining": remaining,
                "completion_status": completion_status,
                "reset_generation": state.reset_generation,
                "camera_off": True,
                "recognition_enabled": state.recognition_enabled,
            })
            state.push_event(state.live_status)
            if remaining > 0:
                await asyncio.sleep(1)
        # Full reset to idle
        _clear_camera_frames()
        state.camera_available = False
        state.live_status.update({"phase": "idle" if state.recognition_enabled else "recognition_off", "person_name": "", "department": "",
                                   "confidence": 0.0, "transcript": [], "event_id": None,
                                    "purpose_summary": "", "completed_at": "",
                                    "cooldown_remaining": 0, "completion_status": "",
                                    "reset_generation": state.reset_generation,
                                    "camera_off": True if state.recognition_enabled else False,
                                    "recognition_enabled": state.recognition_enabled})
        state.push_event(state.live_status)


async def _speak_warning(message: str):
    """Play a one-shot TTS warning via OpenAI TTS API."""
    if not settings.openai_api_key:
        return
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={"model": "tts-1", "input": message, "voice": "alloy",
                      "response_format": "pcm"},
            )
            resp.raise_for_status()
            pcm_bytes = resp.content
        arr = np.frombuffer(pcm_bytes, dtype=np.int16)
        from conversation.realtime import _get_device_rate, _resample
        device = settings.audio_output_device if settings.audio_output_device >= 0 else None
        dev_rate = _get_device_rate(device, "output")
        arr_out = _resample(arr, 24000, dev_rate)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: (sd.play(arr_out, samplerate=dev_rate, device=device), sd.wait()),
        )
    except Exception as exc:
        logger.error("Warning TTS failed: %s", exc)


async def _handle_unknown(frame: np.ndarray, det_score: float):
    """Unknown face: enter warning state, play TTS, hold for 15 s, then idle."""
    now = time.monotonic()
    if now - _last_event.get(None, 0) < 30:  # 30 s debounce
        return
    _last_event[None] = now
    reset_generation = state.reset_generation

    logger.warning("⚠ 未知人臉偵測 (det_score=%.2f)", det_score)
    image_path = _save_snapshot(frame, "unknown")
    _create_event(None, "未知", 0.0, image_path, status="unknown")

    state.live_status.update({
        "phase": "warning",
        "person_name": "",
        "department": "",
        "confidence": float(round(det_score, 3)),
        "transcript": [],
        "event_id": None,
        "purpose_summary": "",
        "completed_at": "",
        "cooldown_remaining": 0,
        "completion_status": "",
        "reset_generation": state.reset_generation,
        "camera_off": False,
        "recognition_enabled": state.recognition_enabled,
    })
    state.push_event(state.live_status)

    await _speak_warning("無法辨識您的身份，將通知安保人員，請稍候。")

    # Hold warning state for 15 s
    for _ in range(15):
        if state.reset_generation != reset_generation:
            return
        await asyncio.sleep(1)

    # Return to idle
    if state.reset_generation == reset_generation:
        _clear_camera_frames()
        state.camera_available = False
        state.live_status.update({
            "phase": "idle", "person_name": "", "department": "",
            "confidence": 0.0, "transcript": [], "event_id": None,
            "purpose_summary": "", "completed_at": "",
            "cooldown_remaining": 0, "completion_status": "",
            "reset_generation": state.reset_generation,
            "camera_off": True if state.recognition_enabled else False,
            "recognition_enabled": state.recognition_enabled,
        })
        state.push_event(state.live_status)


async def _process_frame(frame: np.ndarray):
    """Detect all faces, annotate frame, dispatch best face to handlers."""
    loop = asyncio.get_event_loop()
    faces = await loop.run_in_executor(None, detector.detect, frame)

    detections = []
    best: tuple | None = None  # (face_obj, det_dict)
    best_score = -1.0

    for face in faces:
        if face.det_score < settings.face_det_min_score:
            continue
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        result = matcher.match(face.embedding)
        if result:
            pid, name, dept, conf = result
            det = {"bbox": [x1, y1, x2, y2], "name": name, "is_known": True,
                   "confidence": float(round(conf, 3)), "person_id": int(pid), "department": dept}
        else:
            det = {"bbox": [x1, y1, x2, y2], "name": "未知", "is_known": False,
                   "confidence": float(round(face.det_score, 3)), "person_id": None, "department": ""}
        detections.append(det)
        if face.det_score > best_score:
            best_score = face.det_score
            best = (face, det)

    # Annotate frame with bounding boxes
    state.annotated_frame = await loop.run_in_executor(None, _draw_boxes, frame, detections)

    # Keep faces list in live_status for SSE clients
    state.live_status["faces"] = detections

    # Handle best face
    if best:
        face_obj, det = best
        if det["is_known"]:
            await _handle_matched(frame, det["person_id"], det["name"],
                                  det["department"], det["confidence"])
        else:
            await _handle_unknown(frame, face_obj.det_score)


# ── Camera loop ───────────────────────────────────────────────────────────────

async def camera_loop():
    loop = asyncio.get_event_loop()
    reset_generation = state.reset_generation
    reset_skip_frames = 0
    logger.info("Opening camera %d …", settings.camera_index)
    cap = cv2.VideoCapture(settings.camera_index)

    if not cap.isOpened():
        logger.error("Cannot open camera %d — camera loop exiting.", settings.camera_index)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    interval = settings.face_detection_interval_ms / 1000.0

    logger.info("Camera ready. Face detection interval: %.1f s", interval)

    try:
        while True:
            ret, frame = await loop.run_in_executor(None, cap.read)
            if not ret:
                _mark_camera_unavailable("camera read failed")
                await asyncio.sleep(1)
                continue

            if state.reset_generation != reset_generation:
                reset_generation = state.reset_generation
                _last_event.clear()
                state.camera_available = False
                _clear_camera_frames()
                reset_skip_frames = 4
                logger.info("Camera stream reset requested; dropping buffered frames.")
                continue

            if reset_skip_frames > 0:
                reset_skip_frames -= 1
                continue

            # Share frame for snapshot API
            state.current_frame = frame
            _mark_camera_ready()

            # Enrollment takes priority over conversation
            if state.enrollment_state["active"]:
                state.annotated_frame = frame  # no boxes during enrollment
                await _do_enrollment_capture(frame)
            elif state.enrollment_state.get("completed"):
                state.annotated_frame = frame  # keep preview alive but pause recognition until saved
            elif not state.recognition_enabled:
                state.annotated_frame = frame
                state.live_status.update({
                    "phase": "recognition_off",
                    "faces": [],
                    "person_name": "",
                    "department": "",
                    "confidence": 0.0,
                    "camera_off": False,
                    "recognition_enabled": False,
                    "reset_generation": state.reset_generation,
                })
            elif not _conversation_lock.locked():
                async with _conversation_lock:
                    await _process_frame(frame)
            elif state.annotated_frame is None:
                state.annotated_frame = frame  # ensure stream always has something

            await asyncio.sleep(interval)
    finally:
        cap.release()
        logger.info("Camera released.")


# ── Periodic tasks ────────────────────────────────────────────────────────────

async def _reload_embeddings_loop():
    while True:
        await asyncio.sleep(60)
        try:
            matcher.reload()
        except Exception as exc:
            logger.error("Embedding reload error: %s", exc)


async def _cleanup_loop():
    while True:
        await asyncio.sleep(3600)
        try:
            cleanup_old_events()
        except Exception as exc:
            logger.error("Cleanup error: %s", exc)


# ── Startup ───────────────────────────────────────────────────────────────────

async def main():
    # Ensure directories exist
    os.makedirs(settings.events_dir, exist_ok=True)
    os.makedirs(settings.models_dir, exist_ok=True)

    # Init DB
    init_db()

    # Load face model (blocking — run in executor to avoid blocking loop)
    loop = asyncio.get_event_loop()
    logger.info("Loading face recognition model …")
    await loop.run_in_executor(None, detector.load)

    # Load person embeddings
    matcher.reload()

    # Build FastAPI app
    from api.app import create_app
    app = create_app()
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=settings.dashboard_port,
        log_level=settings.log_level.lower(),
        access_log=False,
    )
    server = uvicorn.Server(config)

    logger.info("Starting all tasks … Dashboard → http://0.0.0.0:%d", settings.dashboard_port)

    await asyncio.gather(
        server.serve(),
        camera_loop(),
        _reload_embeddings_loop(),
        _cleanup_loop(),
    )


if __name__ == "__main__":
    asyncio.run(main())

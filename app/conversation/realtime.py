"""
OpenAI Realtime API WebSocket client for voice-based machine-room entry interview.

Flow:
  1. Connect to wss://api.openai.com/v1/realtime
  2. Send session.update with Chinese system prompt including the visitor's name
  3. Trigger an initial greeting via response.create
  4. Stream microphone → input_audio_buffer.append (PCM16 24 kHz)
  5. Receive response.audio.delta chunks → buffer → play when response.audio.done fires
  6. Collect transcript from response.audio_transcript.done + input transcription events
  7. Detect closing phrase or timeout → return transcript + summary
"""

import asyncio
import base64
import json
import logging
import numpy as np
import sounddevice as sd
import time
from scipy.signal import resample_poly
from math import gcd
import websockets
from config import settings

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """你是備品室入口的語音助理，只負責確認備品室進入登記資訊，不回答其他問題。

目前訪客身份已由人臉辨識確認為：{name}（部門：{department}）。

請依下列步驟進行，並全程使用繁體中文：

1. 簡短問候對方，必須明確說出辨識到的姓名「{name}」，並告知其身份已確認。

2. 必須依序確認以下兩項資訊，缺少任何一項都不能結束對話：
   - 【進入目的】：今天進入備品室的原因是什麼？
   - 【停留時間】：預計停留多久？

3. 若對方一次回答了兩項，確認已收到兩項即可。
   若對方只回答其中一項，必須繼續追問另一項，直到兩項都明確收集到。

4. 確認收到兩項後，請複述確認內容，例如：
   「好的，您今天進入目的是[目的]，預計停留[時間]，已完成記錄，請進。」

5. 【嚴格限制】：在尚未收到【進入目的】且【停留時間】這兩項之前，絕對不能說「請進」或「已完成記錄」。

6. 不要詢問工單號碼、申請紀錄、申請單、核准紀錄或任何額外文件資訊。

保持簡潔、親切但正式的語氣。"""

# Phrases that indicate the AI has finished the interview
_DONE_PHRASES = ["請進", "已記錄", "結束", "再見", "謝謝配合"]


def _refresh_audio_devices() -> None:
    """Refresh PortAudio's cached device list after USB audio changes."""
    try:
        sd._terminate()
        sd._initialize()
    except Exception as exc:
        logger.debug("PortAudio refresh failed: %s", exc)


def _resolve_audio_device(
    device_id: int | None,
    kind: str,
    preferred_name: str = "",
    require_preferred: bool = False,
    refresh_devices: bool = False,
) -> int | None:
    """Return a usable sounddevice device id, or None for system default."""
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    if preferred_name:
        needle = preferred_name.casefold()
        for attempt in range(1, 13):
            if refresh_devices:
                _refresh_audio_devices()
            for idx, info in enumerate(sd.query_devices()):
                name = str(info.get("name", ""))
                if needle in name.casefold() and int(info.get(channel_key, 0)) > 0:
                    logger.info("Using %s audio device by name '%s': #%d %s",
                                kind, preferred_name, idx, name)
                    return idx
            if attempt < 12:
                logger.warning("Waiting for %s audio device name '%s' to appear (attempt %d/12).",
                               kind, preferred_name, attempt)
                time.sleep(0.5)
        message = f"No {kind} audio device name contains '{preferred_name}'"
        if require_preferred:
            raise RuntimeError(message)
        logger.warning("%s; falling back to configured index/default.", message)

    if device_id is None:
        return None
    try:
        info = sd.query_devices(device_id, kind=kind)
        if int(info.get(channel_key, 0)) > 0:
            return device_id
    except Exception as exc:
        logger.warning("Configured %s audio device %s is unavailable; using system default. (%s)",
                       kind, device_id, exc)
        return None

    logger.warning("Configured %s audio device %s has no %s channels; using system default.",
                   kind, device_id, kind)
    return None


def _get_device_rate(device_id: int | None, kind: str) -> int:
    """Return the device's default sample rate as an integer."""
    try:
        info = sd.query_devices(device_id, kind=kind)
        return int(info["default_samplerate"])
    except Exception:
        return settings.audio_sample_rate


def _resample(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample int16 PCM array from src_rate to dst_rate."""
    if src_rate == dst_rate:
        return data
    g = gcd(src_rate, dst_rate)
    up, down = dst_rate // g, src_rate // g
    float_data = data.astype(np.float32)
    resampled = resample_poly(float_data, up, down)
    return np.clip(resampled, -32768, 32767).astype(np.int16)


class RealtimeConversation:
    """Manages a single OpenAI Realtime API session for one entry event."""

    def __init__(self, person_name: str, department: str):
        self.person_name = person_name
        self.department = department
        self._transcript: list[str] = []
        self._done = False
        self._output_playing = False
        self._mic_suppressed_until = 0.0

    async def run(self) -> dict:
        """
        Execute the conversation.
        Returns {"transcript": str, "summary": str, "completed": bool}
        """
        if not settings.openai_api_key:
            logger.warning("OPENAI_API_KEY not set — skipping conversation.")
            return {"transcript": "", "summary": "(API 金鑰未設定)", "completed": False}

        url = f"{settings.openai_realtime_url}?model={settings.openai_realtime_model}"
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
        }

        try:
            async with websockets.connect(
                url,
                extra_headers=headers,
                max_size=10 * 1024 * 1024,
            ) as ws:
                await self._configure_session(ws)

                mic_task = asyncio.create_task(self._mic_sender(ws))
                recv_task = asyncio.create_task(self._receiver(ws))

                try:
                    await asyncio.wait_for(recv_task, timeout=settings.conversation_max_seconds)
                except asyncio.TimeoutError:
                    logger.warning("Conversation timed out after %ds", settings.conversation_max_seconds)
                finally:
                    mic_task.cancel()
                    try:
                        await mic_task
                    except asyncio.CancelledError:
                        pass

        except Exception as exc:
            logger.error("Realtime API error: %s", exc)

        transcript = "\n".join(self._transcript)
        summary = self._build_summary()
        return {"transcript": transcript, "summary": summary, "completed": self._done}

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _configure_session(self, ws):
        instructions = _SYSTEM_PROMPT.format(
            name=self.person_name,
            department=self.department or "未知部門",
        )
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": instructions,
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "noise_reduction": {"type": "far_field"},
                        "transcription": {"model": "whisper-1", "language": "zh"},
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 800,
                            "create_response": True,
                            "interrupt_response": False,
                        },
                    },
                    "output": {
                        "voice": "alloy",
                        "format": {"type": "audio/pcm", "rate": 24000},
                    },
                },
            },
        }))
        # Let the AI open the conversation
        await asyncio.sleep(0.3)
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "instructions": (
                    f"請立即問候「{self.person_name}」，明確說出已辨識為"
                    f"「{self.person_name}」，並詢問兩項資訊：進入備品室的【目的】以及預計【停留多久】。"
                    f"必須同時收集到這兩項才能說請進。"
                ),
            },
        }))

    async def _mic_sender(self, ws):
        """Capture mic at its native rate, resample to 24 kHz, send to OpenAI."""
        api_rate = settings.audio_sample_rate          # 24000 Hz (OpenAI)
        configured_device = settings.audio_input_device if settings.audio_input_device >= 0 else None
        device = _resolve_audio_device(
            configured_device,
            "input",
            settings.audio_input_device_name,
            require_preferred=bool(settings.audio_input_device_name),
            refresh_devices=True,
        )
        dev_rate = _get_device_rate(device, "input")   # e.g. 48000 Hz

        chunk_frames = dev_rate * settings.audio_chunk_ms // 1000
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def callback(indata, _frames, _time, status):
            if status:
                logger.debug("Audio input status: %s", status)
            loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))

        logger.info("Mic: device=%s native=%d Hz → resample to %d Hz", device, dev_rate, api_rate)
        try:
            with sd.RawInputStream(
                samplerate=dev_rate,
                channels=settings.audio_channels,
                dtype="int16",
                blocksize=chunk_frames,
                device=device,
                callback=callback,
            ):
                while True:
                    raw = await queue.get()
                    if self._output_playing or loop.time() < self._mic_suppressed_until:
                        continue
                    pcm = np.frombuffer(raw, dtype=np.int16)
                    pcm_24k = _resample(pcm, dev_rate, api_rate)
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm_24k.tobytes()).decode(),
                    }))
        except Exception as exc:
            logger.error("Mic sender error: %s", exc)
            await ws.close(code=1011, reason="input audio device unavailable")

    async def _receiver(self, ws):
        """Receive events, play AI audio, collect transcript."""
        audio_buf = bytearray()
        ai_text_buf = ""

        async for raw in ws:
            event = json.loads(raw)
            etype = event.get("type", "")

            if etype in ("response.audio.delta", "response.output_audio.delta"):
                audio_buf.extend(base64.b64decode(event.get("delta", "")))

            elif etype in ("response.audio.done", "response.output_audio.done"):
                if audio_buf:
                    await self._play_audio(bytes(audio_buf))
                    audio_buf.clear()

            elif etype in ("response.audio_transcript.delta", "response.output_audio_transcript.delta"):
                ai_text_buf += event.get("delta", "")

            elif etype in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
                text = (event.get("transcript") or ai_text_buf).strip()
                if text:
                    self._transcript.append(f"[助理] {text}")
                    logger.info("AI  : %s", text)
                    # Push to live monitor
                    import state as _state
                    _state.live_status["transcript"].append({"role": "ai", "text": text})
                    _state.push_event(_state.live_status)
                    if any(p in text for p in _DONE_PHRASES):
                        self._done = True
                        await asyncio.sleep(0.5)
                        return
                ai_text_buf = ""

            elif etype == "conversation.item.input_audio_transcription.completed":
                user_text = event.get("transcript", "").strip()
                if user_text:
                    self._transcript.append(f"[訪客] {user_text}")
                    logger.info("訪客 : %s", user_text)
                    # Push to live monitor
                    import state as _state
                    _state.live_status["transcript"].append({"role": "user", "text": user_text})
                    _state.push_event(_state.live_status)

            elif etype == "error":
                logger.error("Realtime error event: %s", event.get("error"))

    async def _play_audio(self, pcm_bytes: bytes):
        """Resample AI audio from 24 kHz to output device rate, then play."""
        api_rate = settings.audio_sample_rate   # 24000 Hz
        configured_device = settings.audio_output_device if settings.audio_output_device >= 0 else None
        device = _resolve_audio_device(configured_device, "output", settings.audio_output_device_name)
        dev_rate = _get_device_rate(device, "output")  # e.g. 44100 Hz

        arr_24k = np.frombuffer(pcm_bytes, dtype=np.int16)
        arr_out = _resample(arr_24k, api_rate, dev_rate)
        loop = asyncio.get_running_loop()
        self._output_playing = True
        try:
            await loop.run_in_executor(
                None,
                lambda: (sd.play(arr_out, samplerate=dev_rate, device=device), sd.wait()),
            )
        finally:
            self._output_playing = False
            self._mic_suppressed_until = loop.time() + 0.7

    def _build_summary(self) -> str:
        visitor_lines = [
            line.replace("[訪客] ", "")
            for line in self._transcript
            if line.startswith("[訪客]")
        ]
        return "；".join(visitor_lines) if visitor_lines else "（未提供理由）"

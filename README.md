# 智慧門禁系統 — Face-Based Access Kiosk

> **語言 / Language**: [中文](#中文說明) | [English](#english-guide)

---

## 示範影片 / Demo Video

[![智慧門禁系統示範](https://img.youtube.com/vi/b6HfHJ3-yZY/maxresdefault.jpg)](https://www.youtube.com/watch?v=b6HfHJ3-yZY)

▶ [https://www.youtube.com/watch?v=b6HfHJ3-yZY](https://www.youtube.com/watch?v=b6HfHJ3-yZY)

---

## 中文說明

### 系統概述

本系統部署於 NVIDIA Jetson Orin Nano，結合臉部辨識與 OpenAI Realtime API 語音對話，實現機房入口身份驗證流程：

1. **臉部辨識** — 攝影機持續偵測並比對已授權人員。
2. **語音面談** — 辨識成功後自動啟動與 OpenAI Realtime API 的語音對話，詢問進入目的與預計停留時間。
3. **事件紀錄** — 自動將辨識結果、逐字稿、目的摘要寫入本機 SQLite 資料庫。
4. **Kiosk 介面** — 全螢幕 Web UI（`http://<jetson-ip>:8000/kiosk`）顯示辨識狀態、對話逐字稿、完成畫面與倒數重啟。
5. **管理 Dashboard** — `http://<jetson-ip>:8000` 提供人員管理、人臉 Enrollment、事件查詢。

---

### 硬體需求

| 元件 | 建議規格 |
|------|---------|
| 主機 | NVIDIA Jetson Orin Nano（或其他 Jetson 系列） |
| 攝影機 | USB 攝影機，解析度 720p 以上 |
| 麥克風 | USB 麥克風或 USB 音訊介面 |
| 喇叭 | 3.5mm 外接喇叭或 USB 喇叭 |
| 儲存 | 建議 16 GB 以上可用空間 |
| 網路 | 穩定網路連線（OpenAI Realtime API 需持續連線） |

---

### 軟體需求

- **Jetson 主機**：JetPack 5.x 或 6.x、Docker Engine、`docker-compose`（v1 CLI）
- **本機電腦**（部署用）：`ssh`、`git`、`scp` 或 `rsync`

---

### 安裝與部署

#### 1. 複製專案到 Jetson

```bash
# 從本機同步到 Jetson
rsync -av --exclude='data/' --exclude='models/' \
  /path/to/face-app-jetson/ \
  rogerroan@192.168.68.105:/home/rogerroan/docker/face/

# 或直接在 Jetson 上 clone
ssh rogerroan@192.168.68.105
git clone <your-repo-url> /home/rogerroan/docker/face
```

#### 2. 建立環境設定檔

```bash
cd /home/rogerroan/docker/face
cp .env.example .env
nano .env   # 填入必要設定（至少 OPENAI_API_KEY）
```

`.env` 設定說明：

```dotenv
# ── 必填 ────────────────────────────────────────────────
OPENAI_API_KEY=sk-...                # OpenAI API Key

# ── 攝影機 ─────────────────────────────────────────────
CAMERA_INDEX=0                       # /dev/video0 → 0，/dev/video2 → 2
                                     # 執行 ls /dev/video* 確認

# ── 臉部辨識 ────────────────────────────────────────────
FACE_CONFIDENCE_THRESHOLD=0.45       # 相似度門檻 (0~1)，越高越嚴格
FACE_DETECTION_INTERVAL_MS=500       # 偵測間隔毫秒（建議 300~1000）

# ── 音訊設備 ─────────────────────────────────────────────
# -1 代表系統預設；若需指定設備，先執行 make audio-devices 查詢 ID
AUDIO_INPUT_DEVICE=-1                # 麥克風設備編號
AUDIO_OUTPUT_DEVICE=-1               # 喇叭設備編號

# ── 對話設定 ─────────────────────────────────────────────
OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview
CONVERSATION_MAX_SECONDS=120         # 最長對話秒數（超時自動結束）
COOLDOWN_SECONDS=30                  # 完成後重啟流程的等待秒數

# ── 資料保留 ─────────────────────────────────────────────
RETENTION_DAYS=30                    # 事件紀錄保留天數

# ── 日誌等級 ─────────────────────────────────────────────
LOG_LEVEL=INFO                       # DEBUG / INFO / WARNING / ERROR
```

#### 3. 建構並啟動

```bash
cd /home/rogerroan/docker/face

# 建構 Docker image（第一次或程式碼有更新時執行）
docker-compose up -d --build

# 之後只需啟動
docker-compose up -d
```

> **注意**：Jetson 使用 `docker-compose`（v1），不是 `docker compose`（v2）。

#### 4. 確認服務狀態

```bash
docker-compose ps
docker-compose logs -f face-app
```

服務正常啟動後可存取：
- Kiosk UI：`http://192.168.68.105:8000/kiosk`
- 管理 Dashboard：`http://192.168.68.105:8000`

---

### 人員 Enrollment（人臉註冊）

#### 新增人員

```bash
# 互動式新增（輸入姓名、部門、員工編號）
docker-compose exec face-app python enrollment/cli.py add \
  --name "王小明" --dept "IT" --employee-id "EMP001"
```

#### 人臉 Enrollment（需要攝影機）

```bash
# 互動式 enrollment（對著攝影機拍攝多張照片）
docker-compose exec -it face-app python enrollment/cli.py enroll \
  --person-id <person-id>
```

#### 列出所有人員

```bash
docker-compose exec face-app python enrollment/cli.py list
```

#### 使用 Makefile 快捷指令

```bash
make enroll-list    # 列出人員
make enroll-add     # 互動新增人員
make enroll-run     # 互動 enrollment
```

---

### 音訊設備設定

如果系統有多個音訊設備，需要指定正確的設備編號：

```bash
# 列出所有可用音訊設備
docker-compose exec face-app python -c "import sounddevice; print(sounddevice.query_devices())"
# 或使用 Makefile
make audio-devices
```

輸出範例：
```
  0 bcm2835 HDMI: - (hw:0,0), ALSA (0 in, 8 out)
  1 USB Audio Device: - (hw:1,0), ALSA (1 in, 0 out)  ← 麥克風
  2 USB Audio Device: - (hw:2,0), ALSA (0 in, 2 out)  ← 喇叭
```

將對應編號填入 `.env`：
```dotenv
AUDIO_INPUT_DEVICE=1
AUDIO_OUTPUT_DEVICE=2
```

更改後重啟：
```bash
docker-compose restart face-app
```

---

### 常用 Makefile 指令

```bash
make build          # 建構 Docker image
make up             # 啟動服務（背景）
make down           # 停止服務
make logs           # 查看即時 log
make ps             # 顯示容器狀態
make restart        # 重啟服務
make shell          # 進入容器 shell
make audio-devices  # 列出音訊設備
make clean          # 停止服務並刪除 image 與本機 data（謹慎使用）
```

---

### 使用流程

1. **服務啟動** → 攝影機開啟，開始偵測人臉。
2. **偵測到已知人員** → 畫面顯示姓名，自動啟動語音對話。
3. **語音對話** → AI 詢問進入目的與預計停留時間，訪客以語音回答。
4. **對話完成** → 畫面顯示「辨識完成」、人員姓名、進入時間與目的摘要，開始倒數 30 秒（預設）。
5. **重啟流程** → 倒數結束自動重啟，或點擊 Kiosk 右上角「重啟流程」按鈕立即重啟。

#### Kiosk 介面說明

| 狀態 | 說明 |
|------|------|
| 等待辨識 | 攝影機偵測中，尚未識別到授權人員 |
| 辨識成功 | 已識別人員，正在啟動對話 |
| 對話中 | 語音面談進行中，顯示即時逐字稿 |
| 已完成 | 顯示進入資訊與倒數，等待流程重啟 |

---

### 事件紀錄

所有辨識事件、對話逐字稿與目的摘要均儲存於：

- **資料庫**：`./data/face.db`（SQLite）
- **事件圖片**：`./data/events/`

資料庫保留天數由 `RETENTION_DAYS` 控制（預設 30 天）。

---

### 目錄結構

```
face-app-jetson/
├── .env.example          # 環境變數範本
├── docker-compose.yml    # Docker Compose 設定
├── Makefile              # 常用指令快捷
├── data/                 # 持久化資料（SQLite、事件圖片）
├── models/               # InsightFace 模型快取
└── app/
    ├── Dockerfile
    ├── requirements.txt
    ├── config.py          # 環境變數載入
    ├── main.py            # 主程式（攝影機迴圈 + 流程控制）
    ├── state.py           # 共享狀態
    ├── database.py        # SQLite 操作
    ├── face/
    │   ├── detector.py    # 臉部偵測（InsightFace）
    │   └── matcher.py     # 身份比對
    ├── conversation/
    │   └── realtime.py    # OpenAI Realtime API 客戶端
    ├── enrollment/
    │   └── cli.py         # 人員/人臉 Enrollment CLI
    └── api/
        ├── app.py         # FastAPI 應用程式
        ├── routes.py      # API 路由
        └── templates/
            ├── kiosk.html     # Kiosk 前台介面
            └── dashboard.html # 管理 Dashboard
```

---

### 常見問題

**Q：攝影機無法開啟 / 顯示黑畫面**
- 確認 `CAMERA_INDEX` 設定正確（`ls /dev/video*`）
- 確認容器以 `privileged: true` 啟動（`docker-compose.yml` 預設已設定）

**Q：音訊設備找不到 / 無法收音**
- 執行 `make audio-devices` 查詢設備編號
- 確認 USB 麥克風/喇叭已插入，且 Jetson 已識別（`aplay -l` / `arecord -l`）

**Q：人臉辨識總是不到位**
- 調低 `FACE_CONFIDENCE_THRESHOLD`（例如 `0.35`）
- 確保 Enrollment 時光線充足、正面對著鏡頭
- 重新進行 Enrollment

**Q：OpenAI Realtime API 無法連線**
- 確認 `OPENAI_API_KEY` 正確且有 Realtime API 存取權限
- 確認 Jetson 有穩定網路連線（`ping api.openai.com`）

**Q：更新程式碼後需要重新建構**
```bash
docker-compose up -d --build
```

---

---

## English Guide

### System Overview

This system runs on an NVIDIA Jetson Orin Nano and combines face recognition with OpenAI Realtime API voice conversation to provide machine-room entry verification:

1. **Face Recognition** — Continuously detects and matches faces against an authorized personnel roster.
2. **Voice Interview** — Upon successful recognition, automatically starts a voice session via OpenAI Realtime API to ask for entry purpose and estimated stay time.
3. **Audit Logging** — Automatically stores recognition results, full transcripts, and purpose summaries into a local SQLite database.
4. **Kiosk UI** — Full-screen web interface (`http://<jetson-ip>:8000/kiosk`) showing recognition status, live transcript, completion screen, and countdown to restart.
5. **Admin Dashboard** — `http://<jetson-ip>:8000` for personnel management, face enrollment, and event queries.

---

### Hardware Requirements

| Component | Recommended Spec |
|-----------|-----------------|
| Host | NVIDIA Jetson Orin Nano (or other Jetson series) |
| Camera | USB webcam, 720p or higher |
| Microphone | USB microphone or USB audio interface |
| Speaker | 3.5mm external speaker or USB speaker |
| Storage | 16 GB+ available disk space recommended |
| Network | Stable internet connection (required for OpenAI Realtime API) |

---

### Software Requirements

- **Jetson host**: JetPack 5.x or 6.x, Docker Engine, `docker-compose` (v1 CLI)
- **Local machine** (for deployment): `ssh`, `git`, `scp` or `rsync`

---

### Installation & Deployment

#### 1. Copy the project to the Jetson

```bash
# Sync from local machine to Jetson
rsync -av --exclude='data/' --exclude='models/' \
  /path/to/face-app-jetson/ \
  rogerroan@192.168.68.105:/home/rogerroan/docker/face/

# Or clone directly on the Jetson
ssh rogerroan@192.168.68.105
git clone <your-repo-url> /home/rogerroan/docker/face
```

#### 2. Create the environment config file

```bash
cd /home/rogerroan/docker/face
cp .env.example .env
nano .env   # Fill in required values (at minimum: OPENAI_API_KEY)
```

`.env` configuration reference:

```dotenv
# ── Required ────────────────────────────────────────────
OPENAI_API_KEY=sk-...                # Your OpenAI API Key

# ── Camera ──────────────────────────────────────────────
CAMERA_INDEX=0                       # /dev/video0 → 0, /dev/video2 → 2
                                     # Run: ls /dev/video* to confirm

# ── Face Recognition ────────────────────────────────────
FACE_CONFIDENCE_THRESHOLD=0.45       # Cosine similarity threshold (0–1); higher = stricter
FACE_DETECTION_INTERVAL_MS=500       # Detection interval in ms (recommended: 300–1000)

# ── Audio Devices ────────────────────────────────────────
# -1 = system default; to specify a device run: make audio-devices
AUDIO_INPUT_DEVICE=-1                # Microphone device index
AUDIO_OUTPUT_DEVICE=-1               # Speaker device index

# ── Conversation ─────────────────────────────────────────
OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview
CONVERSATION_MAX_SECONDS=120         # Max conversation duration in seconds
COOLDOWN_SECONDS=30                  # Countdown (seconds) before restarting the flow

# ── Data Retention ───────────────────────────────────────
RETENTION_DAYS=30                    # Days to keep event records

# ── Logging ──────────────────────────────────────────────
LOG_LEVEL=INFO                       # DEBUG / INFO / WARNING / ERROR
```

#### 3. Build and start

```bash
cd /home/rogerroan/docker/face

# Build the Docker image (required on first run or after code changes)
docker-compose up -d --build

# Subsequent starts (no code change)
docker-compose up -d
```

> **Note**: Jetson uses `docker-compose` (v1 CLI), not `docker compose` (v2).

#### 4. Verify service status

```bash
docker-compose ps
docker-compose logs -f face-app
```

Once the service starts successfully, open:
- Kiosk UI: `http://192.168.68.105:8000/kiosk`
- Admin Dashboard: `http://192.168.68.105:8000`

---

### Personnel Enrollment (Face Registration)

#### Add a person

```bash
docker-compose exec face-app python enrollment/cli.py add \
  --name "John Smith" --dept "IT" --employee-id "EMP001"
```

#### Enroll face (requires camera)

```bash
docker-compose exec -it face-app python enrollment/cli.py enroll \
  --person-id <person-id>
```

Follow the on-screen prompts to capture multiple face samples. Ensure good lighting and face the camera directly.

#### List all personnel

```bash
docker-compose exec face-app python enrollment/cli.py list
```

#### Makefile shortcuts

```bash
make enroll-list    # List all personnel
make enroll-add     # Interactively add a person
make enroll-run     # Interactively enroll face
```

---

### Audio Device Configuration

If the system has multiple audio devices, you need to specify the correct device index:

```bash
# List all available audio devices
docker-compose exec face-app python -c "import sounddevice; print(sounddevice.query_devices())"
# Or use the Makefile shortcut
make audio-devices
```

Example output:
```
  0 bcm2835 HDMI: - (hw:0,0), ALSA (0 in, 8 out)
  1 USB Audio Device: - (hw:1,0), ALSA (1 in, 0 out)  ← microphone
  2 USB Audio Device: - (hw:2,0), ALSA (0 in, 2 out)  ← speaker
```

Update `.env` with the correct indices:
```dotenv
AUDIO_INPUT_DEVICE=1
AUDIO_OUTPUT_DEVICE=2
```

Then restart the service:
```bash
docker-compose restart face-app
```

---

### Makefile Command Reference

```bash
make build          # Build the Docker image
make up             # Start the service (background)
make down           # Stop the service
make logs           # Tail live logs
make ps             # Show container status
make restart        # Restart the service
make shell          # Open a shell inside the container
make audio-devices  # List available audio device indices
make clean          # Stop service and remove image + local data (use with caution)
```

---

### Kiosk Usage Flow

1. **Service starts** → Camera opens, face detection begins.
2. **Known person detected** → Name is displayed, voice session starts automatically.
3. **Voice interview** → AI asks for entry purpose and estimated stay time; visitor answers verbally.
4. **Conversation complete** → Screen shows "Verification Complete", person's name, entry time, and purpose summary; a countdown begins (default 30 s).
5. **Flow restarts** → Countdown auto-restarts, or click **Restart Flow** (top-right of Kiosk) to restart immediately.

#### Kiosk UI States

| State | Description |
|-------|-------------|
| Waiting | Camera is detecting; no authorized person recognized yet |
| Recognized | Person identified; voice session starting |
| Conversation | Voice interview in progress; live transcript displayed |
| Completed | Entry info and countdown shown; waiting for flow restart |

---

### Event Records

All recognition events, conversation transcripts, and purpose summaries are stored in:

- **Database**: `./data/face.db` (SQLite)
- **Event images**: `./data/events/`

Retention period is controlled by `RETENTION_DAYS` (default: 30 days).

---

### Directory Structure

```
face-app-jetson/
├── .env.example          # Environment variable template
├── docker-compose.yml    # Docker Compose configuration
├── Makefile              # Shortcut commands
├── data/                 # Persistent data (SQLite DB, event images)
├── models/               # InsightFace model cache
└── app/
    ├── Dockerfile
    ├── requirements.txt
    ├── config.py          # Environment variable loader
    ├── main.py            # Main process (camera loop + flow control)
    ├── state.py           # Shared state
    ├── database.py        # SQLite operations
    ├── face/
    │   ├── detector.py    # Face detection (InsightFace)
    │   └── matcher.py     # Identity matching
    ├── conversation/
    │   └── realtime.py    # OpenAI Realtime API client
    ├── enrollment/
    │   └── cli.py         # Personnel / face enrollment CLI
    └── api/
        ├── app.py         # FastAPI application
        ├── routes.py      # API routes
        └── templates/
            ├── kiosk.html     # Kiosk frontend
            └── dashboard.html # Admin dashboard
```

---

### Troubleshooting

**Camera not opening / black screen**
- Verify `CAMERA_INDEX` is correct (`ls /dev/video*` on Jetson)
- Ensure the container runs with `privileged: true` (already set in `docker-compose.yml`)

**Audio device not found / no microphone input**
- Run `make audio-devices` to find device indices
- Confirm USB mic/speaker is recognized by Jetson (`aplay -l` / `arecord -l`)

**Face recognition consistently fails**
- Lower `FACE_CONFIDENCE_THRESHOLD` (e.g., `0.35`)
- Re-enroll under good lighting with the subject facing directly at the camera

**OpenAI Realtime API connection failure**
- Verify `OPENAI_API_KEY` is valid and has Realtime API access
- Check Jetson network connectivity (`ping api.openai.com`)

**Code changes not taking effect**
```bash
docker-compose up -d --build
```

---

### License

For internal use. Not for public distribution.

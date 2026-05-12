# 機房進入身份確認與語音詢問系統規劃

## 0. 需求明確度

目前需求「方向明確、可行」，可以規劃成一套在 Jetson Orin Nano 上執行的邊緣身份確認系統：

- 使用攝影機做人臉偵測與身份比對。
- 身份確認後，使用 OpenAI Realtime API 進行語音對話。
- 詢問進機房理由。
- 保存辨識結果、對話逐字稿、理由摘要與事件紀錄。
- 第一版只做紀錄與通知，不直接自動開門。

目前已確認第一版的主要產品範圍；剩餘需要釐清的是實際 USB 攝影機/音訊設備型號、dashboard 存取限制、Jetson 對 OpenAI API 的網路可達性等落地細節。

## 0.1 已確認決策

1. 第一版使用 USB 攝影機。
2. 第一版使用 USB 麥克風與 Jetson 本機外接喇叭。
3. 第一版授權人員採手動建檔並現場 enrollment。
4. 第一版只做身份確認、詢問理由、紀錄與本機 dashboard/查詢頁，不自動開門。
5. 第一版語音對話使用中文。
6. 第一版事件紀錄與逐字稿保存 30 天。
7. 第一版不加入第二因素或活體檢測，但保留後續擴充點。

## 1. 需求

### 1.1 目標

建立一套部署在 Jetson Orin Nano 的機房入口輔助確認系統，能辨識來訪者身份，並在身份確認後以語音詢問進入機房的理由，最後留下可稽核紀錄供管理者查看。

### 1.2 第一版範圍

第一版包含：

1. Jetson 本機攝影機影像擷取。
2. 人臉偵測、特徵向量抽取、身份比對。
3. 授權人員註冊與臉部樣本 enrollment。
4. 身份確認後啟動 OpenAI Realtime API 語音對話。
5. 詢問「請說明進入機房的理由」與必要追問。
6. 儲存事件、身份、confidence、理由、逐字稿、摘要、時間戳。
7. 提供本機 CLI 或簡易 dashboard/查詢頁查詢紀錄。
8. 保留後續串接通知服務的介面。

第一版不包含：

1. 不直接控制門鎖。
2. 不把人臉辨識當成唯一正式門禁憑證。
3. 不在前期串接 HR、AD、正式門禁或工單系統，除非後續明確指定。
4. 不把 OpenAI API key 或其他秘密資訊寫入 source code。

## 2. 現況

- 本機專案目錄 `/Users/rogerroan/share/face-app-jetson` 原本是空目錄。
- 目標設備是 Jetson Orin Nano，連線位置為 `ssh rogerroan@192.168.68.105`。
- 計畫階段尚未連線或修改 Jetson。
- 因為沒有既有程式碼，實作會從新建 Python 專案開始。

## 3. 釐清事項

以下問題需要在正式實作前或實作初期確認：

1. **攝影機型號與連接方式**：已確認第一版使用 USB 攝影機；仍需確認實際型號、解析度與是否支援穩定 FPS。
2. **麥克風與喇叭型號**：已確認第一版使用 USB 麥克風與 Jetson 本機外接喇叭；仍需確認實際型號、取樣率與 Linux audio device 名稱。
3. **授權人員來源**：已確認第一版手動建立人員資料並現場 enrollment；未來再評估 CSV 或 HR/門禁系統串接。
4. **通知方式**：已確認第一版寫入本機資料庫，並提供本機 dashboard/查詢頁；暫不先依賴 Email、LINE、Slack 或 Teams。
5. **對話語言**：已確認第一版使用中文；prompt、逐字稿摘要與 dashboard 顯示以中文為主。
6. **資料保存期限**：已確認第一版事件紀錄與逐字稿保存 30 天；人臉 embedding 則隨授權人員資料保存，停用人員時需可刪除。
7. **安全等級**：已確認第一版不加入 PIN、員工證或活體檢測；架構保留第二因素與活體檢測擴充點。
8. **部署網路條件**：Jetson 是否能穩定連線到 OpenAI API。

## 4. 規劃

### 4.1 系統架構

建議使用 Python 為主：

- **Edge service**：負責 camera/audio、臉部辨識、流程狀態機、OpenAI Realtime API 整合。
- **Face recognition module**：負責偵測、對齊、embedding、比對與門檻判斷。
- **Enrollment module**：負責註冊人員與臉部樣本。
- **Conversation module**：負責 OpenAI Realtime API 連線、Jetson 本機 USB 麥克風/喇叭語音輸入輸出、prompt 與 transcript。
- **Audit database**：SQLite 起步，保存人員、embedding、事件、對話紀錄與通知狀態。
- **Admin interface**：初期用 FastAPI 簡易本機 dashboard/查詢頁，必要時搭配 CLI。
- **Deployment layer**：systemd service、`.env`、健康檢查、logs。

### 4.2 建議技術選型

- Python 3.10+。
- OpenCV 或 GStreamer 做影像擷取。
- InsightFace/SCRFD/ArcFace 或相容 ONNX 模型做人臉辨識。
- SQLite 做第一版本機資料庫。
- FastAPI 做管理 API 或簡易 dashboard。
- OpenAI Realtime API 做低延遲語音對話。
- systemd 管理 Jetson 上的常駐服務。

### 4.3 流程

1. 系統啟動後初始化攝影機、音訊設備、資料庫與模型。
2. 持續偵測畫面中的人臉。
3. 若偵測到人臉，抽取 embedding 並與授權人員資料庫比對。
4. 若 confidence 未達門檻，標記為 unknown 並紀錄。
5. 若 confidence 達門檻，建立進入事件並啟動語音對話。
6. Realtime 助理以中文詢問進入機房理由，必要時追問部門、工單、維護項目或聯絡窗口。
7. 將 transcript 與摘要寫入事件紀錄。
8. 發送通知或讓管理者在 dashboard 查看。
9. 若網路、API、攝影機或音訊失敗，回退為人工登記/人工確認流程。

## 5. 任務

1. **專案基礎建置**
   - 建立 Python 專案結構、依賴管理、設定檔、logging、SQLite schema。

2. **Jetson 硬體驗證**
   - 檢查 Jetson OS、Python、CUDA/TensorRT、攝影機、麥克風、喇叭、網路與儲存空間。

3. **臉部辨識 pipeline**
   - 建立 camera capture、face detection、alignment、embedding、identity matching、threshold。

4. **人員註冊與 enrollment**
   - 建立人員資料、收集多張臉部樣本、產生 embedding、檢查樣本品質。

5. **OpenAI Realtime 語音對話**
   - 建立安全金鑰管理、音訊串流、系統 prompt、transcript、摘要輸出。

6. **稽核紀錄**
   - 儲存事件、身份、confidence、理由、逐字稿、摘要、錯誤與通知狀態。

7. **管理查詢與通知**
   - 建立 CLI 或簡易 dashboard，提供事件查詢、通知與匯出能力。

8. **Jetson 部署**
   - 建立 systemd service、安裝文件、`.env` 範例、健康檢查與 log 查詢。

9. **安全強化**
   - 加入最小權限、管理介面保護、金鑰保護、資料保存策略、活體檢測規劃。

10. **端到端驗證**
    - 測試已註冊人員、未知人員、多臉、低光源、音訊失敗、網路中斷、API 失敗等情境。

## 6. 執行步驟

### Step 1：需求與硬體確認

- 確認 USB 攝影機、USB 麥克風、外接喇叭實際型號。
- 授權人員第一版採手動建立並現場 enrollment。
- 第一版以本機資料庫與本機 dashboard/查詢頁呈現事件。
- 第一版事件紀錄與逐字稿保存 30 天，並提供清理機制。

### Step 2：建立專案骨架

- 建立 Python app 結構。
- 建立設定檔與 `.env.example`。
- 建立 SQLite schema。
- 建立 logging 與錯誤處理框架。

### Step 3：Jetson 診斷

- SSH 到 Jetson。
- 檢查 camera frame capture。
- 檢查 microphone input。
- 檢查 speaker output。
- 檢查 OpenAI API 網路連線。

### Step 4：臉部辨識 PoC

- 實作單張影像偵測。
- 實作即時 camera stream 偵測。
- 實作 embedding 與相似度比對。
- 建立 unknown 與低 confidence handling。

### Step 5：Enrollment

- 建立手動人員建檔與註冊 CLI 或 API。
- 每位人員收集多張樣本。
- 將 embedding 寫入 SQLite。
- 加入重新註冊與停用人員功能。

### Step 6：Realtime 對話

- 建立 OpenAI Realtime API client。
- 串接 Jetson 本機 USB 麥克風與外接喇叭。
- 設計固定職責 prompt。
- 儲存 transcript 與理由摘要。

### Step 7：整合主流程

- 身份確認成功後啟動對話。
- 將辨識結果與對話結果合併成事件。
- 建立通知或查詢介面。

### Step 8：部署與常駐服務

- 建立 systemd service。
- 設定開機自動啟動。
- 建立健康檢查。
- 建立 log 查詢與故障排除文件。

### Step 9：安全與驗收

- 測試錯誤接受率與錯誤拒絕率。
- 測試網路/API/硬體失敗回退。
- 檢查 secrets 不進 source code。
- 檢查敏感資料保存與刪除流程。
- 檢查 30 天事件與逐字稿清理流程。

## 7. 驗收標準

第一版完成時應滿足：

1. 已註冊人員能被辨識並顯示身份與 confidence。
2. 未註冊人員不會被誤認成授權人員。
3. 身份確認後能進行中文語音詢問。
4. 系統能保存進入理由、逐字稿、摘要與時間戳。
5. 管理者能查詢最近事件。
6. OpenAI API key 不會出現在 source code。
7. Jetson 重開機後服務能自動啟動。
8. 網路或 API 失敗時不會產生錯誤授權結果。

## 8. 風險與控制

- **臉部誤判風險**：使用門檻、低品質拒絕、多樣本 enrollment、必要時加入第二因素。
- **照片或影片 spoofing**：第一版先不加入活體檢測；正式版或接門鎖前需加入活體檢測或額外驗證。
- **隱私風險**：embedding 與逐字稿需限制存取並設定保存期限。
- **網路依賴**：Realtime API 失敗時回退人工流程。
- **門禁安全**：第一版不自動開門，避免 AI 或臉部辨識錯誤直接造成實體安全風險。

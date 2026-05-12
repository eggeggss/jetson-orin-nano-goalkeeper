#!/usr/bin/env python3
"""
人員 enrollment CLI  (headless — 不需螢幕，透過 SSH 操作)

用法:
  python enrollment/cli.py list
  python enrollment/cli.py add   --name 張三 --dept IT --employee-id E001
  python enrollment/cli.py enroll --person-id 1 [--samples 5] [--camera 0]
  python enrollment/cli.py remove --person-id 1
"""

import argparse
import sys
import os
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _db():
    from database import get_db
    return get_db()


def cmd_list():
    rows = _db().execute(
        "SELECT id, name, department, employee_id, is_active, created_at FROM persons ORDER BY id"
    ).fetchall()
    if not rows:
        print("(尚無人員資料)")
        return
    print(f"{'ID':<5}  {'姓名':<16}  {'部門':<14}  {'員工編號':<12}  {'啟用':<4}  建立時間")
    print("─" * 75)
    for r in rows:
        active = "✓" if r["is_active"] else "✗"
        print(f"{r['id']:<5}  {r['name']:<16}  {r['department'] or '':<14}  "
              f"{r['employee_id'] or '':<12}  {active:<4}  {r['created_at']}")


def cmd_add(name: str, dept: str, employee_id: str) -> int:
    conn = _db()
    conn.execute(
        "INSERT INTO persons (name, department, employee_id) VALUES (?, ?, ?)",
        (name, dept, employee_id),
    )
    conn.commit()
    pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    print(f"✓ 人員 '{name}' 建立完成，ID = {pid}")
    return pid


def cmd_enroll(person_id: int, camera: int = 0, samples: int = 5):
    """Headless enrollment: auto-capture when a good face is detected."""
    import cv2
    import numpy as np

    conn = _db()
    person = conn.execute("SELECT id, name FROM persons WHERE id = ?", (person_id,)).fetchone()
    conn.close()
    if not person:
        print(f"找不到 ID={person_id} 的人員。")
        sys.exit(1)

    from face.detector import detector
    from database import get_db as _get_db

    detector.load()

    print(f"\n開始 enrollment：{person['name']}（ID: {person_id}）")
    print(f"請正對攝影機，系統將自動擷取 {samples} 個樣本。")
    print("每次擷取間隔 2 秒，請在擷取之間稍微移動角度以增加多樣性。")
    print("按 Ctrl+C 取消。\n")

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"無法開啟攝影機 {camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    embeddings = []
    last_capture_time = 0.0
    capture_interval = 2.0   # seconds between captures
    min_score = 0.85          # detection confidence threshold for enrollment

    try:
        while len(embeddings) < samples:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            face = detector.best_face(frame)
            now = time.time()

            if face is not None and face.det_score >= min_score:
                if now - last_capture_time >= capture_interval:
                    embeddings.append(face.embedding.astype(np.float32))
                    last_capture_time = now
                    idx = len(embeddings)
                    print(f"  ✓ 樣本 {idx}/{samples} 擷取成功 (score={face.det_score:.2f})")
                else:
                    remaining = capture_interval - (now - last_capture_time)
                    print(f"  偵測到人臉，{remaining:.1f}s 後擷取下一個 …", end="\r")
            else:
                print("  等待人臉對準攝影機 …              ", end="\r")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n取消。")
    finally:
        cap.release()

    if not embeddings:
        print("未擷取任何樣本，enrollment 未完成。")
        return

    # Persist embeddings
    db = _get_db()
    for emb in embeddings:
        db.execute(
            "INSERT INTO face_embeddings (person_id, embedding) VALUES (?, ?)",
            (person_id, emb.tobytes()),
        )
    db.commit()
    db.close()
    print(f"\n✓ 已儲存 {len(embeddings)} 個樣本給 {person['name']}。")
    print("  提示：可執行 `python enrollment/cli.py list` 確認。")


def cmd_remove(person_id: int):
    conn = _db()
    conn.execute("UPDATE persons SET is_active = 0 WHERE id = ?", (person_id,))
    conn.commit()
    conn.close()
    print(f"✓ 人員 ID={person_id} 已停用。")


if __name__ == "__main__":
    from database import init_db
    init_db()

    parser = argparse.ArgumentParser(description="人員 enrollment CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="列出所有人員")

    p_add = sub.add_parser("add", help="新增人員")
    p_add.add_argument("--name", required=True, help="姓名")
    p_add.add_argument("--dept", default="", help="部門")
    p_add.add_argument("--employee-id", default="", dest="employee_id", help="員工編號")

    p_enroll = sub.add_parser("enroll", help="人臉 enrollment")
    p_enroll.add_argument("--person-id", type=int, required=True)
    p_enroll.add_argument("--samples", type=int, default=5)
    p_enroll.add_argument("--camera", type=int, default=0)

    p_rm = sub.add_parser("remove", help="停用人員")
    p_rm.add_argument("--person-id", type=int, required=True)

    args = parser.parse_args()

    if args.cmd == "list":
        cmd_list()
    elif args.cmd == "add":
        cmd_add(args.name, args.dept, args.employee_id)
    elif args.cmd == "enroll":
        cmd_enroll(args.person_id, args.camera, args.samples)
    elif args.cmd == "remove":
        cmd_remove(args.person_id)
    else:
        parser.print_help()

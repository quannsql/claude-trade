# -*- coding: utf-8 -*-
"""
reset_dashboard_db.py — Xóa sạch dữ liệu dashboard và đặt mốc reset.

Xóa 2 bảng: account_snapshots (equity curve) + fills (lịch sử lệnh),
sau đó ghi mốc reset_ms vào dashboard_meta để app KHÔNG nạp lại fills cũ
từ Hyperliquid API (API luôn trả ~200 fills gần nhất — không có mốc này
thì vài giây sau data cũ quay lại ngay).

Cách dùng:
    python reset_dashboard_db.py --dry-run   # chỉ xem số dòng, KHÔNG xóa
    python reset_dashboard_db.py             # hỏi xác nhận rồi xóa
    python reset_dashboard_db.py --yes       # xóa luôn không hỏi

DATABASE_URL đọc từ biến môi trường hoặc file .env cùng thư mục.
LƯU Ý: nếu DATABASE_URL trỏ tới Postgres Railway thì chạy từ máy local
sẽ xóa dữ liệu dashboard PRODUCTION (đúng mục đích làm mới).
"""
import os
import sqlite3
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
LOCAL_DB = BASE / "data" / "dashboard_history.sqlite3"
TABLES = ("account_snapshots", "fills")


def load_env() -> None:
    env_path = BASE / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def reset_postgres(db_url: str, reset_ms: str, dry_run: bool) -> None:
    import psycopg
    if db_url.startswith("postgres://"):
        db_url = "postgresql://" + db_url[len("postgres://"):]
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            for t in TABLES:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {t}")
                    n = cur.fetchone()[0]
                except Exception:
                    conn.rollback()
                    print(f"  [postgres] bảng {t}: chưa tồn tại — bỏ qua")
                    continue
                if dry_run:
                    print(f"  [postgres] bảng {t}: {n} dòng (dry-run, không xóa)")
                    continue
                cur.execute(f"DELETE FROM {t}")
                print(f"  [postgres] bảng {t}: đã xóa {n} dòng")
            if not dry_run:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS dashboard_meta (
                        key TEXT PRIMARY KEY, value TEXT NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    INSERT INTO dashboard_meta (key, value) VALUES ('reset_ms', %s)
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                    """,
                    (reset_ms,),
                )
                print(f"  [postgres] đặt mốc reset_ms = {reset_ms}")


def reset_sqlite(reset_ms: str, dry_run: bool) -> None:
    if not LOCAL_DB.exists():
        print("  [sqlite] không có file local — bỏ qua")
        return
    with sqlite3.connect(LOCAL_DB) as conn:
        for t in TABLES:
            try:
                n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            except Exception:
                print(f"  [sqlite] bảng {t}: chưa tồn tại — bỏ qua")
                continue
            if dry_run:
                print(f"  [sqlite] bảng {t}: {n} dòng (dry-run, không xóa)")
                continue
            conn.execute(f"DELETE FROM {t}")
            print(f"  [sqlite] bảng {t}: đã xóa {n} dòng")
        if not dry_run:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS dashboard_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT OR REPLACE INTO dashboard_meta (key, value) VALUES ('reset_ms', ?)",
                (reset_ms,),
            )
            print(f"  [sqlite] đặt mốc reset_ms = {reset_ms}")


def main() -> None:
    load_env()
    dry_run = "--dry-run" in sys.argv
    skip_confirm = "--yes" in sys.argv

    db_url = os.environ.get("DATABASE_URL", "").strip()
    print("=" * 60)
    print("RESET DASHBOARD DATA")
    print(f"  Postgres (DATABASE_URL): {'CÓ' if db_url else 'không có'}")
    print(f"  SQLite local:            {'CÓ' if LOCAL_DB.exists() else 'không có'}")
    print("=" * 60)

    if not db_url and not LOCAL_DB.exists():
        print("Không tìm thấy database nào để xóa.")
        return

    if not dry_run and not skip_confirm:
        try:
            ans = input("Xóa TOÀN BỘ dữ liệu dashboard? Gõ 'yes' để xác nhận: ")
        except EOFError:
            ans = ""
        if ans.strip().lower() != "yes":
            print("Hủy — không xóa gì.")
            return

    reset_ms = str(int(time.time() * 1000))

    if db_url:
        try:
            reset_postgres(db_url, reset_ms, dry_run)
        except Exception as exc:
            print(f"  [postgres] LỖI: {exc}")
            print("  (Nếu DB Railway không mở public network, chạy script này"
                  " bằng 'railway run python reset_dashboard_db.py --yes')")
    reset_sqlite(reset_ms, dry_run)

    if not dry_run:
        print()
        print("XONG. Dashboard sẽ trống và chỉ ghi nhận fills MỚI từ thời điểm này.")
        print("Không cần restart app (mốc reset được đọc lại trong tối đa 60 giây).")


if __name__ == "__main__":
    main()

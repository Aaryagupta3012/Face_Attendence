"""SQLite storage for students and attendance records."""
import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "attendance.db"


def get_conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_no    TEXT UNIQUE NOT NULL,
                name       TEXT NOT NULL,
                department TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attendance (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
                date       TEXT NOT NULL,
                time       TEXT NOT NULL,
                confidence REAL,
                UNIQUE(student_id, date)
            );
            """
        )


# ---------- students ----------
def add_student(roll_no: str, name: str, department: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO students (roll_no, name, department, created_at) VALUES (?, ?, ?, ?)",
            (roll_no.strip(), name.strip(), department.strip(), datetime.now().isoformat(timespec="seconds")),
        )
        return cur.lastrowid


def get_students() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query("SELECT id, roll_no, name, department, created_at FROM students ORDER BY roll_no", conn)


def get_student(student_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
        return dict(row) if row else None


def delete_student(student_id: int) -> None:
    with get_conn() as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
        conn.execute("DELETE FROM students WHERE id = ?", (student_id,))


# ---------- attendance ----------
def mark_attendance(student_id: int, confidence: float | None = None) -> bool:
    """Mark today's attendance. Returns False if already marked today."""
    now = datetime.now()
    with get_conn() as conn:
        try:
            conn.execute(
                "INSERT INTO attendance (student_id, date, time, confidence) VALUES (?, ?, ?, ?)",
                (student_id, now.date().isoformat(), now.strftime("%H:%M:%S"), confidence),
            )
            return True
        except sqlite3.IntegrityError:
            return False


def get_attendance(start: date | None = None, end: date | None = None) -> pd.DataFrame:
    query = """
        SELECT a.date, a.time, s.roll_no, s.name, s.department, a.confidence
        FROM attendance a JOIN students s ON s.id = a.student_id
        WHERE 1=1
    """
    params: list = []
    if start:
        query += " AND a.date >= ?"
        params.append(start.isoformat())
    if end:
        query += " AND a.date <= ?"
        params.append(end.isoformat())
    query += " ORDER BY a.date DESC, a.time DESC"
    with get_conn() as conn:
        return pd.read_sql_query(query, conn, params=params)


def attendance_summary() -> pd.DataFrame:
    """Per-student: days present, last seen, attendance % over all recorded days."""
    with get_conn() as conn:
        total_days = conn.execute("SELECT COUNT(DISTINCT date) FROM attendance").fetchone()[0] or 0
        df = pd.read_sql_query(
            """
            SELECT s.roll_no, s.name, s.department,
                   COUNT(a.id) AS days_present,
                   MAX(a.date) AS last_seen
            FROM students s LEFT JOIN attendance a ON a.student_id = s.id
            GROUP BY s.id ORDER BY s.roll_no
            """,
            conn,
        )
    df["attendance_pct"] = (df["days_present"] / total_days * 100).round(1) if total_days else 0.0
    return df

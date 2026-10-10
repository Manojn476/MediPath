"""
Database schema for MediPath.

SQLite tables for prescription tracking, medication scheduling, and adherence logging.
"""
import sqlite3
from pathlib import Path
from typing import Optional
from datetime import datetime


# Database file path
DB_PATH = Path(__file__).parent / "mediplus.db"


def get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory for dict-like access."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Initialize the database schema.

    Creates tables if they don't exist:
    - prescriptions: Stores OCR-extracted prescription data
    - medication_schedule: Stores reminder schedules for confirmed prescriptions
    - adherence_logs: Tracks when medications are taken or missed
    """
    conn = get_connection()
    cursor = conn.cursor()

    # ── Prescriptions table ──────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prescriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            -- Core prescription fields
            drug_name TEXT NOT NULL,
            dose TEXT,
            frequency TEXT,
            duration TEXT,

            -- Metadata
            status TEXT NOT NULL DEFAULT 'pending',  -- 'pending', 'confirmed', 'archived'
            confidence_score REAL,
            raw_ocr_text TEXT,

            -- Drug matching metadata (JSON or separate fields)
            drug_candidate TEXT,
            drug_edit_distance INTEGER,
            drug_match_confidence REAL,

            -- Warnings from OCR extraction (stored as JSON string)
            warnings TEXT,

            -- Timestamps
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confirmed_at TIMESTAMP,

            -- Optional: User/patient info for future multi-user support
            user_id TEXT,

            -- Optional: Image path reference
            image_path TEXT,

            -- Notes field for manual edits/corrections
            notes TEXT
        )
    """)

    # ── Medication Schedule table ────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS medication_schedule (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            -- Link to prescription
            prescription_id INTEGER NOT NULL,

            -- Schedule details
            time_of_day TEXT NOT NULL,  -- e.g., '08:00', '14:00', '20:00'
            frequency_label TEXT,       -- e.g., 'Morning', 'Afternoon', 'Evening'

            -- Start and end dates for the schedule
            start_date DATE NOT NULL,
            end_date DATE,

            -- Active/inactive flag (for pausing reminders)
            is_active INTEGER DEFAULT 1,

            -- Timestamps
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (prescription_id) REFERENCES prescriptions(id)
        )
    """)

    # ── Adherence Logs table ─────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS adherence_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            -- Link to prescription and schedule
            prescription_id INTEGER NOT NULL,
            schedule_id INTEGER NOT NULL,

            -- When the medication was supposed to be taken
            scheduled_time TIMESTAMP NOT NULL,

            -- When it was actually taken (NULL if missed)
            taken_at TIMESTAMP,

            -- Status: 'taken', 'missed', 'skipped', 'snoozed'
            status TEXT NOT NULL DEFAULT 'missed',

            -- Optional notes from user
            notes TEXT,

            -- How late/early (in minutes, negative = early, positive = late)
            delay_minutes INTEGER,

            -- Timestamps
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (prescription_id) REFERENCES prescriptions(id),
            FOREIGN KEY (schedule_id) REFERENCES medication_schedule(id)
        )
    """)

    # Create indexes for common queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_prescriptions_status
        ON prescriptions(status)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_prescriptions_user
        ON prescriptions(user_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_schedule_prescription
        ON medication_schedule(prescription_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_schedule_active
        ON medication_schedule(is_active)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_adherence_prescription
        ON adherence_logs(prescription_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_adherence_scheduled
        ON adherence_logs(scheduled_time)
    """)

    # ── Journal Entries table ────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS journal_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            -- User identifier
            user_id TEXT NOT NULL,

            -- Journal content
            text TEXT NOT NULL,

            -- Sentiment analysis
            sentiment_score REAL,  -- Range: -1.0 (negative) to +1.0 (positive)
            sentiment_label TEXT,   -- 'positive', 'neutral', 'negative'

            -- Timestamps
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            -- Optional metadata
            word_count INTEGER,
            language TEXT
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_journal_user
        ON journal_entries(user_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_journal_created
        ON journal_entries(created_at)
    """)

    conn.commit()
    conn.close()

    print(f"[DB] Database initialized at: {DB_PATH}")


def reset_db() -> None:
    """Drop all tables and reinitialize (useful for testing)."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS journal_entries")
    cursor.execute("DROP TABLE IF EXISTS adherence_logs")
    cursor.execute("DROP TABLE IF EXISTS medication_schedule")
    cursor.execute("DROP TABLE IF EXISTS prescriptions")

    conn.commit()
    conn.close()

    print("[DB] Database reset complete")
    init_db()


def main():
    """Initialize the database."""
    print("=" * 70)
    print("Initializing MediPath Database")
    print("=" * 70 + "\n")

    init_db()

    # Show table info
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table'
        ORDER BY name
    """)
    tables = cursor.fetchall()

    print("\nTables created:")
    for table in tables:
        print(f"  - {table['name']}")

        # Show column info
        cursor.execute(f"PRAGMA table_info({table['name']})")
        columns = cursor.fetchall()
        for col in columns:
            nullable = "NULL" if col['notnull'] == 0 else "NOT NULL"
            default = f"DEFAULT {col['dflt_value']}" if col['dflt_value'] else ""
            print(f"      {col['name']:<25} {col['type']:<15} {nullable:<10} {default}")
        print()

    conn.close()
    print("=" * 70)


if __name__ == "__main__":
    main()

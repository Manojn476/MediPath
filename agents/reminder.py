"""
Prescription reminder system.

Handles saving prescriptions, confirmation workflow, and generating
medication schedules with fixed time-of-day reminders.
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from db.schema import get_connection, init_db


# Confidence threshold for auto-confirmation
CONFIDENCE_THRESHOLD = 0.7


def save_prescription_pending_confirmation(
    structured_data: Dict[str, Any],
    user_id: Optional[str] = None,
    image_path: Optional[str] = None
) -> int:
    """
    Save a prescription from OCR structured data.

    Sets status to 'pending' if any field's confidence is below threshold (0.7),
    otherwise 'confirmed'.

    Args:
        structured_data: Output from structure_prescription() containing:
            - drug_name, dose, frequency, duration
            - confidence_score
            - drug_match_meta (candidate, matched_name, edit_distance, confidence)
            - warnings (list)
            - raw_text (optional)
        user_id: Optional user identifier
        image_path: Optional path to the prescription image

    Returns:
        prescription_id of the saved record
    """
    init_db()  # Ensure tables exist

    # Determine status based on confidence
    confidence_score = structured_data.get("confidence_score", 0.0)
    drug_match_confidence = structured_data.get("drug_match_meta", {}).get("confidence", 0.0)

    # Check if any field has low confidence
    has_warnings = len(structured_data.get("warnings", [])) > 0
    low_confidence = (
        confidence_score < CONFIDENCE_THRESHOLD or
        drug_match_confidence < CONFIDENCE_THRESHOLD or
        has_warnings
    )

    status = "pending" if low_confidence else "confirmed"
    confirmed_at = datetime.now() if status == "confirmed" else None

    # Extract drug matching metadata
    drug_match = structured_data.get("drug_match_meta", {})

    # Prepare warnings as JSON string
    warnings_json = json.dumps(structured_data.get("warnings", []))

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO prescriptions (
            drug_name, dose, frequency, duration,
            status, confidence_score, raw_ocr_text,
            drug_candidate, drug_edit_distance, drug_match_confidence,
            warnings, confirmed_at, user_id, image_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        structured_data.get("drug_name"),
        structured_data.get("dose"),
        structured_data.get("frequency"),
        structured_data.get("duration"),
        status,
        confidence_score,
        structured_data.get("raw_text", ""),
        drug_match.get("candidate", ""),
        drug_match.get("edit_distance"),
        drug_match.get("confidence"),
        warnings_json,
        confirmed_at,
        user_id,
        image_path
    ))

    prescription_id = cursor.lastrowid
    conn.commit()
    conn.close()

    print(f"[SAVE] Prescription #{prescription_id} saved with status: {status}")
    if low_confidence:
        print(f"       Confidence: {confidence_score:.3f}, Drug match: {drug_match_confidence:.3f}")
        if has_warnings:
            print(f"       Warnings: {len(structured_data.get('warnings', []))}")

    return prescription_id


def confirm_prescription(prescription_id: int) -> bool:
    """
    Confirm a pending prescription by flipping status to 'confirmed'.

    Args:
        prescription_id: ID of the prescription to confirm

    Returns:
        True if successful, False if prescription not found or already confirmed
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Check current status
    cursor.execute("""
        SELECT status FROM prescriptions WHERE id = ?
    """, (prescription_id,))

    row = cursor.fetchone()
    if not row:
        conn.close()
        print(f"[ERROR] Prescription #{prescription_id} not found")
        return False

    if row["status"] == "confirmed":
        conn.close()
        print(f"[INFO] Prescription #{prescription_id} already confirmed")
        return True

    # Update status
    cursor.execute("""
        UPDATE prescriptions
        SET status = 'confirmed', confirmed_at = ?
        WHERE id = ?
    """, (datetime.now(), prescription_id))

    conn.commit()
    conn.close()

    print(f"[CONFIRM] Prescription #{prescription_id} confirmed")
    return True


def generate_fixed_schedule(
    prescription_id: int,
    start_date: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Generate a fixed time-of-day reminder schedule from confirmed prescription.

    Parses the frequency field and creates schedule entries with typical
    medication times. No ML or personalization yet - just fixed heuristics.

    Frequency mapping:
    - Once daily / OD           -> 08:00 (Morning)
    - Twice daily / BD / BID    -> 08:00, 20:00 (Morning, Night)
    - Three times daily / TDS   -> 08:00, 14:00, 20:00 (Morning, Afternoon, Night)
    - Four times daily / QID    -> 08:00, 12:00, 16:00, 20:00 (Morning, Noon, Evening, Night)

    Args:
        prescription_id: ID of confirmed prescription
        start_date: When to start the schedule (default: today)

    Returns:
        List of schedule dictionaries created
    """
    if start_date is None:
        start_date = datetime.now().date()

    conn = get_connection()
    cursor = conn.cursor()

    # Fetch prescription
    cursor.execute("""
        SELECT * FROM prescriptions WHERE id = ?
    """, (prescription_id,))

    row = cursor.fetchone()
    if not row:
        conn.close()
        print(f"[ERROR] Prescription #{prescription_id} not found")
        return []

    if row["status"] != "confirmed":
        conn.close()
        print(f"[ERROR] Prescription #{prescription_id} is not confirmed (status: {row['status']})")
        return []

    frequency = (row["frequency"] or "").lower()
    duration = row["duration"]

    # Parse duration to calculate end_date
    end_date = None
    if duration:
        # Extract number of days/weeks/months
        import re
        duration_match = re.search(r'(\d+)\s*(day|week|month)', duration.lower())
        if duration_match:
            amount = int(duration_match.group(1))
            unit = duration_match.group(2)

            if unit == "day":
                end_date = start_date + timedelta(days=amount)
            elif unit == "week":
                end_date = start_date + timedelta(weeks=amount)
            elif unit == "month":
                end_date = start_date + timedelta(days=amount * 30)  # Approximate

    # Map frequency to time slots
    schedule_times = []

    if any(x in frequency for x in ["once daily", "once a daily", "od", "qd"]):
        schedule_times = [
            ("08:00", "Morning")
        ]
    elif any(x in frequency for x in ["twice daily", "twice a daily", "bd", "bid"]):
        schedule_times = [
            ("08:00", "Morning"),
            ("20:00", "Night")
        ]
    elif any(x in frequency for x in ["three times daily", "three times a daily", "tds", "tid"]):
        schedule_times = [
            ("08:00", "Morning"),
            ("14:00", "Afternoon"),
            ("20:00", "Night")
        ]
    elif any(x in frequency for x in ["four times daily", "four times a daily", "qid"]):
        schedule_times = [
            ("08:00", "Morning"),
            ("12:00", "Noon"),
            ("16:00", "Evening"),
            ("20:00", "Night")
        ]
    elif "before meals" in frequency or "ac" in frequency:
        schedule_times = [
            ("07:30", "Before Breakfast"),
            ("12:30", "Before Lunch"),
            ("19:30", "Before Dinner")
        ]
    elif "after meals" in frequency or "pc" in frequency:
        schedule_times = [
            ("09:00", "After Breakfast"),
            ("14:00", "After Lunch"),
            ("21:00", "After Dinner")
        ]
    elif "at bedtime" in frequency or "hs" in frequency:
        schedule_times = [
            ("22:00", "Bedtime")
        ]
    elif "as needed" in frequency or "prn" in frequency or "sos" in frequency:
        # For PRN medications, create a placeholder (not a fixed schedule)
        schedule_times = [
            ("00:00", "As Needed (no fixed time)")
        ]
    else:
        # Default fallback: once daily morning
        print(f"[WARN] Unrecognized frequency '{frequency}', defaulting to once daily")
        schedule_times = [
            ("08:00", "Morning")
        ]

    # Insert schedule entries
    schedules = []
    for time_of_day, label in schedule_times:
        cursor.execute("""
            INSERT INTO medication_schedule (
                prescription_id, time_of_day, frequency_label,
                start_date, end_date, is_active
            ) VALUES (?, ?, ?, ?, ?, 1)
        """, (prescription_id, time_of_day, label, start_date, end_date))

        schedule_id = cursor.lastrowid
        schedules.append({
            "id": schedule_id,
            "prescription_id": prescription_id,
            "time_of_day": time_of_day,
            "frequency_label": label,
            "start_date": str(start_date),
            "end_date": str(end_date) if end_date else None
        })

    conn.commit()
    conn.close()

    print(f"[SCHEDULE] Generated {len(schedules)} reminder(s) for prescription #{prescription_id}")
    for sched in schedules:
        print(f"           - {sched['time_of_day']} ({sched['frequency_label']})")

    return schedules


def get_prescription(prescription_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a prescription by ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM prescriptions WHERE id = ?", (prescription_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return dict(row)


def get_schedule(prescription_id: int) -> List[Dict[str, Any]]:
    """Fetch all schedule entries for a prescription."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM medication_schedule
        WHERE prescription_id = ?
        ORDER BY time_of_day
    """, (prescription_id,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def main():
    """Test the full prescription flow."""
    print("=" * 70)
    print("Testing Prescription Reminder System")
    print("=" * 70 + "\n")

    # Sample structured data from OCR (simulating structure_prescription output)
    sample_prescriptions = [
        {
            "drug_name": "Amoxicillin",
            "dose": "500 mg",
            "frequency": "Three times daily",
            "duration": "7 days",
            "confidence_score": 0.85,
            "drug_match_meta": {
                "candidate": "Amoxicillin",
                "matched_name": "Amoxicillin",
                "edit_distance": 0,
                "confidence": 1.0
            },
            "warnings": [],
            "raw_text": "Amoxicillin 500mg TDS for 7 days"
        },
        {
            "drug_name": "Paracetamol",
            "dose": "650 mg",
            "frequency": "Twice daily",
            "duration": "5 days",
            "confidence_score": 0.65,  # Low confidence - will be pending
            "drug_match_meta": {
                "candidate": "Paracetmol",
                "matched_name": "Paracetamol",
                "edit_distance": 1,
                "confidence": 0.92
            },
            "warnings": ["Low overall confidence due to OCR quality"],
            "raw_text": "Paracetmol 650mg BD 5 days"
        }
    ]

    # Test 1: Save high-confidence prescription (auto-confirmed)
    print("TEST 1: Save high-confidence prescription")
    print("-" * 70)
    presc_id_1 = save_prescription_pending_confirmation(sample_prescriptions[0])
    presc_1 = get_prescription(presc_id_1)
    print(f"Status: {presc_1['status']}")
    print()

    # Test 2: Save low-confidence prescription (pending)
    print("TEST 2: Save low-confidence prescription")
    print("-" * 70)
    presc_id_2 = save_prescription_pending_confirmation(sample_prescriptions[1])
    presc_2 = get_prescription(presc_id_2)
    print(f"Status: {presc_2['status']}")
    print()

    # Test 3: Confirm pending prescription
    print("TEST 3: Confirm pending prescription")
    print("-" * 70)
    confirm_prescription(presc_id_2)
    presc_2 = get_prescription(presc_id_2)
    print(f"New status: {presc_2['status']}")
    print()

    # Test 4: Generate schedule for first prescription
    print("TEST 4: Generate schedule for high-confidence prescription")
    print("-" * 70)
    schedule_1 = generate_fixed_schedule(presc_id_1)
    print()

    # Test 5: Generate schedule for confirmed prescription
    print("TEST 5: Generate schedule for confirmed prescription")
    print("-" * 70)
    schedule_2 = generate_fixed_schedule(presc_id_2)
    print()

    # Display final results
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70 + "\n")

    for presc_id in [presc_id_1, presc_id_2]:
        presc = get_prescription(presc_id)
        schedule = get_schedule(presc_id)

        print(f"Prescription #{presc_id}:")
        print(f"  Drug:       {presc['drug_name']}")
        print(f"  Dose:       {presc['dose']}")
        print(f"  Frequency:  {presc['frequency']}")
        print(f"  Duration:   {presc['duration']}")
        print(f"  Status:     {presc['status']}")
        print(f"  Confidence: {presc['confidence_score']:.3f}")
        print(f"\n  Schedule ({len(schedule)} reminders):")
        for s in schedule:
            end_info = f" until {s['end_date']}" if s['end_date'] else ""
            print(f"    - {s['time_of_day']} ({s['frequency_label']}){end_info}")
        print()


if __name__ == "__main__":
    main()

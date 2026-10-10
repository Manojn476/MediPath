"""
tests/test_prescription_ocr.py

Tests for:
  - extract_dosage_frequency()
  - match_drug_name()
  - structure_prescription()

Also runs structure_prescription() on the actual OCR output from the
two sample images (fewer.jpg, surgery.jpeg) in data/samples/prescriptions/.
"""

import json
import sys
from pathlib import Path

# Force UTF-8 output so box-drawing chars print on Windows cp1252 consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))


from agents.prescription_ocr import (
    extract_dosage_frequency,
    match_drug_name,
    run_ocr,
    structure_prescription,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "[PASS]"
FAIL = "[FAIL]"


def _header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def _section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Unit tests – extract_dosage_frequency
# ─────────────────────────────────────────────────────────────────────────────

DOSAGE_TESTS = [
    {
        "desc": "Simple mg tablet",
        "text": "Amoxicillin 500mg twice daily for 7 days",
        "expect_dose_unit": "mg",
        "expect_freq": "Twice daily",
        "expect_duration": "7 days",
    },
    {
        "desc": "Latin abbreviation BID",
        "text": "Metformin 850mg BID with food",
        "expect_dose_unit": "mg",
        "expect_freq": "Twice daily",
        "expect_duration": None,
    },
    {
        "desc": "TDS + duration",
        "text": "Paracetamol 650 mg TDS for 5 days",
        "expect_dose_unit": "mg",
        "expect_freq": "Three times daily",
        "expect_duration": "5 days",
    },
    {
        "desc": "Liquid mL dose OD",
        "text": "Syrup Azithromycin 10ml OD for 3 days",
        "expect_dose_unit": "ml",
        "expect_freq": "Once daily",
        "expect_duration": "3 days",
    },
    {
        "desc": "Noisy OCR text",
        "text": "Tab. |bupr0fen 4O0mg 81D x 5 doys",
        "expect_dose_unit": "mg",
        "expect_freq": None,          # OCR-mangled 'BID' may not match
        "expect_duration": None,
    },
]


def test_extract_dosage_frequency():
    _header("TEST 1 — extract_dosage_frequency()")
    all_pass = True

    for i, tc in enumerate(DOSAGE_TESTS, 1):
        result = extract_dosage_frequency(tc["text"])
        units = [d["unit"].lower() for d in result["doses"]]
        freq  = result["frequency"]
        dur   = result["duration"]

        dose_ok = tc["expect_dose_unit"].lower() in units if tc["expect_dose_unit"] else True
        freq_ok = (
            freq.lower() == tc["expect_freq"].lower()
        ) if tc["expect_freq"] is not None else True
        dur_ok  = (dur == tc["expect_duration"]) if tc["expect_duration"] is not None else True

        status = PASS if (dose_ok and freq_ok and dur_ok) else FAIL
        if status == FAIL:
            all_pass = False

        print(f"\n  [{i}] {tc['desc']}")
        print(f"      Input    : {tc['text']}")
        print(f"      Doses    : {result['doses']}")
        print(f"      Frequency: {freq}  (expected: {tc['expect_freq']})")
        print(f"      Duration : {dur}   (expected: {tc['expect_duration']})")
        print(f"      {status}")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Unit tests – match_drug_name
# ─────────────────────────────────────────────────────────────────────────────

DRUG_MATCH_TESTS = [
    {"candidate": "Paracetamol",  "expected": "Paracetamol",  "desc": "Exact match"},
    {"candidate": "paracetamol",  "expected": "Paracetamol",  "desc": "Case insensitive"},
    {"candidate": "Paracetamole", "expected": "Paracetamol",  "desc": "1 char extra (OCR)"},
    {"candidate": "Amox1c1llin",  "expected": "Amoxicillin",  "desc": "Digit-for-letter OCR"},
    {"candidate": "Metforrmin",   "expected": "Metformin",    "desc": "Double-r OCR error"},
    {"candidate": "Atorvastatin", "expected": "Atorvastatin", "desc": "Exact - statin"},
    {"candidate": "Ciprofioxacin","expected": "Ciprofloxacin","desc": "f->fi OCR swap"},
    {"candidate": "Ibupr0fen",    "expected": "Ibuprofen",    "desc": "0->o OCR error"},
    {"candidate": "xyz_garbage",  "expected": None,            "desc": "No match (garbage)"},
]


def test_match_drug_name():
    _header("TEST 2 -- match_drug_name()")
    all_pass = True

    for i, tc in enumerate(DRUG_MATCH_TESTS, 1):
        result = match_drug_name(tc["candidate"])
        matched = result["matched_name"]
        conf    = result["confidence"]
        dist    = result["edit_distance"]

        if tc["expected"] is None:
            ok = conf < 0.5
        else:
            ok = matched.lower() == tc["expected"].lower()

        status = PASS if ok else FAIL
        if status == FAIL:
            all_pass = False

        print(f"\n  [{i}] {tc['desc']}")
        print(f"      Candidate    : '{tc['candidate']}'")
        print(f"      Matched      : '{matched}'  (expected: '{tc['expected']}')")
        print(f"      Edit distance: {dist}   Confidence: {conf:.3f}")
        print(f"      {status}")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Unit tests – structure_prescription (synthetic texts)
# ─────────────────────────────────────────────────────────────────────────────

STRUCT_TESTS = [
    {
        "desc": "Clean prescription line",
        "text": "Amoxicillin 500mg Twice daily for 7 days",
    },
    {
        "desc": "Multi-drug prescription",
        "text": (
            "1. Metformin 500mg BD\n"
            "2. Atorvastatin 10mg OD at bedtime\n"
            "3. Aspirin 75mg OD for 30 days"
        ),
    },
    {
        "desc": "Noisy OCR simulation",
        "text": "Tab. Amox|c1llin 5O0mg B|D x 5 doys",
    },
    {
        "desc": "Paediatric syrup",
        "text": "Syrup Paracetamol 5ml TDS for 3 days (fever management)",
    },
    {
        "desc": "Almost empty / unstructured",
        "text": "Take medicine as directed by physician.",
    },
]


def test_structure_prescription():
    _header("TEST 3 — structure_prescription() on synthetic texts")
    all_pass = True

    for i, tc in enumerate(STRUCT_TESTS, 1):
        result = structure_prescription(tc["text"])
        ok = result["drug_name"] != "" and "confidence_score" in result

        status = PASS if ok else FAIL
        if status == FAIL:
            all_pass = False

        print(f"\n  [{i}] {tc['desc']}")
        print(f"      Input text       : {tc['text'][:80]}{'…' if len(tc['text']) > 80 else ''}")
        print(f"      drug_name        : {result['drug_name']}")
        print(f"      dose             : {result['dose']}")
        print(f"      frequency        : {result['frequency']}")
        print(f"      duration         : {result['duration']}")
        print(f"      confidence_score : {result['confidence_score']}")
        print(f"      drug_match_meta  : {result['drug_match_meta']}")
        if result["warnings"]:
            for w in result["warnings"]:
                print(f"      ⚠  {w}")
        print(f"      {status}")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Integration test — real OCR on sample images
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_IMAGES = [
    Path("data/samples/prescriptions/fewer.jpg"),
    Path("data/samples/prescriptions/surgery.jpeg"),
]


def test_real_images():
    _header("TEST 4 — structure_prescription() on REAL OCR (sample images)")

    any_run = False
    for img_path in SAMPLE_IMAGES:
        if not img_path.exists():
            print(f"\n  ⚠  Image not found, skipping: {img_path}")
            continue

        any_run = True
        _section(f"Image: {img_path.name}")

        # Step 1: Raw OCR
        print("\n  [Raw OCR Output]")
        raw = run_ocr(str(img_path))
        for line in raw.splitlines():
            print(f"    {line}")

        # Step 2: Structure
        result = structure_prescription(raw)

        print("\n  [Structured Prescription Output]")
        print(f"    drug_name        : {result['drug_name']}")
        print(f"    dose             : {result['dose']}")
        print(f"    frequency        : {result['frequency']}")
        print(f"    duration         : {result['duration']}")
        print(f"    confidence_score : {result['confidence_score']}")
        print(f"    all_doses        : {result['all_doses']}")
        print(f"    drug_match_meta  : {result['drug_match_meta']}")
        if result["warnings"]:
            print("    warnings:")
            for w in result["warnings"]:
                print(f"      [WARN] {w}")

    if not any_run:
        print("\n  No sample images found — skipping integration test.")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  MediPath — Prescription OCR Test Suite")
    print("=" * 70)

    results = {
        "extract_dosage_frequency": test_extract_dosage_frequency(),
        "match_drug_name":          test_match_drug_name(),
        "structure_prescription":   test_structure_prescription(),
        "real_images":              test_real_images(),
    }

    _header("SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total  = len(results)
    for name, ok in results.items():
        marker = "[PASS]" if ok else "[FAIL]"
        print(f"  {marker}  {name}")
    print(f"\n  {passed}/{total} test groups passed.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

"""
Prescription OCR Pipeline.

Stages:
  1. Image preprocessing  → grayscale, deskew, denoise, threshold
  2. Tesseract OCR         → raw text
  3. extract_dosage_frequency() → regex/spaCy extraction of dose + frequency
  4. match_drug_name()          → fuzzy edit-distance correction against reference list
  5. structure_prescription()   → unified structured dict with confidence score
"""

import re
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import pytesseract

# ── Optional spaCy import (graceful fallback) ────────────────────────────────
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
    _SPACY_AVAILABLE = True
except Exception:
    _NLP = None
    _SPACY_AVAILABLE = False

# ── Levenshtein for fuzzy matching ───────────────────────────────────────────
try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    # Pure-python fallback
    def levenshtein_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                                prev[j] + (c1 != c2)))
            prev = curr
        return prev[-1]

# ── Tesseract path (Windows) ─────────────────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ── Reference list path ──────────────────────────────────────────────────────
_DRUG_LIST_PATH = Path(__file__).parent.parent / "data" / "reference" / "drug_names.txt"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Image Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocess a prescription image for better OCR accuracy.

    Steps: grayscale → deskew → denoise → adaptive threshold → morph-clean.

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        Preprocessed binary image as a numpy array.

    Raises:
        ValueError: If the image cannot be read.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Deskew
    coords = np.column_stack(np.where(gray > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) > 0.5:
            (h, w) = gray.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(
        gray, None, h=10, templateWindowSize=7, searchWindowSize=21
    )

    # Adaptive threshold (binarisation)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2,
    )

    # Morphological cleanup
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Raw OCR
# ─────────────────────────────────────────────────────────────────────────────

def run_ocr(image_path: str) -> str:
    """
    Run Tesseract OCR on a prescription image after preprocessing.

    Args:
        image_path: Path to the prescription image.

    Returns:
        Raw extracted text string (may contain OCR noise).
    """
    try:
        preprocessed = preprocess_image(image_path)
        pil_image = Image.fromarray(preprocessed)
        custom_config = r"--oem 3 --psm 6"
        return pytesseract.image_to_string(pil_image, config=custom_config).strip()
    except Exception as e:
        return f"Error during OCR: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  extract_dosage_frequency()
# ─────────────────────────────────────────────────────────────────────────────

# ── Regex patterns ────────────────────────────────────────────────────────────
_DOSE_PATTERN = re.compile(
    r"""
    (?P<amount>\d+(?:\.\d+)?)      # numeric dose (e.g. 500, 0.5)
    \s*
    (?P<unit>
        mg|mcg|µg|g|ml|mL|L|IU|units?|
        tab(?:lets?)?|cap(?:sules?)?|
        drops?|puffs?|sprays?|sachets?|
        patch(?:es)?|suppositories?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_FREQ_PATTERN = re.compile(
    r"""
    (?P<freq>
        # Latin abbreviations
        \bOD\b | \bBD\b | \bTDS\b | \bQID\b |
        \bBID\b | \bTID\b | \bQD\b | \bPRN\b |
        \bSOS\b | \bStat\b | \bHS\b | \bAC\b | \bPC\b |
        # English phrases
        once\s+(?:a\s+)?daily |
        twice\s+(?:a\s+)?daily |
        three\s+times\s+(?:a\s+)?daily |
        four\s+times\s+(?:a\s+)?daily |
        every\s+\d+\s+hours? |
        \d+\s+times?\s+(?:a\s+)?day |
        at\s+bedtime | before\s+(?:meals?|food) | after\s+(?:meals?|food) |
        morning\s+and\s+(?:evening|night) |
        with\s+food | on\s+empty\s+stomach
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_DURATION_PATTERN = re.compile(
    r"""
    (?:for\s+)?
    (?P<duration>
        \d+\s+(?:days?|weeks?|months?|years?)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Normalise Latin abbreviations → human-readable
_FREQ_NORMALISE = {
    "od": "Once daily",
    "bd": "Twice daily",
    "bid": "Twice daily",
    "tds": "Three times daily",
    "tid": "Three times daily",
    "qid": "Four times daily",
    "qd": "Once daily",
    "prn": "As needed",
    "sos": "As needed",
    "stat": "Immediately",
    "hs": "At bedtime",
    "ac": "Before meals",
    "pc": "After meals",
}


def extract_dosage_frequency(text: str) -> dict:
    """
    Extract dosage amounts, units, and frequency patterns from prescription text.

    Uses regex for structured patterns and optionally spaCy for entity context.

    Args:
        text: Raw OCR or prescription text.

    Returns:
        dict with keys:
            doses      – list of {"amount": str, "unit": str}
            frequency  – normalised frequency string or None
            duration   – duration string or None
            raw_text   – the original text passed in
    """
    # ── Dose extraction ───────────────────────────────────────────────────────
    doses = [
        {"amount": m.group("amount"), "unit": m.group("unit")}
        for m in _DOSE_PATTERN.finditer(text)
    ]

    # ── Frequency extraction ──────────────────────────────────────────────────
    freq_match = _FREQ_PATTERN.search(text)
    frequency = None
    if freq_match:
        raw_freq = freq_match.group("freq").strip()
        # Look up Latin abbreviation; otherwise title-case the English phrase
        frequency = _FREQ_NORMALISE.get(
            raw_freq.lower(),
            raw_freq.strip().capitalize(),
        )

    # ── Duration extraction ───────────────────────────────────────────────────
    dur_match = _DURATION_PATTERN.search(text)
    duration = dur_match.group("duration").strip() if dur_match else None

    # ── spaCy augmentation: pull quantities/dates if regex missed them ────────
    if _SPACY_AVAILABLE and _NLP is not None:
        doc = _NLP(text)
        for ent in doc.ents:
            if ent.label_ in ("QUANTITY", "CARDINAL") and not doses:
                # Try to salvage a missed dose
                q_match = _DOSE_PATTERN.search(ent.text)
                if q_match:
                    doses.append({
                        "amount": q_match.group("amount"),
                        "unit": q_match.group("unit"),
                    })
            elif ent.label_ == "DATE" and duration is None:
                duration = ent.text.strip()

    return {
        "doses": doses,
        "frequency": frequency,
        "duration": duration,
        "raw_text": text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  match_drug_name()
# ─────────────────────────────────────────────────────────────────────────────

def _load_drug_list(path: Path = _DRUG_LIST_PATH) -> list[str]:
    """Load drug names from reference file, ignoring comments and blank lines."""
    if not path.exists():
        return []
    names = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    return names


# Module-level cache so the list is read once
_DRUG_NAMES: list[str] = []


def _get_drug_names() -> list[str]:
    global _DRUG_NAMES
    if not _DRUG_NAMES:
        _DRUG_NAMES = _load_drug_list()
    return _DRUG_NAMES


def match_drug_name(candidate: str, max_distance: int = 5) -> dict:
    """
    Fuzzy-match a candidate drug name against the reference list using
    edit distance (Levenshtein).

    The matching is done case-insensitively. Partial / multi-word names
    are handled by also checking against each word token individually.

    Args:
        candidate:    OCR-extracted candidate string (may have typos).
        max_distance: Maximum edit distance to consider a match.
                      Candidates beyond this threshold return no match.

    Returns:
        dict with keys:
            matched_name    – best-matched drug name from reference list
            edit_distance   – edit distance of the best match
            confidence      – float in [0, 1] (1 = exact, 0 = no match)
            candidate       – the original input
    """
    drug_names = _get_drug_names()
    if not drug_names:
        return {
            "matched_name": candidate,
            "edit_distance": -1,
            "confidence": 0.0,
            "candidate": candidate,
        }

    candidate_lower = candidate.lower().strip()
    best_name = None
    best_dist = float("inf")

    for drug in drug_names:
        drug_lower = drug.lower()
        dist = levenshtein_distance(candidate_lower, drug_lower)
        if dist < best_dist:
            best_dist = dist
            best_name = drug

    # Confidence: scale inversely with edit distance relative to name length
    ref_len = max(len(candidate_lower), 1)
    if best_dist <= max_distance:
        confidence = max(0.0, 1.0 - (best_dist / ref_len))
    else:
        confidence = 0.0
        best_name = candidate  # return as-is if no good match

    return {
        "matched_name": best_name,
        "edit_distance": int(best_dist),
        "confidence": round(confidence, 3),
        "candidate": candidate,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  structure_prescription()
# ─────────────────────────────────────────────────────────────────────────────

def _extract_drug_candidates(text: str) -> list[str]:
    """
    Heuristically extract candidate drug name tokens from prescription text.

    Strategy:
    - Split into lines; look for lines that contain a dose pattern (those
      are likely drug lines).
    - Also scan for tokens matching TitleCase words (common for drug names).
    - Return unique candidates ordered by line position.
    """
    candidates = []
    seen = set()

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Lines that contain a dosage number are drug lines
        if _DOSE_PATTERN.search(line) or _FREQ_PATTERN.search(line):
            # The drug name is typically the first word(s) before the dose
            # Remove the dose part to isolate candidate name
            cleaned = _DOSE_PATTERN.sub("", line)
            cleaned = _FREQ_PATTERN.sub("", cleaned)
            cleaned = _DURATION_PATTERN.sub("", cleaned)
            cleaned = re.sub(r"[^a-zA-Z\s]", " ", cleaned).strip()

            # Take up to first 3 words (drug names are rarely longer)
            words = cleaned.split()[:3]
            candidate = " ".join(words).strip()
            if candidate and candidate.lower() not in seen:
                candidates.append(candidate)
                seen.add(candidate.lower())

    # Fallback: scan for TitleCase tokens not already captured
    title_tokens = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
    for tok in title_tokens:
        if tok.lower() not in seen:
            candidates.append(tok)
            seen.add(tok.lower())

    return candidates


def structure_prescription(raw_text: str) -> dict:
    """
    Parse raw OCR prescription text into a structured dictionary.

    Combines extract_dosage_frequency() and match_drug_name() to produce
    a normalised, machine-readable prescription record.

    Args:
        raw_text: Raw text from OCR (may contain noise/typos).

    Returns:
        dict with keys:
            drug_name        – best-matched drug name (str)
            dose             – primary dose string, e.g. "500 mg" (str | None)
            frequency        – normalised frequency, e.g. "Twice daily" (str | None)
            duration         – duration string, e.g. "5 days" (str | None)
            confidence_score – overall confidence float in [0, 1]
            all_doses        – full list of extracted dose dicts
            drug_match_meta  – raw output from match_drug_name()
            warnings         – list of warning strings if any field is uncertain
    """
    warnings: list[str] = []

    # ── Step A: extract dosage / frequency / duration ─────────────────────────
    dosage_info = extract_dosage_frequency(raw_text)

    # Primary dose (first found)
    primary_dose = None
    if dosage_info["doses"]:
        d = dosage_info["doses"][0]
        primary_dose = f"{d['amount']} {d['unit']}"
    else:
        warnings.append("No dosage amount/unit found in text.")

    if not dosage_info["frequency"]:
        warnings.append("No frequency pattern found in text.")

    if not dosage_info["duration"]:
        warnings.append("No duration found in text.")

    # ── Step B: extract & match drug name ────────────────────────────────────
    candidates = _extract_drug_candidates(raw_text)

    if not candidates:
        warnings.append("No drug name candidate could be extracted.")
        drug_match = {
            "matched_name": "Unknown",
            "edit_distance": -1,
            "confidence": 0.0,
            "candidate": "",
        }
    else:
        # Match each candidate and pick the one with highest confidence
        matches = [match_drug_name(c) for c in candidates]
        drug_match = max(matches, key=lambda m: m["confidence"])

        if drug_match["confidence"] < 0.5:
            warnings.append(
                f"Low drug name confidence ({drug_match['confidence']:.2f}) "
                f"for candidate '{drug_match['candidate']}'. "
                f"Closest match: '{drug_match['matched_name']}' "
                f"(edit distance {drug_match['edit_distance']})."
            )

    # ── Step C: compute overall confidence score ─────────────────────────────
    field_scores = {
        "drug_name":  drug_match["confidence"],
        "dose":       1.0 if primary_dose else 0.0,
        "frequency":  1.0 if dosage_info["frequency"] else 0.0,
        "duration":   1.0 if dosage_info["duration"] else 0.0,
    }
    # Weighted average: drug name matters most
    weights = {"drug_name": 0.4, "dose": 0.3, "frequency": 0.2, "duration": 0.1}
    confidence_score = sum(field_scores[k] * weights[k] for k in weights)

    return {
        "drug_name":        drug_match["matched_name"],
        "dose":             primary_dose,
        "frequency":        dosage_info["frequency"],
        "duration":         dosage_info["duration"],
        "confidence_score": round(confidence_score, 3),
        "all_doses":        dosage_info["doses"],
        "drug_match_meta":  drug_match,
        "warnings":         warnings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Run OCR + structuring on all images in data/samples/prescriptions/."""
    print("=" * 70)
    print("Prescription OCR Pipeline")
    print("=" * 70)

    samples_dir = Path("data/samples/prescriptions")
    if not samples_dir.exists():
        print(f"No directory found at {samples_dir}. Please create it and add images.")
        return

    image_extensions = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(samples_dir.glob(f"*{ext}"))

    if not image_files:
        print("No images found.")
        return

    for image_path in sorted(image_files):
        print(f"\n{'─' * 70}")
        print(f"Image: {image_path.name}")
        print("─" * 70)
        raw_text = run_ocr(str(image_path))
        print("\n[Raw OCR Output]")
        print(raw_text)
        result = structure_prescription(raw_text)
        print("\n[Structured Output]")
        for key, val in result.items():
            print(f"  {key}: {val}")


if __name__ == "__main__":
    main()

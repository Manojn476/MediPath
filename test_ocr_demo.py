"""Quick demo of prescription OCR structured output."""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.prescription_ocr import structure_prescription

# Sample OCR text (simulating what Tesseract would extract)
sample_texts = [
    # Sample 1: Clear prescription
    """
    Dr. Smith Clinic
    Patient: John Doe

    Amoxicillin 500mg
    Take twice daily (BID)
    For 7 days
    """,

    # Sample 2: Prescription with typo (Paracetmol -> Paracetamol)
    """
    RX:
    Paracetmol 650 mg
    Three times daily
    Duration: 5 days
    """,

    # Sample 3: Latin abbreviations
    """
    Ibuprfen 400mg TDS
    PC (after meals)
    10 days
    """,

    # Sample 4: Complex prescription
    """
    Metformin 500mg tablets
    OD (once daily)
    Before breakfast
    Continue for 30 days
    """,
]

def main():
    print("=" * 70)
    print("Prescription OCR: Structured Output Demo")
    print("=" * 70 + "\n")

    for i, raw_text in enumerate(sample_texts, 1):
        print(f"\n{'─' * 70}")
        print(f"SAMPLE {i}")
        print("─" * 70)
        print("\n[Raw Text]")
        print(raw_text.strip())

        # Structure the prescription
        result = structure_prescription(raw_text)

        print("\n[Structured Output]")
        print(f"  Drug Name:        {result['drug_name']}")
        print(f"  Dose:             {result['dose']}")
        print(f"  Frequency:        {result['frequency']}")
        print(f"  Duration:         {result['duration']}")
        print(f"  Confidence Score: {result['confidence_score']:.3f}")

        if result['warnings']:
            print(f"\n  Warnings:")
            for warning in result['warnings']:
                print(f"    - {warning}")

        print(f"\n  Drug Match Details:")
        print(f"    Candidate:      {result['drug_match_meta']['candidate']}")
        print(f"    Matched:        {result['drug_match_meta']['matched_name']}")
        print(f"    Edit Distance:  {result['drug_match_meta']['edit_distance']}")
        print(f"    Match Confidence: {result['drug_match_meta']['confidence']:.3f}")

        if len(result['all_doses']) > 1:
            print(f"\n  All Doses Detected: {result['all_doses']}")

if __name__ == "__main__":
    main()

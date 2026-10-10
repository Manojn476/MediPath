"""
Mental health journal with sentiment analysis.

Provides journaling capabilities with sentiment tracking, trend analysis,
and professional resource recommendations.
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from db.schema import get_connection, init_db

# Try to import transformers for sentiment analysis
try:
    from transformers import pipeline
    _SENTIMENT_AVAILABLE = True
except ImportError:
    _SENTIMENT_AVAILABLE = False
    print("[WARN] transformers not available. Install with: pip install transformers torch")


# Global sentiment analyzer (lazy loaded)
_sentiment_analyzer = None


def get_sentiment_analyzer():
    """Get or initialize the sentiment analyzer (lazy loading)."""
    global _sentiment_analyzer

    if not _SENTIMENT_AVAILABLE:
        return None

    if _sentiment_analyzer is None:
        print("[INFO] Loading sentiment analysis model (multilingual)...")
        # Using a multilingual sentiment model that works across languages
        _sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=-1  # CPU
        )
        print("[INFO] Model loaded.")

    return _sentiment_analyzer


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of text using pretrained HuggingFace model.

    Args:
        text: Journal entry text

    Returns:
        Dict with sentiment_label, sentiment_score, and confidence
    """
    analyzer = get_sentiment_analyzer()

    if analyzer is None:
        # Fallback if model not available
        return {
            "sentiment_label": "neutral",
            "sentiment_score": 0.0,
            "confidence": 0.0
        }

    try:
        # Truncate long text (model has token limits)
        max_chars = 512
        text_truncated = text[:max_chars] if len(text) > max_chars else text

        # Run sentiment analysis
        result = analyzer(text_truncated)[0]

        # Model returns 1-5 star ratings, convert to sentiment
        # 1-2 stars = negative, 3 stars = neutral, 4-5 stars = positive
        label_mapping = {
            "1 star": ("negative", -1.0),
            "2 stars": ("negative", -0.5),
            "3 stars": ("neutral", 0.0),
            "4 stars": ("positive", 0.5),
            "5 stars": ("positive", 1.0),
        }

        raw_label = result["label"]
        sentiment_label, sentiment_score = label_mapping.get(
            raw_label, ("neutral", 0.0)
        )

        return {
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "confidence": result["score"]
        }

    except Exception as e:
        print(f"[ERROR] Sentiment analysis failed: {e}")
        return {
            "sentiment_label": "neutral",
            "sentiment_score": 0.0,
            "confidence": 0.0
        }


def add_entry(user_id: str, text: str) -> int:
    """
    Add a journal entry with sentiment analysis.

    Args:
        user_id: User identifier
        text: Journal entry text

    Returns:
        entry_id of the saved entry
    """
    init_db()  # Ensure tables exist

    # Analyze sentiment
    sentiment = analyze_sentiment(text)

    # Calculate metadata
    word_count = len(text.split())

    # Save to database
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO journal_entries (
            user_id, text, sentiment_score, sentiment_label, word_count
        ) VALUES (?, ?, ?, ?, ?)
    """, (
        user_id,
        text,
        sentiment["sentiment_score"],
        sentiment["sentiment_label"],
        word_count
    ))

    entry_id = cursor.lastrowid
    conn.commit()
    conn.close()

    print(f"[JOURNAL] Entry #{entry_id} saved for user {user_id}")
    print(f"          Sentiment: {sentiment['sentiment_label']} "
          f"(score: {sentiment['sentiment_score']:.2f})")

    return entry_id


def get_trend(user_id: str, window_days: int = 14) -> str:
    """
    Get sentiment trend description for a user over a time window.

    Returns plain, non-diagnostic language describing the trend.

    Args:
        user_id: User identifier
        window_days: Number of days to look back (default 14)

    Returns:
        Plain language trend description string
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Get entries from the past window_days
    cutoff_date = datetime.now() - timedelta(days=window_days)

    cursor.execute("""
        SELECT sentiment_score, created_at
        FROM journal_entries
        WHERE user_id = ?
        AND created_at >= ?
        ORDER BY created_at ASC
    """, (user_id, cutoff_date))

    entries = cursor.fetchall()
    conn.close()

    if not entries:
        return f"You haven't written any journal entries in the past {window_days} days."

    if len(entries) == 1:
        score = entries[0]["sentiment_score"]
        tone = "positive" if score > 0.2 else "neutral" if score > -0.2 else "negative"
        return f"You have one journal entry from the past {window_days} days with a {tone} tone."

    # Calculate rolling average and trend
    scores = [entry["sentiment_score"] for entry in entries]
    avg_score = sum(scores) / len(scores)

    # Split into first half and second half to detect trend
    mid_point = len(scores) // 2
    first_half_avg = sum(scores[:mid_point]) / len(scores[:mid_point]) if mid_point > 0 else avg_score
    second_half_avg = sum(scores[mid_point:]) / len(scores[mid_point:])

    trend_change = second_half_avg - first_half_avg

    # Build description
    num_entries = len(entries)

    # Overall tone
    if avg_score > 0.3:
        overall_tone = "mostly positive"
    elif avg_score > 0.1:
        overall_tone = "somewhat positive"
    elif avg_score > -0.1:
        overall_tone = "neutral"
    elif avg_score > -0.3:
        overall_tone = "somewhat negative"
    else:
        overall_tone = "mostly negative"

    # Trend direction
    if abs(trend_change) < 0.1:
        trend_desc = "stayed fairly consistent"
    elif trend_change > 0.2:
        trend_desc = "shifted notably more positive"
    elif trend_change > 0.05:
        trend_desc = "leaned a bit more positive"
    elif trend_change < -0.2:
        trend_desc = "shifted notably more negative"
    elif trend_change < -0.05:
        trend_desc = "leaned a bit more negative"
    else:
        trend_desc = "stayed relatively stable"

    description = (
        f"Over the past {window_days} days, you've written {num_entries} journal entries. "
        f"Your entries have been {overall_tone} overall and have {trend_desc} during this time."
    )

    return description


def get_resources() -> List[Dict[str, str]]:
    """
    Get professional mental health and crisis resources.

    Always returns the same list regardless of sentiment, ensuring everyone
    has access to support resources.

    Returns:
        List of resource dictionaries with name, description, and contact info
    """
    return [
        {
            "name": "National Suicide Prevention Lifeline (USA)",
            "description": "24/7 free and confidential support for people in distress",
            "contact": "988 (call or text)",
            "website": "https://988lifeline.org"
        },
        {
            "name": "Crisis Text Line (USA)",
            "description": "Free, 24/7 crisis support via text message",
            "contact": "Text HOME to 741741",
            "website": "https://www.crisistextline.org"
        },
        {
            "name": "SAMHSA National Helpline (USA)",
            "description": "Treatment referral and information service for mental health and substance use",
            "contact": "1-800-662-4357 (24/7)",
            "website": "https://www.samhsa.gov/find-help/national-helpline"
        },
        {
            "name": "International Association for Suicide Prevention",
            "description": "Directory of crisis centers around the world",
            "contact": "Find your country's hotline",
            "website": "https://www.iasp.info/resources/Crisis_Centres/"
        },
        {
            "name": "NAMI (National Alliance on Mental Illness)",
            "description": "Education, support, and advocacy for individuals and families affected by mental illness",
            "contact": "1-800-950-6264 (Mon-Fri 10am-10pm ET)",
            "website": "https://www.nami.org"
        },
        {
            "name": "Psychology Today Therapist Finder",
            "description": "Find licensed therapists and psychiatrists near you",
            "contact": "Online directory",
            "website": "https://www.psychologytoday.com/us/therapists"
        },
        {
            "name": "BetterHelp / Talkspace",
            "description": "Online therapy platforms for convenient professional counseling",
            "contact": "Register online",
            "website": "https://www.betterhelp.com or https://www.talkspace.com"
        },
        {
            "name": "Emergency Services",
            "description": "In case of immediate danger or medical emergency",
            "contact": "911 (USA) or your local emergency number",
            "website": None
        }
    ]


def get_entry(entry_id: int) -> Optional[Dict[str, Any]]:
    """Get a journal entry by ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM journal_entries WHERE id = ?", (entry_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return dict(row)


def get_user_entries(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent entries for a user."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM journal_entries
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (user_id, limit))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def main():
    """Test the journal system."""
    print("=" * 70)
    print("Mental Health Journal System")
    print("=" * 70 + "\n")

    # Test user
    test_user = "test_user_001"

    # Sample journal entries with varying tones
    sample_entries = [
        "Today was absolutely wonderful! I felt energized and accomplished so much. Met up with friends and had a great time laughing together.",

        "Had an okay day. Nothing special happened but nothing bad either. Just went through the motions at work and came home.",

        "I'm struggling today. Everything feels heavy and I can't seem to shake this tired feeling. Work was stressful and I feel overwhelmed.",

        "Better day than yesterday. Managed to get some exercise in the morning which helped my mood. Still feeling a bit anxious but trying to stay positive.",

        "Really rough night. Couldn't sleep well and my thoughts kept racing. Feeling pretty down and don't have much motivation to do anything."
    ]

    print("Adding 5 sample journal entries...\n")

    entry_ids = []
    for i, entry_text in enumerate(sample_entries, 1):
        print(f"Entry {i}:")
        print(f"  Text: {entry_text[:60]}...")
        entry_id = add_entry(test_user, entry_text)
        entry_ids.append(entry_id)
        print()

    # Get sentiment trend
    print("\n" + "=" * 70)
    print("SENTIMENT TREND ANALYSIS")
    print("=" * 70 + "\n")

    trend = get_trend(test_user, window_days=14)
    print(trend)

    # Show individual entries and their sentiment
    print("\n" + "─" * 70)
    print("Individual Entry Sentiments:")
    print("─" * 70 + "\n")

    entries = get_user_entries(test_user, limit=5)
    for entry in reversed(entries):  # Show in chronological order
        print(f"Entry #{entry['id']}:")
        print(f"  Sentiment: {entry['sentiment_label']} (score: {entry['sentiment_score']:.2f})")
        print(f"  Text: {entry['text'][:80]}...")
        print()

    # Display resources
    print("\n" + "=" * 70)
    print("MENTAL HEALTH RESOURCES")
    print("=" * 70 + "\n")

    print("These resources are always available to support your wellbeing:\n")

    resources = get_resources()
    for i, resource in enumerate(resources, 1):
        print(f"{i}. {resource['name']}")
        print(f"   {resource['description']}")
        print(f"   Contact: {resource['contact']}")
        if resource.get('website'):
            print(f"   Website: {resource['website']}")
        print()

    print("=" * 70)
    print("\nRemember: This is a journaling tool, not a substitute for")
    print("professional mental health care. If you're struggling, please")
    print("reach out to a qualified mental health professional.")
    print("=" * 70)


if __name__ == "__main__":
    main()

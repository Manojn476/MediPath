"""
Fitness tracking and activity recommendation system.

Provides MET-based calorie estimation (deterministic) and ML-based
activity goal suggestions from Fitbit data patterns.
"""
import sys
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Paths
MET_TABLE_PATH = Path("data/met_table.csv")
MODEL_PATH = Path("models/fitness_trend_model.pkl")
SCALER_PATH = Path("models/fitness_scaler.pkl")

# Load MET table (cached)
_met_table = None


def load_met_table() -> pd.DataFrame:
    """Load MET (Metabolic Equivalent of Task) values table."""
    global _met_table

    if _met_table is None:
        if not MET_TABLE_PATH.exists():
            raise FileNotFoundError(
                f"MET table not found at {MET_TABLE_PATH}. "
                "Please create it with activity MET values."
            )
        _met_table = pd.read_csv(MET_TABLE_PATH)
        print(f"[LOAD] MET table loaded with {len(_met_table)} activities")

    return _met_table


def estimate_calories(
    activity_type: str,
    duration_minutes: float,
    weight_kg: float
) -> float:
    """
    Estimate calories burned using MET formula (deterministic).

    Formula: Calories = MET × weight_kg × duration_hours

    Args:
        activity_type: Activity type (must match MET table)
        duration_minutes: Duration of activity in minutes
        weight_kg: User's weight in kilograms

    Returns:
        Estimated calories burned

    Raises:
        ValueError: If activity type not found in MET table
    """
    met_table = load_met_table()

    # Look up MET value
    activity_row = met_table[met_table['activity'] == activity_type]

    if activity_row.empty:
        available = met_table['activity'].tolist()
        raise ValueError(
            f"Activity '{activity_type}' not found in MET table. "
            f"Available activities: {available[:10]}... ({len(available)} total)"
        )

    met_value = activity_row.iloc[0]['met_value']

    # Calculate calories: MET × weight (kg) × time (hours)
    duration_hours = duration_minutes / 60.0
    calories = met_value * weight_kg * duration_hours

    return round(calories, 2)


def generate_synthetic_fitbit_data(n_samples: int = 500) -> pd.DataFrame:
    """
    Generate synthetic Fitbit-style data for testing.

    Features: steps, active_minutes, sleep_hours
    Labels: sedentary, lightly_active, moderately_active, very_active

    Args:
        n_samples: Number of samples to generate

    Returns:
        DataFrame with features and activity_level label
    """
    np.random.seed(42)

    data = []

    # Define activity level profiles
    profiles = {
        'sedentary': {
            'steps': (1000, 4000),
            'active_minutes': (0, 20),
            'sleep_hours': (5, 8)
        },
        'lightly_active': {
            'steps': (4000, 7500),
            'active_minutes': (20, 40),
            'sleep_hours': (6, 8)
        },
        'moderately_active': {
            'steps': (7500, 12000),
            'active_minutes': (40, 70),
            'sleep_hours': (6.5, 8.5)
        },
        'very_active': {
            'steps': (12000, 20000),
            'active_minutes': (70, 150),
            'sleep_hours': (7, 9)
        }
    }

    samples_per_class = n_samples // len(profiles)

    for activity_level, ranges in profiles.items():
        for _ in range(samples_per_class):
            steps = np.random.randint(ranges['steps'][0], ranges['steps'][1])
            active_minutes = np.random.randint(
                ranges['active_minutes'][0],
                ranges['active_minutes'][1]
            )
            sleep_hours = np.random.uniform(
                ranges['sleep_hours'][0],
                ranges['sleep_hours'][1]
            )

            data.append({
                'steps': steps,
                'active_minutes': active_minutes,
                'sleep_hours': sleep_hours,
                'activity_level': activity_level
            })

    df = pd.DataFrame(data)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def train_fitness_trend_model(
    data_path: Optional[Path] = None,
    test_size: float = 0.2
) -> tuple:
    """
    Train activity trend prediction model from Fitbit data.

    Predicts activity level category from steps, active minutes, and sleep.

    Args:
        data_path: Path to Fitbit CSV data (if None, uses synthetic data)
        test_size: Proportion for test set

    Returns:
        Tuple of (model, scaler, X_test, y_test)
    """
    print("=" * 70)
    print("Training Fitness Trend Model")
    print("=" * 70 + "\n")

    # Load or generate data
    if data_path and data_path.exists():
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("[INFO] No Fitbit data provided, generating synthetic data...")
        df = generate_synthetic_fitbit_data(n_samples=500)

        # Save synthetic data for reference
        synthetic_path = Path("data/processed/synthetic_fitbit.csv")
        synthetic_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(synthetic_path, index=False)
        print(f"[SAVE] Synthetic data saved to: {synthetic_path}")

    print(f"\nDataset: {len(df)} samples")
    print(f"Features: {[col for col in df.columns if col != 'activity_level']}")
    print(f"Target: activity_level")
    print(f"\nClass distribution:")
    print(df['activity_level'].value_counts())
    print()

    # Prepare features and target
    feature_cols = ['steps', 'active_minutes', 'sleep_hours']
    X = df[feature_cols].values
    y = df['activity_level'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples\n")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train)

    # Feature importance
    feature_importance = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nFeature Importance:")
    for feature, importance in feature_importance:
        print(f"  {feature:<20} {importance:.4f}")

    # Evaluate
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70 + "\n")

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.3f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model and scaler
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"[SAVE] Model saved to: {MODEL_PATH}")
    print(f"[SAVE] Scaler saved to: {SCALER_PATH}\n")
    print("=" * 70)

    return model, scaler, X_test_scaled, y_test


def load_fitness_model() -> tuple:
    """Load trained fitness trend model and scaler."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train the model first."
        )

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler


def suggest_activity_goal(user_features: Dict[str, float]) -> str:
    """
    Suggest activity goal based on user's current fitness patterns.

    Uses ML model trained on Fitbit data to classify activity level
    and provide personalized recommendations.

    Args:
        user_features: Dict with 'steps', 'active_minutes', 'sleep_hours'

    Returns:
        Personalized activity goal suggestion string
    """
    model, scaler = load_fitness_model()

    # Extract features in correct order
    feature_order = ['steps', 'active_minutes', 'sleep_hours']
    feature_vector = np.array([[user_features[f] for f in feature_order]])

    # Scale and predict
    feature_vector_scaled = scaler.transform(feature_vector)
    predicted_level = model.predict(feature_vector_scaled)[0]
    probabilities = model.predict_proba(feature_vector_scaled)[0]

    # Get class labels and confidence
    classes = model.classes_
    confidence = probabilities[list(classes).index(predicted_level)]

    # Generate personalized suggestion based on current level
    suggestions = {
        'sedentary': (
            f"Your activity level is currently sedentary (daily average: "
            f"{user_features['steps']:.0f} steps, {user_features['active_minutes']:.0f} active minutes). "
            f"Goal suggestion: Start with a modest increase to 5,000 steps per day and "
            f"30 minutes of light activity. Consider taking short walking breaks throughout the day."
        ),
        'lightly_active': (
            f"You're lightly active (daily average: "
            f"{user_features['steps']:.0f} steps, {user_features['active_minutes']:.0f} active minutes). "
            f"Goal suggestion: Aim for 8,000 steps daily and 45 minutes of moderate activity. "
            f"Try adding one 20-minute brisk walk to your routine."
        ),
        'moderately_active': (
            f"You're moderately active (daily average: "
            f"{user_features['steps']:.0f} steps, {user_features['active_minutes']:.0f} active minutes). "
            f"Great progress! Goal suggestion: Push toward 10,000 steps and 60+ minutes of activity. "
            f"Consider adding strength training 2-3 times per week."
        ),
        'very_active': (
            f"You're very active (daily average: "
            f"{user_features['steps']:.0f} steps, {user_features['active_minutes']:.0f} active minutes). "
            f"Excellent! Goal suggestion: Maintain this level with variety—mix cardio, strength, "
            f"and flexibility training. Ensure adequate rest and recovery."
        )
    }

    suggestion = suggestions.get(
        predicted_level,
        "Unable to generate suggestion for current activity level."
    )

    # Add sleep recommendation if needed
    sleep = user_features['sleep_hours']
    if sleep < 7:
        suggestion += (
            f"\n\nNote: You're averaging {sleep:.1f} hours of sleep. "
            f"Consider aiming for 7-9 hours to support recovery and overall health."
        )

    return suggestion


def main():
    """Test calorie estimation and activity goal suggestion."""
    print("=" * 70)
    print("Fitness Tracking System")
    print("=" * 70 + "\n")

    # Part 1: Test MET-based calorie estimation (deterministic)
    print("PART 1: CALORIE ESTIMATION (MET-based, deterministic)")
    print("=" * 70 + "\n")

    # Load MET table
    met_table = load_met_table()
    print(f"\nAvailable activities ({len(met_table)}):")
    for _, row in met_table.head(10).iterrows():
        print(f"  - {row['activity']}: {row['met_value']} MET ({row['description']})")
    print(f"  ... and {len(met_table) - 10} more\n")

    # Sample calorie estimations
    test_cases = [
        {
            'activity': 'running_6mph',
            'duration': 30,
            'weight': 70,
            'description': '30-minute run at 6 mph for 70kg person'
        },
        {
            'activity': 'walking_brisk',
            'duration': 45,
            'weight': 65,
            'description': '45-minute brisk walk for 65kg person'
        },
        {
            'activity': 'swimming_laps',
            'duration': 60,
            'weight': 80,
            'description': '1-hour vigorous swimming for 80kg person'
        },
        {
            'activity': 'yoga',
            'duration': 60,
            'weight': 60,
            'description': '1-hour yoga session for 60kg person'
        }
    ]

    print("Sample Calorie Calculations:")
    print("-" * 70)

    for case in test_cases:
        calories = estimate_calories(
            case['activity'],
            case['duration'],
            case['weight']
        )

        # Get MET value for display
        met_row = met_table[met_table['activity'] == case['activity']].iloc[0]
        met = met_row['met_value']

        print(f"\n{case['description']}")
        print(f"  Activity: {case['activity']}")
        print(f"  MET value: {met}")
        print(f"  Formula: {met} × {case['weight']}kg × {case['duration']/60:.2f}h")
        print(f"  Calories burned: {calories} kcal")

    # Part 2: ML-based activity goal suggestion
    print("\n\n" + "=" * 70)
    print("PART 2: ACTIVITY GOAL SUGGESTIONS (ML-based)")
    print("=" * 70 + "\n")

    # Train model if not exists
    fitbit_data_path = Path("data/raw/fitbit/activity_data.csv")

    if not MODEL_PATH.exists():
        print("[TRAIN] Training fitness trend model...\n")
        model, scaler, X_test, y_test = train_fitness_trend_model(
            data_path=fitbit_data_path if fitbit_data_path.exists() else None
        )
    else:
        print(f"[LOAD] Model already exists at {MODEL_PATH}\n")
        model, scaler = load_fitness_model()

    # Test goal suggestions for different user profiles
    print("\n" + "=" * 70)
    print("TESTING GOAL SUGGESTIONS")
    print("=" * 70 + "\n")

    test_users = [
        {
            'name': 'Sedentary User',
            'features': {'steps': 2500, 'active_minutes': 10, 'sleep_hours': 6.5}
        },
        {
            'name': 'Lightly Active User',
            'features': {'steps': 6000, 'active_minutes': 35, 'sleep_hours': 7.0}
        },
        {
            'name': 'Very Active User',
            'features': {'steps': 15000, 'active_minutes': 100, 'sleep_hours': 8.0}
        }
    ]

    for user in test_users:
        print(f"User Profile: {user['name']}")
        print("-" * 70)
        print(f"Current Stats:")
        print(f"  Steps/day: {user['features']['steps']}")
        print(f"  Active minutes/day: {user['features']['active_minutes']}")
        print(f"  Sleep hours/night: {user['features']['sleep_hours']}")
        print(f"\nPersonalized Goal:")

        suggestion = suggest_activity_goal(user['features'])
        # Indent the suggestion
        for line in suggestion.split('\n'):
            print(f"  {line}")

        print("\n")

    print("=" * 70)
    print("\nKEY OBSERVATIONS:")
    print("  1. Calorie estimation (Part 1) is purely deterministic (MET formula)")
    print("  2. Goal suggestions (Part 2) use ML classification of activity patterns")
    print("  3. Both systems are independent - calories don't depend on ML model")
    print("=" * 70)


if __name__ == "__main__":
    main()

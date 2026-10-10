"""
Medication adherence prediction model.

Builds features from adherence logs and trains a classifier to predict
the risk of missing a dose based on temporal patterns and user history.
"""
import sys
import pickle
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# Paths
DATA_PATH = Path("data/processed/adherence_logs.csv")
MODEL_PATH = Path("models/adherence_model.pkl")
SCALER_PATH = Path("models/adherence_scaler.pkl")


def load_adherence_data(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load adherence logs from CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {csv_path}. "
            "Run synthetic_adherence_generator.py first."
        )

    df = pd.read_csv(csv_path)
    print(f"[LOAD] Loaded {len(df)} adherence records")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for adherence prediction.

    Features:
    - time_of_day (encoded: morning=0, afternoon=1, evening=2)
    - day_of_week (0-6, Monday=0)
    - is_weekend (binary)
    - trailing_7day_miss_count (per user)
    - trailing_14day_miss_count (per user)
    - current_streak_length (consecutive takes or misses)
    - user_overall_adherence_rate (per user, lifetime)
    - dose_number_in_day (1st, 2nd, or 3rd dose of the day)

    Args:
        df: Raw adherence logs DataFrame

    Returns:
        DataFrame with features and target variable
    """
    print("\n[BUILD] Building features...")

    # Sort by user and time
    df = df.sort_values(['user_id', 'scheduled_time']).reset_index(drop=True)

    # Parse datetime
    df['scheduled_dt'] = pd.to_datetime(df['scheduled_time'])
    df['date_only'] = df['scheduled_dt'].dt.date

    # Encode time_of_day
    time_encoding = {'morning': 0, 'afternoon': 1, 'evening': 2}
    df['time_of_day_encoded'] = df['time_of_day'].map(time_encoding)

    # Dose number within the day (for users taking multiple doses per day)
    df['dose_number_in_day'] = df.groupby(['user_id', 'date_only']).cumcount() + 1

    # Calculate trailing miss counts per user
    df['trailing_7day_miss_count'] = 0
    df['trailing_14day_miss_count'] = 0
    df['current_streak_length'] = 0
    df['user_overall_adherence'] = 0.0

    # Process per user
    for user_id in df['user_id'].unique():
        user_mask = df['user_id'] == user_id
        user_df = df[user_mask].copy()

        # Calculate trailing misses and streaks
        trailing_7day = []
        trailing_14day = []
        streaks = []
        adherence_rates = []

        for idx in range(len(user_df)):
            # Look back 7 days
            current_date = user_df.iloc[idx]['scheduled_dt']
            date_7days_ago = current_date - pd.Timedelta(days=7)
            date_14days_ago = current_date - pd.Timedelta(days=14)

            # Count misses in trailing windows (excluding current dose)
            prior_doses = user_df.iloc[:idx]

            misses_7day = prior_doses[
                (prior_doses['scheduled_dt'] >= date_7days_ago) &
                (prior_doses['taken'] == 0)
            ].shape[0]

            misses_14day = prior_doses[
                (prior_doses['scheduled_dt'] >= date_14days_ago) &
                (prior_doses['taken'] == 0)
            ].shape[0]

            trailing_7day.append(misses_7day)
            trailing_14day.append(misses_14day)

            # Current streak length (from consecutive_takes_before or consecutive_misses_before)
            # Positive = taking streak, Negative = missing streak
            takes_before = user_df.iloc[idx]['consecutive_takes_before']
            misses_before = user_df.iloc[idx]['consecutive_misses_before']

            if takes_before > 0:
                streak = takes_before
            else:
                streak = -misses_before  # Negative for miss streaks

            streaks.append(streak)

            # Overall adherence rate (up to but not including current dose)
            if idx > 0:
                adherence = prior_doses['taken'].mean()
            else:
                adherence = 0.8  # Prior for first dose

            adherence_rates.append(adherence)

        # Assign back to dataframe
        df.loc[user_mask, 'trailing_7day_miss_count'] = trailing_7day
        df.loc[user_mask, 'trailing_14day_miss_count'] = trailing_14day
        df.loc[user_mask, 'current_streak_length'] = streaks
        df.loc[user_mask, 'user_overall_adherence'] = adherence_rates

    # Select feature columns
    feature_cols = [
        'time_of_day_encoded',
        'day_of_week',
        'is_weekend',
        'trailing_7day_miss_count',
        'trailing_14day_miss_count',
        'current_streak_length',
        'user_overall_adherence',
        'dose_number_in_day',
    ]

    # Target: 0 = taken, 1 = missed (we're predicting "miss risk")
    df['missed'] = 1 - df['taken']

    feature_df = df[feature_cols + ['missed', 'user_id']].copy()

    print(f"  Built {len(feature_cols)} features for {len(feature_df)} samples")
    print(f"  Class distribution: {df['missed'].value_counts().to_dict()}")

    return feature_df


def train_model(
    feature_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Any, Any, pd.DataFrame, pd.DataFrame]:
    """
    Train a Random Forest classifier to predict missed doses.

    Uses class weighting to handle the imbalanced dataset (more takes than misses).

    Args:
        feature_df: DataFrame with features and target
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (trained_model, scaler, X_test, y_test)
    """
    print("\n[TRAIN] Training adherence prediction model...")

    # Split by user to avoid leakage (keep users separate in train/test)
    unique_users = feature_df['user_id'].unique()
    train_users, test_users = train_test_split(
        unique_users, test_size=test_size, random_state=random_state
    )

    train_df = feature_df[feature_df['user_id'].isin(train_users)]
    test_df = feature_df[feature_df['user_id'].isin(test_users)]

    # Drop user_id from features
    X_train = train_df.drop(['missed', 'user_id'], axis=1)
    y_train = train_df['missed']
    X_test = test_df.drop(['missed', 'user_id'], axis=1)
    y_test = test_df['missed']

    print(f"  Train set: {len(X_train)} samples ({len(train_users)} users)")
    print(f"  Test set:  {len(X_test)} samples ({len(test_users)} users)")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest with class weighting
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',  # Handle class imbalance
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train)

    print(f"  Model trained: {model.__class__.__name__}")

    # Feature importance
    feature_names = X_train.columns
    importances = model.feature_importances_
    feature_importance = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\n  Feature Importance:")
    for feat, imp in feature_importance:
        print(f"    {feat:<30} {imp:.4f}")

    return model, scaler, pd.DataFrame(X_test_scaled, columns=X_train.columns), y_test


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> None:
    """
    Evaluate model performance on test set.

    Reports precision, recall, F1-score, and confusion matrix.
    """
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70 + "\n")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of miss

    # Classification report
    print("Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Taken (0)', 'Missed (1)'],
        digits=3
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"                Predicted Taken  Predicted Missed")
    print(f"Actual Taken    {cm[0][0]:<16} {cm[0][1]:<16}")
    print(f"Actual Missed   {cm[1][0]:<16} {cm[1][1]:<16}")
    print()

    # ROC AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {auc:.3f}")

    # Additional metrics
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=[0, 1]
    )

    print(f"\nKey Metrics for 'Missed' class:")
    print(f"  Precision: {precision[1]:.3f} (of predicted misses, how many were correct)")
    print(f"  Recall:    {recall[1]:.3f} (of actual misses, how many did we catch)")
    print(f"  F1-Score:  {f1[1]:.3f}")

    print("\n" + "=" * 70)


def save_model(model: Any, scaler: Any) -> None:
    """Save trained model and scaler to disk."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n[SAVE] Model saved to: {MODEL_PATH}")
    print(f"[SAVE] Scaler saved to: {SCALER_PATH}")


def load_model() -> Tuple[Any, Any]:
    """Load trained model and scaler from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train the model first."
        )

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler


def predict_miss_risk(features: Dict[str, float]) -> float:
    """
    Predict the probability of missing a dose given features.

    Args:
        features: Dictionary with feature values:
            - time_of_day_encoded: 0=morning, 1=afternoon, 2=evening
            - day_of_week: 0-6 (Monday=0)
            - is_weekend: 0 or 1
            - trailing_7day_miss_count: integer
            - trailing_14day_miss_count: integer
            - current_streak_length: integer (positive=taking, negative=missing)
            - user_overall_adherence: float 0-1
            - dose_number_in_day: 1, 2, or 3

    Returns:
        Float probability of missing (0-1, higher = more likely to miss)
    """
    model, scaler = load_model()

    # Feature order must match training
    feature_order = [
        'time_of_day_encoded',
        'day_of_week',
        'is_weekend',
        'trailing_7day_miss_count',
        'trailing_14day_miss_count',
        'current_streak_length',
        'user_overall_adherence',
        'dose_number_in_day',
    ]

    # Build feature vector
    feature_vector = np.array([[features[f] for f in feature_order]])

    # Scale and predict
    feature_vector_scaled = scaler.transform(feature_vector)
    miss_probability = model.predict_proba(feature_vector_scaled)[0, 1]

    return miss_probability


def main():
    """Train and evaluate the adherence prediction model."""
    print("=" * 70)
    print("Medication Adherence Prediction Model")
    print("=" * 70 + "\n")

    # Load data
    df = load_adherence_data()

    # Build features
    feature_df = build_features(df)

    # Train model
    model, scaler, X_test, y_test = train_model(feature_df)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model, scaler)

    # Test prediction on sample cases
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70 + "\n")

    sample_cases = [
        {
            "name": "High-risk evening dose on weekend",
            "features": {
                'time_of_day_encoded': 2,  # Evening
                'day_of_week': 6,  # Sunday
                'is_weekend': 1,
                'trailing_7day_miss_count': 3,
                'trailing_14day_miss_count': 6,
                'current_streak_length': -2,  # Missing streak
                'user_overall_adherence': 0.70,
                'dose_number_in_day': 2,
            }
        },
        {
            "name": "Low-risk morning dose, adherent user",
            "features": {
                'time_of_day_encoded': 0,  # Morning
                'day_of_week': 2,  # Wednesday
                'is_weekend': 0,
                'trailing_7day_miss_count': 0,
                'trailing_14day_miss_count': 1,
                'current_streak_length': 15,  # Strong taking streak
                'user_overall_adherence': 0.95,
                'dose_number_in_day': 1,
            }
        },
        {
            "name": "Moderate-risk afternoon dose",
            "features": {
                'time_of_day_encoded': 1,  # Afternoon
                'day_of_week': 4,  # Friday
                'is_weekend': 0,
                'trailing_7day_miss_count': 2,
                'trailing_14day_miss_count': 4,
                'current_streak_length': 3,  # Short taking streak
                'user_overall_adherence': 0.82,
                'dose_number_in_day': 2,
            }
        }
    ]

    for case in sample_cases:
        risk = predict_miss_risk(case["features"])
        risk_level = "HIGH" if risk > 0.5 else "MODERATE" if risk > 0.3 else "LOW"

        print(f"Case: {case['name']}")
        print(f"  Miss Risk: {risk:.1%} ({risk_level})")
        print()

    print("=" * 70)


if __name__ == "__main__":
    main()

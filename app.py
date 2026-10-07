from __future__ import annotations

import json

import streamlit as st

from medipath.db import init_db
from medipath.orchestrator import run_supervisor
from medipath.state import SharedState

st.set_page_config(page_title="MediPath", layout="wide")
init_db()

LANGUAGES = ["English", "Hindi", "Kannada", "Tamil"]
INTENTS = ["health", "prescription", "facility", "journal", "nutrition", "fitness"]

RESOURCE_DIRECTORY = [
    "Emergency services: 112",
    "Mental health helpline (India): 1800-599-0019",
    "Women helpline: 181",
    "Child helpline: 1098",
]

st.sidebar.title("Always-visible support resources")
for item in RESOURCE_DIRECTORY:
    st.sidebar.info(item)

st.title("MediPath – AI-Powered Health, Nutrition & Wellness Navigator")
st.caption("Target region: India · Supported languages: English, Hindi, Kannada, Tamil")

user_id = st.text_input("User ID", value="demo-user")
language = st.selectbox("Language", LANGUAGES)
input_type = st.radio("Input type", ["text", "voice"], horizontal=True)
intent = st.selectbox("Intent", INTENTS)
user_text = st.text_area("Message / transcript", placeholder="Ask a question or describe what you need")

payload = {}

if intent == "prescription":
    uploaded_prescription = st.file_uploader("Upload prescription image", type=["png", "jpg", "jpeg"])
    prescription_text = st.text_area("OCR text (or paste extracted text)", value="Paracetamol 500 mg BD 5 days")
    payload["prescription_text"] = prescription_text
    if uploaded_prescription:
        st.success("Prescription image received. OCR output must be user-confirmed before use.")
elif intent == "facility":
    payload["region"] = st.text_input("Region", value="Bengaluru")
    payload["age"] = st.number_input("Age", min_value=0, max_value=120, value=30)
    payload["income"] = st.number_input("Annual income", min_value=0, value=200000)
    payload["residency"] = st.text_input("Residency", value="Karnataka")
elif intent == "journal":
    payload["mood"] = st.selectbox("Mood", ["happy", "neutral", "sad", "stressed"])
    payload["notes"] = st.text_area("Journal note")
elif intent in {"nutrition", "fitness"}:
    meal_photo = st.file_uploader("Upload meal image (optional)", type=["png", "jpg", "jpeg"])
    payload["goal"] = st.text_input("Goal", value="balanced energy")
    payload["preferences"] = st.text_input("Food preferences", value="vegetarian")
    payload["food_label"] = st.text_input("Detected/entered meal name", value="idli")
    payload["food_confidence"] = st.slider("Meal confidence", 0.0, 1.0, 0.7, 0.05)
    payload["weight_kg"] = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0)
    payload["met"] = st.number_input("MET value", min_value=1.0, max_value=20.0, value=4.0)
    payload["minutes"] = st.number_input("Activity duration (minutes)", min_value=1.0, max_value=300.0, value=30.0)
    if meal_photo:
        st.info("Meal image received. Please confirm detected meal when confidence is low.")

new_adherence_data = st.checkbox("New adherence log data arrived")
new_activity_data = st.checkbox("New activity log data arrived")

if st.button("Run MediPath"):
    state = SharedState(
        user_id=user_id,
        language=language,
        input_type=input_type,
        intent=intent,
        raw_input=user_text,
        new_data_flags={
            "adherence": new_adherence_data,
            "activity": new_activity_data,
            "journal": intent == "journal",
        },
    )
    result = run_supervisor(state, payload)

    st.subheader("Care Summary")
    st.json(result["care_summary"])

    st.subheader("Follow-up Tracker")
    st.json(result["follow_up"])

    st.subheader("Full Output")
    st.code(json.dumps(result, indent=2), language="json")

st.markdown(
    "**Safety:** MediPath provides general information and organization support only. "
    "It does not diagnose, triage, or provide clinical conclusions."
)

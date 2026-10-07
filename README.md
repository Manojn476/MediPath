# MediPath

MediPath is a multilingual, voice-enabled, multi-agent health and wellness navigator focused on **information support** (not diagnosis).

## Target region and languages
- **Target region:** India
- **Supported languages:** English, Hindi, Kannada, Tamil

## Safety and hard constraints
- No diagnosis, triage, or clinical inference.
- Health Q&A answers are grounded in configured official sources and include source attribution.
- Prescription OCR output is treated as draft and must be user-confirmed before reminder use.
- Crisis/professional support resources are always visible in the UI sidebar.
- Stack and dependencies are free/open-source or free-tier compatible.

## Architecture (Python + LangChain + LangGraph)
- `Supervisor` (`medipath/orchestrator.py`) coordinates flow.
- `Intake Agent` classifies intent and selects branches.
- Selected branch agents run and feed `Care Summary Agent`.
- `Follow-up Tracker` checks new-data flags and triggers loop targets for adherence/activity.

Shared state includes:
`user_id`, `language`, `input_type`, `intent`, `active_branches`, `branch_outputs`, and `new_data_flags`.

## Implemented flows
1. Health literacy Q&A (source-grounded fallback behavior)
2. Prescription digitization (text pathway + mandatory user confirmation flag)
3. Facility finder + scheme eligibility explanation
4. Wellness journal sentiment/trend note with always-visible resources
5. Nutrition + food recognition + fitness calorie estimation (MET formula)

## Storage
SQLite database initialized with required tables:
- `users`
- `prescriptions`
- `medication_schedule`
- `adherence_logs`
- `journal_entries`
- `activity_logs`
- `facility_cache`
- `scheme_criteria`

## Run locally
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Run tests
```bash
python -m unittest discover -s tests -q
```

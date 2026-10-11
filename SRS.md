# Software Requirements Specification (SRS)
## Project: MediPath — Multi-Agent Health, Nutrition & Wellness Assistant
**Document Version:** 1.0.0 (Milestone Snapshot)  
**Status:** Baseline Implemented & Verified (Active Working Draft)  
**Date:** October 2026  
**Repository:** [Manojn476/MediPath](https://github.com/Manojn476/MediPath)  

> **Note on Current Status:** This document specifies the software requirements and system architecture for the MediPath system reflecting the currently built and tested codebase. Requirements marked **[IMPLEMENTED & TESTED]** represent operational modules. Requirements marked **[PLANNED / NEXT PHASE]** specify upcoming components (Facility Finder, Government Scheme Advisor, and UI Frontend) that will be incorporated in subsequent milestones.

---

## 1. Introduction

### 1.1 Purpose
This Software Requirements Specification (SRS) establishes the functional and non-functional requirements, architectural characteristics, and interface specifications for **MediPath**. MediPath is an accessible, multilingual, voice-enabled multi-agent health and wellness platform designed to guide users with verified health literacy, adherence tracking, nutrition guidance, and wellness management.

### 1.2 Document Conventions
- **MUST / SHALL**: Mandatory requirement.
- **SHOULD**: Strongly recommended feature or behavior.
- **MAY**: Optional capability.
- **[IMPLEMENTED & TESTED]**: Code exists, is functional, and unit/integration tested.
- **[PLANNED / NEXT PHASE]**: Target specification for upcoming implementation cycles.

### 1.3 Intended Audience
- AI and Machine Learning Engineers developing agents and pipelines.
- Full-Stack Developers implementing user interfaces and database integrations.
- Quality Assurance Engineers maintaining test suites.
- Health Informatics Reviewers verifying non-diagnostic safety guardrails.

### 1.4 Product Scope
MediPath acts as a preventative wellness companion and health navigation assistant. It encompasses:
1. Multilingual speech-to-text processing for voice queries.
2. Fast regex and keyword-based intent classification.
3. RAG-driven health literacy question-answering grounded in WHO public data.
4. Computer vision OCR for prescription documents with fuzzy drug matching and dosage parsing.
5. Automated medication reminder scheduling.
6. Machine learning-driven prediction of missed-dose adherence risk.
7. Empathetic, non-diagnostic mental health journaling with longitudinal trend analysis.
8. Deterministic MET fitness calorie estimation and adaptive step goal recommendations.
9. RAG nutrition guidance and deep-learning food image classification.
10. Unified care summary aggregation with dynamic state-loop recalculation.
11. Localized facility finding and government healthcare scheme eligibility guidance.

### 1.5 Medical & Legal Disclaimer (Safety Mandate)
> **CRITICAL MANDATE:** MediPath is strictly a health literacy and wellness management assistant. It **SHALL NOT**:
> 1. Make medical diagnoses.
> 2. Prescribe or alter medication dosages.
> 3. Provide clinical assessments of psychiatric disorders.
> 4. Offer speculative treatments.
> All medical information must be cited from curated health publications. Mental health interactions must always supply emergency crisis helpline contacts.

---

## 2. Overall Description

### 2.1 Product Perspective & Context
MediPath operates as a local-first, multi-agent pipeline orchestrated through LangGraph. The platform coordinates between specialized task agents, local SQLite persistence, vector databases (FAISS), and fine-tuned machine learning models (Random Forest, ResNet-18, Sentence Transformers).

### 2.2 System Component Status Matrix

| Component | Responsible Module | Current Status | Primary Technology |
| :--- | :--- | :---: | :--- |
| Audio Transcription | `agents/voice_client.py` | [IMPLEMENTED & TESTED] | `faster-whisper` (small) |
| Intent Classification | `agents/intake.py` | [IMPLEMENTED & TESTED] | Regex / Keyword Heuristics |
| Health Literacy RAG | `agents/health_rag.py` | [IMPLEMENTED & TESTED] | FAISS + `all-MiniLM-L6-v2` + Groq LLM |
| Prescription OCR | `agents/prescription_ocr.py` | [IMPLEMENTED & TESTED] | OpenCV + Tesseract + Levenshtein |
| Medication Reminders | `agents/reminder.py` | [IMPLEMENTED & TESTED] | SQLite Schedule Manager |
| Adherence Risk Model | `agents/adherence_model.py` | [IMPLEMENTED & TESTED] | Scikit-Learn Random Forest (100 trees) |
| Mental Health Journal | `agents/journal.py` | [IMPLEMENTED & TESTED] | Sentiment Analysis + Curated Resources |
| Nutrition RAG | `agents/nutrition_rag.py` | [IMPLEMENTED & TESTED] | FAISS + WHO Nutrition Guidelines |
| Food Recognition | `agents/food_recognition.py` | [IMPLEMENTED & TRAINED] | PyTorch Transfer Learning (ResNet-18) |
| Fitness Tracking | `agents/fitness.py` | [IMPLEMENTED & TESTED] | Deterministic MET + Random Forest Trend |
| Care Summary Aggregator | `agents/care_summary.py` | [IMPLEMENTED & TESTED] | Fact-Preserving LLM Synthesis |
| Cyclic Follow-up Engine | `agents/followup.py`<br/>`agents/graph_with_followup.py` | [IMPLEMENTED & TESTED] | LangGraph Conditional Routing |
| Database Layer | `db/schema.py` | [IMPLEMENTED & TESTED] | SQLite3 with Indexes |
| Nearby Facility Finder | `agents/facility_finder.py` | [PLANNED / NEXT PHASE] | OpenStreetMap Overpass API + Caching |
| Health Scheme Advisor | `agents/scheme_eligibility.py` | [PLANNED / NEXT PHASE] | PDF Criteria Extraction + Rule Engine |
| User Interface | `ui/` | [PLANNED / NEXT PHASE] | Web / Mobile Frontend |

---

## 3. System Architecture & Workflow

### 3.1 LangGraph State Machine
MediPath employs a directed graph with conditional edges and cyclic loops.

```
START ──> transcribe_node ──> intake_node ──(Intent Routing)──>
    ├──> health_literacy_node ────────┐
    ├──> nutrition_node ──────────────┤
    ├──> prescription_node ───────────┤
    ├──> fitness_node ────────────────┼──> care_summary_node ──> followup_check_node
    ├──> journal_node ────────────────┤                               │
    └──> not_implemented_node ───────┘                               │
                                                                      ├─[new_adherence_log]──> adherence_update_node ──┐
                                                                      ├─[new_activity_log]───> fitness_update_node ────┼──> care_summary_node
                                                                      ├─[new_journal_entry]──> journal_update_node ────┘
                                                                      └─[no flags active]────> END
```

### 3.2 Graph State Schema (`GraphState`)
The central state dictionary passed between all nodes is structured as follows:

```python
class GraphState(TypedDict):
    input_text: str                     # Raw user query or transcribed speech
    output_text: str                    # Aggregated response returned to user
    audio_path: Optional[str]           # Path to input audio WAV/MP3 file
    intent: Optional[str]               # Detected intent (health, prescription, etc.)
    active_branches: List[str]          # List of executed agent nodes
    detected_language: str              # User language code (e.g., 'en', 'hi')
    
    # Dynamic followup flags
    new_adherence_log: bool             # Triggered when medication adherence is logged
    new_activity_log: bool              # Triggered when physical activity is recorded
    new_journal_entry: bool             # Triggered when a journal entry is submitted
    
    # Agent domain payloads
    health_output: Optional[str]        # Grounded health text from health_rag
    nutrition_output: Optional[str]     # Nutrition guidance from nutrition_rag
    prescription_data: Optional[dict]   # Extracted drug, dosage, and frequency
    medication_schedule: Optional[list] # Assigned reminder schedule slots
    adherence_risk: Optional[float]     # Risk probability (0.0 to 1.0)
    journal_output: Optional[str]       # Journal feedback with sentiment trend
    fitness_data: Optional[dict]        # Calorie burn and adaptive goals
    nearby_facilities: Optional[list]   # Overpass API facility query output
    scheme_eligibility: Optional[dict]  # Matching government health schemes
    food_prediction: Optional[dict]     # Food class and nutrition metadata
```

### 3.3 Dynamic Follow-Up Loop Mechanics
1. If an agent logs new operational data during or immediately following execution (e.g., marking a medication dose taken or logging a workout), helper functions in `agents/followup.py` set the corresponding flag to `True`.
2. Following `care_summary_node`, the flow enters `followup_check_node`.
3. If flags are active, execution branches directly to the corresponding updater (`adherence_update_node`, `fitness_update_node`, or `journal_update_node`).
4. The updater refreshes calculated metrics in `state` and clears the flag to prevent infinite loops.
5. The updated state loops back to `care_summary_node` to update the final summary before reaching `END`.
6. This cycle executes **without** re-running `intake_node` or transcription.

---

## 4. Functional Requirements

### 4.1 Speech Transcription Agent (FR-1) `[IMPLEMENTED & TESTED]`
- **FR-1.1**: The system SHALL accept audio input formats (WAV, MP3, M4A) via `audio_path`.
- **FR-1.2**: Audio SHALL be transcribed using the `faster-whisper` library utilizing the `small` model for efficient local execution.
- **FR-1.3**: The system SHALL automatically detect the spoken language and populate `state["detected_language"]`.
- **FR-1.4**: If no audio is supplied, the system SHALL pass through `state["input_text"]` untouched.

### 4.2 Intent Classification & Intake (FR-2) `[IMPLEMENTED & TESTED]`
- **FR-2.1**: The system SHALL classify user requests into one of 9 distinct intents:
  1. `health` (Medical literacy, symptoms, diseases, public health)
  2. `prescription` (Prescription scans, medications, dosage reminders)
  3. `facility` (Nearby clinics, hospitals, pharmacies)
  4. `scheme` (Ayushman Bharat, government health insurance schemes)
  5. `journal` (Mental health logs, reflections, mood tracking)
  6. `nutrition` (Dietary recommendations, micronutrients, food science)
  7. `food` (Food recognition images, dish recipes)
  8. `fitness` (Calorie burn, workouts, exercise recommendations)
  9. `unclear` (Ambiguous queries, greetings, unclassified input)
- **FR-2.2**: Intake SHALL utilize deterministic regex pattern matching for sub-millisecond execution without requiring LLM inference latency.
- **FR-2.3**: Intake SHALL inspect media file extensions and path hints (e.g., filenames containing "prescription" or "food") to disambiguate intent.
- **FR-2.4**: In benchmark verification (`tests/test_intake.py`), intent routing accuracy SHALL achieve 100% on the standard evaluation suite.

### 4.3 Grounded Health Literacy RAG (FR-3) `[IMPLEMENTED & TESTED]`
- **FR-3.1**: The system SHALL build and maintain a FAISS vector database chunked from verified public health documents (`data/raw/health/who_health_data.pdf`).
- **FR-3.2**: Documents SHALL be split into 300–600 token chunks with 15% overlap and encoded using `all-MiniLM-L6-v2`.
- **FR-3.3**: Retrieval SHALL return the top-$k$ most relevant chunks ranked by Euclidean distance.
- **FR-3.4**: Generation via Groq LLM SHALL strictly enforce grounding instructions:
  - Answers must only draw from retrieved context.
  - Every answer must cite its source document and chunk index.
  - If information is not present in the corpus, the model must explicitly state its limitation rather than hallucinating.
  - The model must never suggest self-treatment or clinical diagnoses.

### 4.4 Prescription OCR & Drug Extraction (FR-4) `[IMPLEMENTED & TESTED]`
- **FR-4.1**: The system SHALL accept prescription images (`.jpg`, `.jpeg`, `.png`).
- **FR-4.2**: The image preprocessing pipeline SHALL apply grayscale conversion, bilateral filtering/denoising, and adaptive Gaussian thresholding before OCR invocation.
- **FR-4.3**: Text SHALL be extracted using Tesseract OCR.
- **FR-4.4**: Candidate drug tokens SHALL be matched against a curated dictionary of 400+ known medications (`data/reference/drug_names.txt`) using normalized Levenshtein edit distance.
- **FR-4.5**: Dosages, units (`mg`, `mcg`, `ml`, `g`, `drops`, `puff`), frequencies (`once daily`, `twice daily`, `tds`, `bd`, `od`), and durations (`days`, `weeks`, `months`) SHALL be extracted via compiled regex expressions.
- **FR-4.6**: An overall confidence score (0.0 to 1.0) SHALL be computed based on the weighted product of drug name distance, dosage recognition, frequency extraction, and duration clarity.

### 4.5 Medication Scheduling & Reminders (FR-5) `[IMPLEMENTED & TESTED]`
- **FR-5.1**: Extracted prescriptions SHALL be recorded in the SQLite database with `status = 'pending'`.
- **FR-5.2**: If confidence score $\ge 0.75$, the system MAY automatically transition the prescription to `status = 'confirmed'`.
- **FR-5.3**: Prescriptions below the threshold SHALL require explicit user confirmation (`confirm_prescription()`).
- **FR-5.4**: The system SHALL generate standardized fixed daily reminder schedules:
  - Once Daily: 08:00 (Morning)
  - Twice Daily: 08:00 (Morning), 20:00 (Evening)
  - Thrice Daily: 08:00 (Morning), 14:00 (Afternoon), 20:00 (Evening)
  - Four Times Daily: 08:00, 12:00, 16:00, 20:00
  - Bedtime: 22:00 (Night)

### 4.6 Machine Learning Adherence Risk Modeling (FR-6) `[IMPLEMENTED & TESTED]`
- **FR-6.1**: The system SHALL maintain a trained Random Forest model (`models/adherence_model.pkl`) to predict the probability of a user missing an upcoming dose.
- **FR-6.2**: Model training SHALL use a synthetic adherence dataset (`data/processed/adherence_logs.csv`) simulating 52 users over 60 days (7,500 records) across 5 realistic personas (High Adherent, Weekend Misser, Erratic, Evening Fatigue, Chronic Misser).
- **FR-6.3**: The inference pipeline (`predict_miss_risk()`) SHALL accept 8 normalized features:
  - `time_of_day_encoded`
  - `day_of_week`
  - `is_weekend`
  - `trailing_7day_miss_count`
  - `trailing_14day_miss_count`
  - `current_streak_length`
  - `user_overall_adherence`
  - `dose_number_in_day`
- **FR-6.4**: Performance requirement: The adherence model SHALL achieve $\ge 95\%$ recall on missed doses (achieved: **96.8% recall**, **0.962 ROC AUC**) to ensure at-risk patients are not missed.

### 4.7 Mental Health Journal & Sentiment Analysis (FR-7) `[IMPLEMENTED & TESTED]`
- **FR-7.1**: The system SHALL store daily journal reflections in the SQLite `journal_entries` table.
- **FR-7.2**: Entries SHALL be evaluated for sentiment polarity on a continuous scale from -1.0 (strongly negative) to +1.0 (strongly positive).
- **FR-7.3**: Longitudinal trends SHALL be computed across a rolling 14-day window (`get_trend()`).
- **FR-7.4**: Responses SHALL strictly use supportive, non-diagnostic, and non-stigmatizing language.
- **FR-7.5**: Every response from the journal agent SHALL append verified emergency crisis and mental health helpline contacts (`get_resources()`).

### 4.8 Nutrition Guidance RAG (FR-8) `[IMPLEMENTED & TESTED]`
- **FR-8.1**: Nutrition queries SHALL be retrieved against a dedicated FAISS vector store containing nutritional science and dietary guidelines.
- **FR-8.2**: The agent SHALL deliver factual breakdowns of macronutrients, micronutrients, dietary sources, and general healthy eating habits.
- **FR-8.3**: The agent SHALL cite source material and refrain from prescribing clinical disease-specific medical diets.

### 4.9 Food Image Recognition (FR-9) `[IMPLEMENTED & TRAINED]`
- **FR-9.1**: The system SHALL accept food photographs and classify dish categories using a fine-tuned ResNet-18 model (`models/food_classifier.pt`).
- **FR-9.2**: Early convolutional layers (layers 1–3) SHALL remain frozen during fine-tuning to retain general ImageNet feature extraction, while layer 4 and the fully connected classification head are trained on target food categories.
- **FR-9.3**: Predictions with confidence below 60% ($< 0.60$) SHALL be flagged for user visual confirmation.
- **FR-9.4**: Successful predictions SHALL link directly to nutritional profile lookups (`lookup_nutrition()`).

### 4.10 Fitness & Calorie Modeling (FR-10) `[IMPLEMENTED & TESTED]`
- **FR-10.1**: The system SHALL calculate caloric expenditure deterministically using Metabolic Equivalent of Task (MET) values:
  $$\text{Calories Burned} = \text{MET} \times \text{Weight (kg)} \times \text{Duration (hours)}$$
  using reference values defined in `data/met_table.csv`.
- **FR-10.2**: Activity patterns SHALL be modeled using a trained Random Forest trend classifier (`models/fitness_trend_model.pkl`) to output adaptive step and activity goals (`suggest_activity_goal()`) based on steps, active minutes, and sleep hours.

### 4.11 Multi-Branch Care Summary Aggregator (FR-11) `[IMPLEMENTED & TESTED]`
- **FR-11.1**: The system SHALL aggregate results from all active branches (`state["active_branches"]`) into a single, cohesive patient care summary.
- **FR-11.2**: Numerical facts, dosages, drug names, risk probabilities, and citation references SHALL be preserved verbatim without distortion or loss of precision.
- **FR-11.3**: The Groq LLM SHALL smooth transition phrasing and translate the synthesized summary into the user's detected language if non-English.

### 4.12 Cyclic Dynamic Follow-Up Loop Engine (FR-12) `[IMPLEMENTED & TESTED]`
- **FR-12.1**: Followup flags (`new_adherence_log`, `new_activity_log`, `new_journal_entry`) SHALL trigger dedicated state updater nodes after summary generation.
- **FR-12.2**: The graph SHALL execute updater nodes in deterministic priority order: Adherence $\to$ Fitness $\to$ Journal.
- **FR-12.3**: Updater nodes SHALL clear their respective flags upon execution.
- **FR-12.4**: Upon clearing flags, the updated metrics SHALL trigger re-synthesis in `care_summary_node` and cleanly terminate at `END`.

### 4.13 Nearby Healthcare Facility Finder (FR-13) `[PLANNED / NEXT PHASE]`
- **FR-13.1**: The system SHALL accept geographic coordinates (`lat`, `lon`) and a search radius (default 5 km).
- **FR-13.2**: The system SHALL query the OpenStreetMap Overpass API for amenities tagged `hospital`, `clinic`, and `pharmacy`.
- **FR-13.3**: Responses SHALL be formatted into `{name, type, lat, lon, address}` objects.
- **FR-13.4**: A `facility_cache` table in SQLite SHALL cache query responses for 7 days based on geographic region keys to prevent repetitive external network calls.
- **FR-13.5**: The facility finder SHALL be wired into `agents/graph.py` to handle the `facility` intent.

### 4.14 Government Health Scheme Eligibility Advisor (FR-14) `[PLANNED / NEXT PHASE]`
- **FR-14.1**: The system SHALL extract official eligibility criteria from government health scheme policy PDFs (e.g., Ayushman Bharat PM-JAY) stored in `data/raw/schemes/`.
- **FR-14.2**: Extracted rules SHALL be compiled into a structured criteria JSON table (`data/processed/scheme_criteria.json`) containing:
  - `scheme_name`
  - `min_income`, `max_income`
  - `age_range`
  - `region` / `state`
  - `target_category`
- **FR-14.3**: An eligibility evaluator (`check_eligibility(user_profile: dict)`) in `agents/scheme_eligibility.py` SHALL evaluate applicant profiles against all criteria and return matching schemes with concise eligibility rationales.
- **FR-14.4**: The scheme advisor SHALL be wired into the LangGraph workflow under the `scheme` intent.

### 4.15 User Interface & Presentation Layer (FR-15) `[PLANNED / NEXT PHASE]`
- **FR-15.1**: The system SHALL provide an interactive user interface (`ui/`) supporting text dialogue, microphone audio recording, and image uploads (prescriptions, food).
- **FR-15.2**: The interface SHALL display interactive medication reminder timelines, adherence risk gauges, and downloadable care summaries.

---

## 5. External Interface Requirements

### 5.1 Software Interfaces
1. **Groq Cloud API**: Accessed via `langchain-groq` using model `llama-3.3-70b-versatile` for question answering and summary generation.
2. **Tesseract OCR Engine**: Local binary integration via `pytesseract` for text extraction from prescription images.
3. **OpenStreetMap Overpass API**: External REST interface for geospatial facility discovery.
4. **Hugging Face / PyTorch**: Pre-trained transformer checkpoints (`sentence-transformers/all-MiniLM-L6-v2`) and torchvision model weights (`ResNet-18`).
5. **SQLite3**: Embedded relational database engine.

### 5.2 Hardware & Audio Interfaces
1. **Audio Input**: 16 kHz mono or stereo WAV/MP3 microphone streams.
2. **Compute Architecture**: CPU or CUDA-enabled GPU (PyTorch and faster-whisper run locally on CPU by default with CUDA acceleration when available).

---

## 6. Non-Functional Requirements

### 6.1 Clinical Safety & Non-Diagnostic Guardrails
- **NFR-1.1**: The system MUST refuse requests to diagnose diseases, prescribe drugs, or adjust dosages.
- **NFR-1.2**: All health knowledge delivered to users MUST contain unambiguous citations to the underlying corpus.
- **NFR-1.3**: Mental health responses MUST maintain a supportive, non-clinical stance and unconditionally provide helpline contacts.

### 6.2 Performance & Latency
- **NFR-2.1**: Intent classification MUST execute in $\le 5$ milliseconds using regex/keyword heuristics.
- **NFR-2.2**: Vector similarity retrieval from FAISS indices MUST complete in $\le 50$ milliseconds for a 1,000-chunk index.
- **NFR-2.3**: Followup loop state re-runs MUST NOT repeat intake classification.

### 6.3 Security, Privacy & Data Protection
- **NFR-3.1**: Patient health logs, journal entries, and prescription images MUST be stored locally within the user's SQLite database (`db/mediplus.db`).
- **NFR-3.2**: API credentials (`GROQ_API_KEY`) MUST be stored strictly in environment files (`.env`) and never checked into source control.

### 6.4 Reliability & Maintainability
- **NFR-4.1**: If external LLM or Whisper models fail to load or experience network timeouts, fallback deterministic handlers MUST catch exceptions gracefully.
- **NFR-4.2**: The codebase MUST adhere to PEP 8 standards with modular agent separation under `agents/`.

---

## 7. Data Models & Database Specifications

### 7.1 SQLite Database Schema (`db/mediplus.db`)

#### 1. Table: `prescriptions`
```sql
CREATE TABLE prescriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drug_name TEXT NOT NULL,
    dose TEXT,
    frequency TEXT,
    duration TEXT,
    status TEXT NOT NULL DEFAULT 'pending', -- 'pending', 'confirmed', 'archived'
    confidence_score REAL,
    raw_ocr_text TEXT,
    drug_candidate TEXT,
    drug_edit_distance INTEGER,
    drug_match_confidence REAL,
    warnings TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confirmed_at TIMESTAMP,
    user_id TEXT,
    image_path TEXT,
    notes TEXT
);
CREATE INDEX idx_prescriptions_status ON prescriptions(status);
CREATE INDEX idx_prescriptions_user ON prescriptions(user_id);
```

#### 2. Table: `medication_schedule`
```sql
CREATE TABLE medication_schedule (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prescription_id INTEGER NOT NULL,
    time_of_day TEXT NOT NULL,  -- e.g. '08:00', '14:00', '20:00'
    frequency_label TEXT,       -- e.g. 'Morning', 'Afternoon', 'Evening'
    start_date DATE NOT NULL,
    end_date DATE,
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prescription_id) REFERENCES prescriptions(id)
);
CREATE INDEX idx_schedule_prescription ON medication_schedule(prescription_id);
CREATE INDEX idx_schedule_active ON medication_schedule(is_active);
```

#### 3. Table: `adherence_logs`
```sql
CREATE TABLE adherence_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prescription_id INTEGER NOT NULL,
    schedule_id INTEGER NOT NULL,
    scheduled_time TIMESTAMP NOT NULL,
    taken_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'missed', -- 'taken', 'missed', 'skipped', 'snoozed'
    notes TEXT,
    delay_minutes INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prescription_id) REFERENCES prescriptions(id),
    FOREIGN KEY (schedule_id) REFERENCES medication_schedule(id)
);
CREATE INDEX idx_adherence_prescription ON adherence_logs(prescription_id);
CREATE INDEX idx_adherence_scheduled ON adherence_logs(scheduled_time);
```

#### 4. Table: `journal_entries`
```sql
CREATE TABLE journal_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    text TEXT NOT NULL,
    sentiment_score REAL,  -- Range: -1.0 to +1.0
    sentiment_label TEXT,   -- 'positive', 'neutral', 'negative'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    word_count INTEGER,
    language TEXT
);
CREATE INDEX idx_journal_user ON journal_entries(user_id);
CREATE INDEX idx_journal_created ON journal_entries(created_at);
```

#### 5. Table: `facility_cache` *(Planned)*
```sql
CREATE TABLE facility_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    region_key TEXT NOT NULL UNIQUE,  -- e.g. "lat:28.61,lon:77.20,r:5"
    results_json TEXT NOT NULL,
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_facility_region ON facility_cache(region_key);
```

---

## 8. Verification & Testing Status

### 8.1 Automated Test Execution Results

| Test Suite File | Focus Area | Cases | Result | Key Verified Behavior |
| :--- | :--- | :---: | :---: | :--- |
| `tests/test_followup_loop.py` | Graph Loop Engine | 7 | **PASS (100%)** | Verified cyclic routing on `new_adherence_log=True`, updater node execution, and clean loop termination at `END`. |
| `tests/test_intake.py` | Intent Classification | 10 | **PASS (100%)** | Evaluated all 8 intent classes + unclear; 100% precision with zero misclassifications. |
| `tests/test_prescription_ocr.py` | OCR & Drug Parsing | Suite | **PASS** | Evaluated dosage/frequency regex, Levenshtein matching on 400+ drugs, and sample image extraction (`fewer.jpg`). |
| `tests/test_graph_routing.py` | StateGraph Routing | 2 | **PASS** | Validated health query routing to `health_literacy_node` and facility query fallback to placeholder node. |
| `tests/test_voice.py` | Speech Transcription | 1 | **PASS** | Validated whisper transcription pipeline connectivity. |
| `tests/test_llm.py` | Groq LLM Client | 1 | **PASS** | Validated ChatGroq authentication and prompt completion. |

---

## 9. Traceability Matrix & Evolution Roadmap

```
+--------------------------------------------------------------------------+
|                        MEDIPATH ROADMAP STATUS                           |
+--------------------------------------------------------------------------+
| [PHASE 1] Core Foundations & Audio/LLM Intake         --> COMPLETED     |
| [PHASE 2] Health RAG & Curated Corpus Pipeline        --> COMPLETED     |
| [PHASE 3] Prescription OCR & Scheduling DB Schema     --> COMPLETED     |
| [PHASE 4] Adherence ML Model & Mental Health Journal  --> COMPLETED     |
| [PHASE 5] Nutrition RAG & Food Transfer Learning      --> COMPLETED     |
| [PHASE 6] Fitness MET Engine & Care Summary Synthesizer-> COMPLETED     |
| [PHASE 7] Cyclic Followup Graph Loops & Integration   --> COMPLETED     |
| ------------------------------------------------------------------------ |
| [PHASE 8] Overpass API Facility Finder & Cache Table  --> NEXT PHASE    |
| [PHASE 9] Government Scheme PDF Criteria & Evaluator  --> NEXT PHASE    |
| [PHASE 10] Multi-Modal User Interface (ui/)           --> NEXT PHASE    |
+--------------------------------------------------------------------------+
```

### Document Sign-Off Note
This Software Requirements Specification will be revised and appended with the exact request/response schemas for the OpenStreetMap Overpass facility query and Government Scheme criteria extractor as those milestones are delivered.

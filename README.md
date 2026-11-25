# SalesPilot AI — Autonomous AI Sales Voice Agent

**SalesPilot AI** is an end-to-end AI-powered Sales Voice & Lead Qualification MVP built with **Python, Streamlit, Google Gemini (`google-genai`), scikit-learn TF-IDF RAG, SpeechRecognition, gTTS, Plotly, and SQLite**.

---

## 1. Business Problem

Growing businesses lose revenue every day due to:
- **Delayed customer responses** during peak hours or after-hours.
- **Manual lead qualification** wasting sales rep time on low-intent enquiries.
- **Repeated customer queries** about pricing, plans, and refund policies.
- **Missed follow-ups** and inconsistent CRM note-taking.
- **Manual appointment scheduling** causing friction before demos are booked.

SalesPilot AI automates the entire initial sales conversation from greeting to CRM logging and demo booking.

---

## 2. Key Features

- **Bilingual Voice & Text Agent (English & Hindi):** Type or record audio directly in the browser (`st.audio_input` / WAV upload), transcribe via `SpeechRecognition`, and listen to AI replies synthesized via `gTTS`.
- **Grounded RAG Knowledge Base:** Loads `data/sample_knowledge.txt` and user-uploaded `.txt`/`.pdf` documents, chunks them, builds a **TF-IDF + Cosine Similarity** index (`scikit-learn`), and displays retrieved snippets transparently.
- **Gemini AI + Deterministic Demo Fallback:** Uses the official `google-genai` SDK when `GEMINI_API_KEY` is provided, and seamlessly switches to a clearly labelled **Demo Fallback Mode** when no API key is configured or if offline.
- **Automated Lead Qualification (0–100 Scoring):** Extracts Name, Email, Phone, Company, Product Interest, Budget, Timeline, and Requirements; classifies leads as **Hot (70–100)**, **Warm (40–69)**, or **Cold (0–39)** with a clear signal breakdown.
- **Real-Time Sentiment Analysis:** Classifies customer messages as **Positive**, **Neutral**, or **Negative** with negation awareness.
- **Prototype Demo Appointment Scheduler:** Displays available slots, prevents duplicate bookings, and links confirmed demos directly to CRM leads in SQLite.
- **Executive CRM, Call Summaries & Plotly Analytics:** Automatically writes structured 8-field call summaries, searchable CRM tables, CSV exports, and interactive Plotly dashboards.

---

## 3. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Frontend / UI | Streamlit |
| LLM SDK | Google GenAI Python SDK (`google-genai`) |
| RAG Engine | `scikit-learn` (TF-IDF + Cosine Similarity) & `pypdf` |
| Speech-to-Text | `SpeechRecognition` |
| Text-to-Speech | `gTTS` |
| Database | SQLite (`sqlite3`) |
| Visualization & Data | `Plotly`, `Pandas` |
| Configuration | `python-dotenv` |

---

## 4. Installation & Running Locally

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables (Optional for Demo Mode)
Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```
Add your Gemini API key inside `.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```
> **Note:** If `GEMINI_API_KEY` is left blank, SalesPilot AI automatically runs in **Demo Fallback Mode** so every workflow still works end-to-end.

### Step 3: Run the Application
```bash
streamlit run app.py
```
Open the local URL printed in the terminal (typically `http://localhost:8501`).

---

## 5. Complete End-to-End Workflow

1. **Customer Interaction:** User speaks into the microphone (or types / clicks a Quick Demo Scenario) in English or Hindi.
2. **Speech-to-Text:** Recorded WAV audio is transcribed via `SpeechRecognition`.
3. **Sentiment & Lead Extraction:** The message is analyzed for sentiment and scanned for contact, budget, product interest, timeline, and requirement signals, updating the 0–100 score in SQLite.
4. **RAG Retrieval:** The top-3 relevant chunks are retrieved from the TF-IDF knowledge base and displayed in an expandable inspector.
5. **AI Response Generation:** Gemini (or Demo Fallback) crafts a natural, polite sales reply grounded in retrieved chunks and conversation memory.
6. **Text-to-Speech:** `gTTS` converts the reply into playable MP3 audio.
7. **Appointment Booking & Call Summary:** Qualified prospects book an open demo slot; an 8-field structured summary and recommended next action are persisted to SQLite and visible in the **CRM & Leads** and **Call Analytics** dashboards.

---

## 6. Database Design (SQLite)

- `leads` — Contact details, qualification attributes, `lead_score`, `lead_status`, `score_explanation`, `conversation_summary`, `recommended_action`, `created_at`
- `conversations` — Per-turn `lead_id` (FK), `customer_message`, `ai_response`, `sentiment`, `timestamp`
- `appointments` — `lead_id` (FK), `appointment_date`, `appointment_time`, `status`, `created_at`
- `documents` — `filename`, `content`, `created_at`

---

## 7. Architecture & Documentation

- [System Architecture](docs/ARCHITECTURE.md)
- [Scalability to 100,000 Users](docs/SCALABILITY.md)
- [Security Specification](docs/SECURITY.md)
- [Cost Estimation & Commercial Feasibility](docs/COST_ESTIMATION.md)
- [Technical Interview Preparation](INTERVIEW_PREPARATION.md)

---

## 8. Known Limitations & Future Improvements

- **MVP Limitations:** Uses local SQLite (single-node), in-memory TF-IDF (keyword/n-gram matching rather than dense semantic embeddings), Google's free web STT/TTS endpoints (require internet access), and an internal prototype slot scheduler rather than Google Calendar OAuth.
- **Future Improvements:** Migrate to FastAPI + React, PostgreSQL + `pgvector`, streaming Whisper STT + low-latency neural TTS, Twilio/SIP telephony integration, and HubSpot/Salesforce CRM webhooks.

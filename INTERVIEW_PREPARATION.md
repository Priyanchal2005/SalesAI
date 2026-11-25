# SalesPilot AI — Technical Interview Preparation Guide

This guide equips you to confidently explain every layer of **SalesPilot AI** during a technical evaluation round.

---

## 1. Elevator Pitch & System Overview
> "SalesPilot AI is an autonomous bilingual (English & Hindi) AI Sales Voice Agent MVP built with Python, Streamlit, Google Gemini (`google-genai`), TF-IDF RAG, SpeechRecognition, gTTS, Plotly, and SQLite. It automates initial inbound sales enquiries from voice/text input through RAG-grounded answers, real-time 0–100 lead scoring, sentiment tracking, duplicate-safe demo booking, and structured CRM call summaries."

---

## 2. How Each Core Component Works

### A. End-to-End Workflow
1. **Input**: Customer types text or records audio (`st.audio_input` produces WAV bytes).
2. **STT (`modules/voice_engine.py`)**: `SpeechRecognition` decodes WAV bytes using `en-IN` or `hi-IN` locale.
3. **Sentiment (`modules/sentiment.py`)**: Rule-based lexicon + negation detector labels turn as `Positive`, `Neutral`, or `Negative`.
4. **Lead Qualification (`modules/lead_qualification.py`)**: Extracts 8 structured fields using Gemini JSON mode (with regex/keyword fallback) and computes a transparent 0–100 score.
5. **RAG (`modules/rag_engine.py`)**: Splits docs into 450-char overlapping chunks, builds a TF-IDF matrix (`TfidfVectorizer`), and ranks chunks via `cosine_similarity`.
6. **LLM (`modules/ai_engine.py`)**: Passes system prompt, known lead profile, chat history, and top-3 RAG chunks to Gemini (`gemini-2.5-flash`)—or uses deterministic Demo Fallback if unkeyed/offline.
7. **Persistence & Summary (`database.py`, `modules/summarizer.py`)**: Saves turn + updated lead + structured 8-field executive summary in SQLite.
8. **TTS (`modules/voice_engine.py`)**: Strips markdown and synthesizes MP3 playback via `gTTS`.

### B. How TF-IDF & Cosine Similarity Work
- **TF (Term Frequency)**: How often a word/bigram appears in a chunk.
- **IDF (Inverse Document Frequency)**: Downweights common words across all chunks and boosts rare, discriminative terms (e.g., `"$149"`, `"refund"`, `"enterprise"`).
- **Cosine Similarity**: Measures the cosine of the angle between the query TF-IDF vector and each chunk vector ($\text{sim} \in [0, 1]$). Value `1.0` means identical direction; `0.0` means zero term overlap.
- **Why TF-IDF for a 2-hour MVP?** Zero external vector DB setup, deterministic, instant indexing in milliseconds, and completely free/offline.

### C. How Contextual Memory Works
- **Short-term UI memory**: `st.session_state.chat_history` keeps the active turn-by-turn dialogue.
- **Structured entity memory**: Extracted fields (`name`, `company`, `budget`, `timeline`, etc.) are persisted in SQLite `leads` and injected into every Gemini prompt as `Known Lead Profile`, preventing repetitive questioning.

### D. Lead-Scoring Algorithm (0–100)
- Clear product interest: **+20**
- Budget provided: **+20**
- Purchase timeline within 30 days: **+25** (`+10` if timeline > 30 days)
- Business requirements provided: **+20**
- Contact info (email or phone) provided: **+15**
- **Classification**: `70–100` = **Hot**, `40–69` = **Warm**, `0–39` = **Cold**.

### E. Database Relationships
- `leads` (1) $\rightarrow$ (N) `conversations` via `lead_id` (foreign key, `ON DELETE CASCADE`)
- `leads` (1) $\rightarrow$ (N) `appointments` via `lead_id` (foreign key, `ON DELETE CASCADE`)
- `documents` stores raw text of uploaded TXT/PDF files for RAG re-indexing across restarts.

---

## 3. Common Technical Interview Questions & Strong Answers

**Q1: Why did you choose TF-IDF instead of dense embeddings + Pinecone/FAISS?**
> "For a rapid 2-hour local MVP, TF-IDF + cosine similarity gives sub-10ms retrieval with zero external API latency, zero cost, and deterministic explainability. For production scale (`docs/SCALABILITY.md`), we upgrade cleanly behind the same `RAGEngine.retrieve()` interface to dense embeddings + `pgvector` or Qdrant for semantic synonym matching."

**Q2: How do you prevent LLM hallucinations on pricing or refund policies?**
> "Two layers: First, we retrieve the top-3 matching chunks from our verified knowledge base and inject them into the prompt with a strict instruction never to invent prices or policies outside the context. Second, we expose the retrieved chunks and cosine scores directly in the UI expander so evaluators and sales ops can audit every answer."

**Q3: What happens if the Gemini API key is missing or fails mid-demo?**
> "`modules/ai_engine.py` catches missing keys and API exceptions and switches automatically to a clearly labelled **Demo Fallback Mode** (`mode='demo_fallback'`). It still runs RAG retrieval, injects the highest-scoring chunk excerpt, tracks missing qualification fields, and responds contextually in English or Hindi."

**Q4: How do you prevent double-booking appointment slots?**
> "`database.book_appointment()` runs a parameterized check against `appointments` for `(appointment_date, appointment_time)` where `status != 'Cancelled'` inside a transaction before inserting. In production with multiple workers, we add a PostgreSQL unique partial index and a Redis distributed lock."

**Q5: Can this Streamlit prototype handle 100,000 concurrent users?**
> "No—and we explicitly document that distinction in `docs/SCALABILITY.md`. Streamlit + SQLite is built for local single-node evaluation. To serve **100,000 registered users**, we separate into a React frontend + stateless FastAPI cluster on AWS ECS, Amazon RDS PostgreSQL with read replicas, Redis for session state/slot locks, SQS/Kafka for async summarization and indexing, and dedicated STT/TTS/vector services."

**Q6: How do you protect customer PII and prevent SQL injection?**
> "All SQL queries in `database.py` use parameterized `?` bindings, never string formatting. API keys stay in `.env` (gitignored). Rendered HTML in Streamlit cards uses `html.escape()` to prevent XSS. Production security (`docs/SECURITY.md`) adds TLS 1.3, AES-256 encryption at rest, RBAC, and GDPR/DPDP retention policies."

**Q7: How does your cost model stay commercially viable?**
> "By pairing a Flash-tier LLM with short retrieved chunks (~1,500 input tokens/turn) and asynchronous summaries, a 5-turn qualified conversation costs roughly \$0.025–\$0.060 (`docs/COST_ESTIMATION.md`). Compared to manual tier-1 SDR qualification, automation cuts first-response latency to <2 seconds and preserves >60% SaaS gross margins on our \$149/month Growth plan."

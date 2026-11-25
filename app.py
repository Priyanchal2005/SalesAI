"""SalesPilot AI — Autonomous AI Sales Voice Agent (Streamlit MVP)."""

from datetime import datetime
import hashlib
import html
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
import database
from modules.ai_engine import generate_sales_response
from modules.appointment import get_available_dates, get_slot_availability, schedule_demo
from modules.lead_qualification import extract_and_score_lead
from modules.rag_engine import RAGEngine, extract_text_from_upload
from modules.sentiment import analyze_sentiment
from modules.summarizer import generate_call_summary
from modules.ui_components import (
    inject_custom_css,
    render_hero,
    sentiment_badge,
    status_badge,
)
from modules.voice_engine import synthesize_speech, transcribe_audio

st.set_page_config(
    page_title="SalesPilot AI | Autonomous AI Sales Voice Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()


@st.cache_resource
def _bootstrap_database() -> bool:
    """Initialize SQLite schema and sample data once per server startup."""
    database.init_db()
    return True


_bootstrap_database()


def _init_session_state() -> None:
    """Ensure all required Streamlit session state keys exist."""
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = RAGEngine()
    if "active_lead_id" not in st.session_state:
        st.session_state.active_lead_id = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "latest_chunks" not in st.session_state:
        st.session_state.latest_chunks = []
    if "latest_sentiment" not in st.session_state:
        st.session_state.latest_sentiment = {
            "label": "Neutral",
            "score": 0.0,
            "confidence": 0.6,
            "matched_signals": [],
        }
    if "latest_mode" not in st.session_state:
        st.session_state.latest_mode = "gemini" if config.GEMINI_API_KEY else "demo_fallback"
    if "latest_mode_note" not in st.session_state:
        st.session_state.latest_mode_note = ""
    if "latest_summary" not in st.session_state:
        st.session_state.latest_summary = None
    if "last_audio_hash" not in st.session_state:
        st.session_state.last_audio_hash = None
    if "language" not in st.session_state:
        st.session_state.language = "English"
    if "auto_tts" not in st.session_state:
        st.session_state.auto_tts = True
    if "latest_audio_bytes" not in st.session_state:
        st.session_state.latest_audio_bytes = None
    if "latest_tts_error" not in st.session_state:
        st.session_state.latest_tts_error = None


_init_session_state()


def _reset_conversation() -> None:
    """Start a brand-new prospect conversation."""
    st.session_state.active_lead_id = None
    st.session_state.chat_history = []
    st.session_state.latest_chunks = []
    st.session_state.latest_sentiment = {
        "label": "Neutral",
        "score": 0.0,
        "confidence": 0.6,
        "matched_signals": [],
    }
    st.session_state.latest_summary = None
    st.session_state.latest_audio_bytes = None
    st.session_state.latest_tts_error = None


def _process_customer_turn(user_text: str) -> None:
    """Execute full SalesPilot AI pipeline: Sentiment -> RAG -> Lead Qual -> LLM -> DB -> TTS."""
    user_text = user_text.strip()
    if not user_text:
        return

    # 1. Sentiment analysis
    sentiment_res = analyze_sentiment(user_text)
    st.session_state.latest_sentiment = sentiment_res

    # 2. Ensure a lead record exists in SQLite
    lead_id = st.session_state.active_lead_id
    if lead_id is None:
        lead_id = database.create_lead(name="New Prospect", lead_status="Cold", lead_score=0)
        st.session_state.active_lead_id = lead_id

    current_lead = database.get_lead(lead_id) or {}

    # 3. Lead extraction & scoring
    updated_lead = extract_and_score_lead(user_text, current_lead)
    if updated_lead.get("name") in (None, "", "New Prospect") and current_lead.get("name") == "New Prospect":
        updated_lead["name"] = "New Prospect"

    database.update_lead(
        lead_id,
        name=updated_lead.get("name", "New Prospect"),
        email=updated_lead.get("email", ""),
        phone=updated_lead.get("phone", ""),
        company=updated_lead.get("company", ""),
        product_interest=updated_lead.get("product_interest", ""),
        budget=updated_lead.get("budget", ""),
        timeline=updated_lead.get("timeline", ""),
        requirements=updated_lead.get("requirements", ""),
        lead_score=updated_lead.get("lead_score", 0),
        lead_status=updated_lead.get("lead_status", "Cold"),
        score_explanation=updated_lead.get("score_explanation", ""),
    )
    fresh_lead = database.get_lead(lead_id) or updated_lead

    # 4. RAG retrieval
    retrieved_chunks = st.session_state.rag_engine.retrieve(user_text, top_k=config.TOP_K_CHUNKS)
    st.session_state.latest_chunks = retrieved_chunks

    # 5. AI response generation (Gemini or Demo Fallback)
    ai_result = generate_sales_response(
        user_message=user_text,
        chat_history=st.session_state.chat_history,
        retrieved_chunks=retrieved_chunks,
        lead_profile=fresh_lead,
        language=st.session_state.language,
        sentiment=sentiment_res["label"],
    )
    ai_reply = ai_result["reply"]
    st.session_state.latest_mode = ai_result["mode"]
    st.session_state.latest_mode_note = ""

    # 6. Persist conversation turn to SQLite
    database.add_conversation(
        lead_id=lead_id,
        customer_message=user_text,
        ai_response=ai_reply,
        sentiment=sentiment_res["label"],
    )

    # 7. Refresh structured summary in SQLite
    st.session_state.latest_summary = generate_call_summary(lead_id)

    # 8. Text-to-Speech (if enabled)
    audio_bytes = None
    tts_err = None
    if st.session_state.auto_tts:
        audio_bytes, tts_err = synthesize_speech(ai_reply, language=st.session_state.language)
    st.session_state.latest_audio_bytes = audio_bytes
    st.session_state.latest_tts_error = tts_err

    # 9. Append to session conversation history
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_text,
            "sentiment": sentiment_res["label"],
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
    )
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": ai_reply,
            "mode": ai_result["mode"],
            "model": ai_result["model"],
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
    )


def _apply_plotly_theme(fig: go.Figure, height: int = 320) -> go.Figure:
    """Apply consistent dark-navy Plotly layout."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.45)",
        font=dict(color="#E2E8F0", family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=42, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 8px 4px 14px 4px;">
            <div style="font-size:1.35rem; font-weight:800; background:linear-gradient(135deg,#60A5FA,#A78BFA); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                SalesPilot AI
            </div>
            <div style="font-size:0.78rem; color:#94A3B8; margin-top:2px;">
                Autonomous AI Sales Voice Agent
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "AI Voice Agent",
            "CRM & Leads",
            "Appointments",
            "Knowledge Base",
            "Call Analytics",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    sidebar_metrics = database.get_dashboard_metrics()
    st.markdown(
        f"""
        <div class="sp-card" style="margin-top:14px; padding:12px;">
            <div class="sp-muted">Pipeline Snapshot</div>
            <div style="margin-top:6px; font-size:0.9rem;">
                Leads: <b>{sidebar_metrics['total_leads']}</b> &nbsp;|&nbsp;
                Hot: <b>{sidebar_metrics['hot_leads']}</b><br/>
                Demos Booked: <b>{sidebar_metrics['booked_appointments']}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# PAGE 1: OVERVIEW
# ---------------------------------------------------------------------------
if page == "Overview":
    render_hero(
        "SalesPilot AI — Executive Overview",
        "Real-time inbound voice & chat qualification, RAG grounding, sentiment intelligence, and automated demo scheduling.",
    )

    metrics = database.get_dashboard_metrics()
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(
        f'<div class="sp-kpi"><div class="sp-kpi-label">Total Leads</div>'
        f'<div class="sp-kpi-val">{metrics["total_leads"]}</div>'
        f'<div class="sp-kpi-sub">{metrics["total_conversations"]} total turns logged</div></div>',
        unsafe_allow_html=True,
    )
    k2.markdown(
        f'<div class="sp-kpi"><div class="sp-kpi-label">Qualified Leads (40+)</div>'
        f'<div class="sp-kpi-val">{metrics["qualified_leads"]}</div>'
        f'<div class="sp-kpi-sub">{metrics["hot_leads"]} Hot leads ready to close</div></div>',
        unsafe_allow_html=True,
    )
    k3.markdown(
        f'<div class="sp-kpi"><div class="sp-kpi-label">Appointments Booked</div>'
        f'<div class="sp-kpi-val">{metrics["booked_appointments"]}</div>'
        f'<div class="sp-kpi-sub">Scheduled product demos</div></div>',
        unsafe_allow_html=True,
    )
    k4.markdown(
        f'<div class="sp-kpi"><div class="sp-kpi-label">Average Lead Score</div>'
        f'<div class="sp-kpi-val">{metrics["avg_score"]}</div>'
        f'<div class="sp-kpi-sub">Scale: 0 – 100</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        sent_counts = metrics["sentiment_counts"]
        sent_df = pd.DataFrame(
            {"Sentiment": list(sent_counts.keys()), "Count": list(sent_counts.values())}
        )
        fig_sent = px.pie(
            sent_df,
            names="Sentiment",
            values="Count",
            hole=0.55,
            title="Sentiment Distribution",
            color="Sentiment",
            color_discrete_map={
                "Positive": "#10B981",
                "Neutral": "#64748B",
                "Negative": "#F43F5E",
            },
        )
        st.plotly_chart(_apply_plotly_theme(fig_sent), width="stretch")

    with c2:
        leads = database.get_all_leads()
        status_counts = {"Hot": 0, "Warm": 0, "Cold": 0}
        for l in leads:
            status_counts[l.get("lead_status", "Cold")] = status_counts.get(l.get("lead_status", "Cold"), 0) + 1
        conv_df = pd.DataFrame(
            {"Stage": ["Cold (0-39)", "Warm (40-69)", "Hot (70-100)"], "Leads": [status_counts["Cold"], status_counts["Warm"], status_counts["Hot"]]}
        )
        fig_conv = px.bar(
            conv_df,
            x="Stage",
            y="Leads",
            title="Lead Conversion Overview",
            color="Stage",
            color_discrete_map={
                "Cold (0-39)": "#3B82F6",
                "Warm (40-69)": "#F59E0B",
                "Hot (70-100)": "#EF4444",
            },
        )
        fig_conv.update_layout(showlegend=False)
        st.plotly_chart(_apply_plotly_theme(fig_conv), width="stretch")

    st.subheader("Recent Conversations")
    recent = database.get_recent_conversations(limit=6)
    if not recent:
        st.info("No conversations recorded yet. Visit the AI Voice Agent page to start one.")
    else:
        for item in recent:
            lead_name = html.escape(item.get("lead_name") or "Prospect")
            company = html.escape(item.get("lead_company") or "")
            cust_msg = html.escape(item["customer_message"])
            ai_msg = html.escape(item["ai_response"])
            st.markdown(
                f"""
                <div class="sp-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <div><b>{lead_name}</b> {f'· <span class="sp-muted">{company}</span>' if company else ''}</div>
                        <div>{sentiment_badge(item['sentiment'])} &nbsp; {status_badge(item.get('lead_status') or 'Cold')} &nbsp; <span class="sp-muted">{item['timestamp']}</span></div>
                    </div>
                    <div style="font-size:0.9rem; color:#CBD5E1;"><b>Customer:</b> {cust_msg}</div>
                    <div style="font-size:0.88rem; color:#A5B4FC; margin-top:4px;"><b>SalesPilot AI:</b> {ai_msg}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# PAGE 2: AI VOICE AGENT
# ---------------------------------------------------------------------------
elif page == "AI Voice Agent":
    render_hero(
        "AI Voice & Conversational Sales Agent",
        "Speak or type in English or Hindi. Every turn runs RAG retrieval, sentiment detection, lead scoring, and SQLite persistence.",
    )

    # Top controls row
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1.5])
    with ctrl1:
        st.session_state.language = st.selectbox(
            "Conversation Language",
            list(config.SUPPORTED_LANGUAGES.keys()),
            index=0 if st.session_state.language == "English" else 1,
        )
    with ctrl2:
        st.session_state.auto_tts = st.toggle(
            "Auto Voice Reply (gTTS)",
            value=st.session_state.auto_tts,
            help="Automatically synthesize spoken MP3 response after each AI reply.",
        )
    with ctrl3:
        st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
        if st.button("New Conversation", width="stretch"):
            _reset_conversation()
            st.rerun()

    # Quick demo scenario buttons for live evaluation
    st.caption("Quick Demo Scenarios (click to simulate a customer message):")
    q1, q2, q3, q4 = st.columns(4)
    if q1.button("1. Ask Pricing", width="stretch"):
        _process_customer_turn("Hi! What are your pricing plans and do you support Hindi voice?")
        st.rerun()
    if q2.button("2. Share Lead Info", width="stretch"):
        _process_customer_turn(
            "I am Neha Verma from CloudPulse India (neha@cloudpulse.in, +91-9811223344). "
            "We need the Growth Plan for automating inbound sales leads, budget is $150/month, and want to start within 15 days."
        )
        st.rerun()
    if q3.button("3. Refund Policy", width="stretch"):
        _process_customer_turn("Wait, what if it doesn't work for us? What is your refund and cancellation policy?")
        st.rerun()
    if q4.button("4. Hindi Inquiry", width="stretch"):
        st.session_state.language = "Hindi"
        _process_customer_turn("Namaste, mujhe Growth Plan aur demo booking ke baare mein jaankari chahiye.")
        st.rerun()

    left_col, right_col = st.columns([1.65, 1.0], gap="large")

    with left_col:
        st.subheader("Conversation")
        if not st.session_state.chat_history:
            st.markdown(
                """
                <div class="sp-card">
                    <b>SalesPilot AI:</b> Hello! I'm SalesPilot AI from Fortivex Solutions.
                    Ask me about our plans, features, ROI, or refund policy—or record a voice message below to get started!
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.chat_history:
                safe_text = html.escape(msg["content"]).replace("\n", "<br>")
                if msg["role"] == "user":
                    st.markdown(
                        f"""
                        <div class="sp-chat-user">
                            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                                <b>Customer</b>
                                <span>{sentiment_badge(msg.get('sentiment', 'Neutral'))} &nbsp; <span class="sp-muted">{msg.get('timestamp', '')}</span></span>
                            </div>
                            <div>{safe_text}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="sp-chat-ai">
                            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                                <b>SalesPilot AI</b>
                                <span class="sp-muted">{msg.get('timestamp', '')}</span>
                            </div>
                            <div>{safe_text}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Latest AI audio playback
        if st.session_state.latest_audio_bytes:
            st.markdown("**AI Voice Response (gTTS):**")
            st.audio(st.session_state.latest_audio_bytes, format="audio/mp3")
        elif st.session_state.latest_tts_error:
            st.warning(st.session_state.latest_tts_error)
        elif st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
            if st.button("🔊 Play Latest Response Audio"):
                audio_bytes, tts_err = synthesize_speech(
                    st.session_state.chat_history[-1]["content"],
                    language=st.session_state.language,
                )
                if audio_bytes:
                    st.session_state.latest_audio_bytes = audio_bytes
                    st.session_state.latest_tts_error = None
                    st.rerun()
                else:
                    st.warning(tts_err or "Audio synthesis failed.")

        st.divider()

        # Voice recording & audio upload
        v_col1, v_col2 = st.columns([1.4, 1])
        with v_col1:
            recorded_audio = None
            if hasattr(st, "audio_input"):
                recorded_audio = st.audio_input("🎤 Record Voice Message")
            else:
                st.caption("Microphone widget unavailable in this Streamlit version; upload a WAV clip instead.")
        with v_col2:
            uploaded_audio = st.file_uploader(
                "Or Upload WAV Audio",
                type=["wav"],
                help="Upload a PCM WAV file for Speech-to-Text transcription.",
            )

        audio_source = recorded_audio or uploaded_audio
        if audio_source is not None:
            raw_audio = audio_source.getvalue()
            audio_hash = hashlib.sha256(raw_audio).hexdigest()
            if audio_hash != st.session_state.last_audio_hash:
                st.session_state.last_audio_hash = audio_hash
                with st.spinner("Transcribing audio via SpeechRecognition..."):
                    transcript, stt_err = transcribe_audio(raw_audio, language=st.session_state.language)
                if transcript:
                    st.success(f"Transcribed: \"{transcript}\"")
                    _process_customer_turn(transcript)
                    st.rerun()
                else:
                    st.warning(stt_err or "Could not transcribe audio.")

        # Text input + Send button
        with st.form("chat_input_form", clear_on_submit=True):
            t1, t2 = st.columns([4.5, 1])
            with t1:
                user_msg = st.text_input(
                    "Type your message",
                    placeholder="Ask about pricing, features, refund policy, or share your requirements...",
                    label_visibility="collapsed",
                )
            with t2:
                submitted = st.form_submit_button("Send", width="stretch")
            if submitted and user_msg.strip():
                _process_customer_turn(user_msg)
                st.rerun()

        # Expandable RAG retrieval inspector
        with st.expander("🔍 Retrieved RAG Knowledge Snippets (TF-IDF + Cosine Similarity)", expanded=False):
            if not st.session_state.latest_chunks:
                st.caption("Send a message to inspect the retrieved company knowledge chunks.")
            else:
                for idx, chunk in enumerate(st.session_state.latest_chunks, start=1):
                    st.markdown(
                        f"**#{idx} — `{chunk['filename']}` (Chunk {chunk['chunk_index']}) | "
                        f"Cosine Similarity: `{chunk['score']:.4f}`**"
                    )
                    st.code(chunk["text"], language="text")

    # Right column: Sentiment, Lead Qualification, Quick Demo Booking, Call Summary
    with right_col:
        sent = st.session_state.latest_sentiment
        signals_str = ", ".join(sent.get("matched_signals") or []) or "Baseline context"
        st.markdown(
            f"""
            <div class="sp-card">
                <div class="sp-kpi-label">Live Sentiment Indicator</div>
                <div style="margin-top:8px; display:flex; justify-content:space-between; align-items:center;">
                    {sentiment_badge(sent['label'])}
                    <span class="sp-muted">Confidence: {int(sent.get('confidence', 0.6) * 100)}%</span>
                </div>
                <div class="sp-muted" style="margin-top:6px;">Signals: {html.escape(signals_str)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Live Lead Qualification Panel
        active_lead = (
            database.get_lead(st.session_state.active_lead_id)
            if st.session_state.active_lead_id
            else None
        )
        st.markdown('<div class="sp-card">', unsafe_allow_html=True)
        st.markdown('<div class="sp-kpi-label">Live Lead Qualification</div>', unsafe_allow_html=True)
        if not active_lead:
            st.caption("Start conversing to capture lead signals and calculate the 0–100 score.")
        else:
            score_val = int(active_lead.get("lead_score") or 0)
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; align-items:center; margin:8px 0;'>"
                f"<span style='font-size:1.35rem; font-weight:800;'>Score: {score_val}/100</span>"
                f"{status_badge(active_lead.get('lead_status', 'Cold'))}</div>",
                unsafe_allow_html=True,
            )
            st.progress(min(max(score_val, 0), 100) / 100.0)
            st.markdown(
                f"""
                - **Name:** {active_lead.get('name') or '—'}
                - **Email:** {active_lead.get('email') or '—'}
                - **Phone:** {active_lead.get('phone') or '—'}
                - **Company:** {active_lead.get('company') or '—'}
                - **Product Interest:** {active_lead.get('product_interest') or '—'}
                - **Budget:** {active_lead.get('budget') or '—'}
                - **Timeline:** {active_lead.get('timeline') or '—'}
                - **Requirements:** {active_lead.get('requirements') or '—'}
                """
            )
            with st.expander("Score Explanation"):
                st.write(active_lead.get("score_explanation") or "No signals yet.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Quick Demo Booking Panel
        with st.expander("📅 Quick Demo Booking", expanded=True):
            st.caption("Select an available date and time slot to schedule a live demo.")
            if not active_lead:
                st.caption("Send at least one message to create a lead before booking a slot.")
            else:
                avail_dates = get_available_dates(7)
                sel_date = st.selectbox("Select Date", avail_dates, key="va_appt_date")
                slots_info = get_slot_availability(sel_date)
                free_slots = [s["time"] for s in slots_info if s["available"]]
                if not free_slots:
                    st.warning("All slots are booked for this date. Choose another date.")
                else:
                    sel_slot = st.selectbox("Available Time Slot", free_slots, key="va_appt_time")
                    if st.button("Book Demo Slot", width="stretch"):
                        ok, msg = schedule_demo(active_lead["id"], sel_date, sel_slot)
                        if ok:
                            st.success(msg)
                            st.session_state.latest_summary = generate_call_summary(active_lead["id"])
                        else:
                            st.error(msg)

        # Call Summary Panel
        with st.expander("📝 Structured Call Summary", expanded=True):
            if active_lead and st.button("Refresh / Generate Call Summary", width="stretch"):
                st.session_state.latest_summary = generate_call_summary(active_lead["id"])

            summary = st.session_state.latest_summary
            if not summary:
                st.caption("Structured summary will appear automatically once the conversation starts.")
            else:
                st.markdown(
                    f"""
                    - **Product Interest:** {summary['product_interest']}
                    - **Budget:** {summary['budget']}
                    - **Requirements:** {summary['requirements']}
                    - **Sentiment:** {summary['sentiment']}
                    - **Lead Score:** {summary['lead_score']}/100 ({summary['qualification_status']})
                    - **Appointment:** {summary['appointment_details']}
                    - **Recommended Next Action:** {summary['recommended_next_action']}
                    """
                )
                st.info(summary["executive_summary"])


# ---------------------------------------------------------------------------
# PAGE 3: CRM & LEADS
# ---------------------------------------------------------------------------
elif page == "CRM & Leads":
    render_hero(
        "CRM & Lead Pipeline",
        "Search, filter, inspect conversation transcripts, and review structured AI call summaries stored in SQLite.",
    )

    f1, f2, f3 = st.columns([2.5, 1.2, 1])
    with f1:
        search_q = st.text_input("Search leads", placeholder="Search by name, company, email, product, or requirement...")
    with f2:
        status_filter = st.selectbox("Filter by Status", ["All", "Hot", "Warm", "Cold"])
    with f3:
        all_leads_export = database.get_all_leads(search=search_q, status_filter=status_filter)
        csv_bytes = pd.DataFrame(all_leads_export).to_csv(index=False).encode("utf-8") if all_leads_export else b""
        st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
        st.download_button(
            "Export CSV",
            data=csv_bytes,
            file_name="salespilot_leads.csv",
            mime="text/csv",
            disabled=not bool(all_leads_export),
            width="stretch",
        )

    leads = all_leads_export
    if not leads:
        st.info("No matching leads found.")
    else:
        display_df = pd.DataFrame(leads)[
            [
                "id",
                "name",
                "company",
                "email",
                "phone",
                "product_interest",
                "budget",
                "timeline",
                "lead_score",
                "lead_status",
                "created_at",
            ]
        ]
        st.dataframe(display_df, width="stretch", hide_index=True)

        st.divider()
        st.subheader("Lead Inspector & Call Summary")
        lead_options = {
            f"#{l['id']} — {l['name'] or 'Unnamed'} ({l['company'] or 'No Company'}) [{l['lead_status']} · {l['lead_score']}]": l["id"]
            for l in leads
        }
        selected_label = st.selectbox("Select Lead to Inspect", list(lead_options.keys()))
        sel_id = lead_options[selected_label]
        sel_lead = database.get_lead(sel_id)

        if sel_lead:
            d1, d2 = st.columns([1.1, 1.2])
            with d1:
                st.markdown('<div class="sp-card">', unsafe_allow_html=True)
                st.markdown(
                    f"### {html.escape(sel_lead.get('name') or 'Unnamed Lead')} &nbsp; {status_badge(sel_lead.get('lead_status', 'Cold'))}",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    - **Company:** {sel_lead.get('company') or '—'}
                    - **Email:** {sel_lead.get('email') or '—'}
                    - **Phone:** {sel_lead.get('phone') or '—'}
                    - **Product Interest:** {sel_lead.get('product_interest') or '—'}
                    - **Budget:** {sel_lead.get('budget') or '—'}
                    - **Timeline:** {sel_lead.get('timeline') or '—'}
                    - **Requirements:** {sel_lead.get('requirements') or '—'}
                    - **Lead Score:** {sel_lead.get('lead_score', 0)}/100
                    """
                )
                st.caption(f"Score Signals: {sel_lead.get('score_explanation') or 'N/A'}")
                if st.button("Regenerate Call Summary", key=f"regen_{sel_id}"):
                    generate_call_summary(sel_id)
                    st.success("Call summary updated in SQLite.")
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            with d2:
                st.markdown('<div class="sp-card">', unsafe_allow_html=True)
                st.markdown("#### Executive Call Summary & Next Action")
                st.write(sel_lead.get("conversation_summary") or "No summary generated yet.")
                st.success(f"Recommended Next Action: {sel_lead.get('recommended_action') or 'Follow up with prospect.'}")

                appts = database.get_appointments_for_lead(sel_id)
                st.markdown("#### Linked Appointments")
                if not appts:
                    st.caption("No appointments scheduled for this lead.")
                else:
                    for a in appts:
                        st.write(f"- **{a['appointment_date']}** at **{a['appointment_time']}** — `{a['status']}`")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("#### Conversation History")
            convs = database.get_conversations_for_lead(sel_id)
            if not convs:
                st.caption("No messages logged for this lead.")
            else:
                for c in convs:
                    st.markdown(
                        f"""
                        <div class="sp-card">
                            <div style="display:flex; justify-content:space-between;">
                                <b>Customer</b>
                                <span>{sentiment_badge(c['sentiment'])} &nbsp; <span class="sp-muted">{c['timestamp']}</span></span>
                            </div>
                            <div style="margin:4px 0 8px 0;">{html.escape(c['customer_message'])}</div>
                            <div style="color:#A5B4FC;"><b>SalesPilot AI:</b> {html.escape(c['ai_response'])}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# PAGE 4: APPOINTMENTS
# ---------------------------------------------------------------------------
elif page == "Appointments":
    render_hero(
        "Appointment Scheduler",
        "Schedule and manage product demo slots with real-time availability and duplicate-booking protection.",
    )

    col_book, col_avail = st.columns([1.15, 1])

    with col_avail:
        st.subheader("Slot Availability")
        check_date = st.date_input("Inspect Date", value=datetime.now().date())
        check_date_str = check_date.strftime("%Y-%m-%d")
        slots = get_slot_availability(check_date_str)
        for s in slots:
            badge = (
                '<span class="sp-badge badge-pos">Available</span>'
                if s["available"]
                else '<span class="sp-badge badge-neg">Booked</span>'
            )
            st.markdown(
                f"<div class='sp-card' style='padding:10px 14px; display:flex; justify-content:space-between; align-items:center;'>"
                f"<b>{s['time']}</b> {badge}</div>",
                unsafe_allow_html=True,
            )

    with col_book:
        st.subheader("Book New Demo Slot")
        leads = database.get_all_leads()
        if not leads:
            st.warning("No leads found in CRM yet. Start a conversation first.")
        else:
            lead_map = {
                f"#{l['id']} — {l['name'] or 'Unnamed'} ({l['company'] or 'N/A'}) [{l['lead_status']}]": l["id"]
                for l in leads
            }
            with st.form("appointment_booking_form"):
                sel_lead_lbl = st.selectbox("Select CRM Lead", list(lead_map.keys()))
                book_date = st.selectbox("Appointment Date", get_available_dates(14))
                book_time = st.selectbox("Time Slot", config.AVAILABLE_TIME_SLOTS)
                book_btn = st.form_submit_button("Confirm Appointment", width="stretch")
                if book_btn:
                    ok, msg = schedule_demo(lead_map[sel_lead_lbl], book_date, book_time)
                    if ok:
                        generate_call_summary(lead_map[sel_lead_lbl])
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

    st.divider()
    st.subheader("Appointment Records")
    appts = database.get_all_appointments()
    if not appts:
        st.caption("No appointments scheduled yet.")
    else:
        appt_df = pd.DataFrame(appts)[
            ["id", "lead_name", "lead_company", "lead_email", "appointment_date", "appointment_time", "status", "lead_status"]
        ]
        st.dataframe(appt_df, width="stretch", hide_index=True)

        u1, u2, u3 = st.columns([2, 1.5, 1])
        with u1:
            appt_map = {
                f"#{a['id']} — {a['lead_name']} ({a['appointment_date']} {a['appointment_time']}) [{a['status']}]": a["id"]
                for a in appts
            }
            sel_appt = st.selectbox("Select Appointment to Update", list(appt_map.keys()))
        with u2:
            new_status = st.selectbox("New Status", ["Confirmed", "Completed", "Cancelled"])
        with u3:
            st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
            if st.button("Update Status", width="stretch"):
                database.update_appointment_status(appt_map[sel_appt], new_status)
                st.success(f"Updated appointment #{appt_map[sel_appt]} to {new_status}.")
                st.rerun()


# ---------------------------------------------------------------------------
# PAGE 5: KNOWLEDGE BASE
# ---------------------------------------------------------------------------
elif page == "Knowledge Base":
    render_hero(
        "RAG Knowledge Base Manager",
        "Upload TXT or PDF company documents, inspect TF-IDF chunks, and test cosine-similarity retrieval.",
    )

    up_col, test_col = st.columns([1, 1.2], gap="large")

    with up_col:
        st.subheader("Upload & Index Document")
        uploaded_file = st.file_uploader("Upload Company Brochure / FAQ (.txt or .pdf)", type=["txt", "pdf", "md"])
        if uploaded_file is not None:
            if st.button("Index into Knowledge Base", width="stretch"):
                text, err = extract_text_from_upload(uploaded_file.name, uploaded_file.getvalue())
                if err:
                    st.error(err)
                else:
                    database.add_document(uploaded_file.name, text)
                    total_chunks = st.session_state.rag_engine.rebuild_index()
                    st.success(f"Indexed `{uploaded_file.name}`! Total chunks in vector index: {total_chunks}.")
                    st.rerun()

        st.divider()
        st.subheader("Indexed Documents")
        docs = database.get_all_documents()
        for doc in docs:
            d_c1, d_c2 = st.columns([3.5, 1])
            with d_c1:
                st.markdown(
                    f"<div class='sp-card' style='padding:10px 14px;'>"
                    f"<b>{html.escape(doc['filename'])}</b><br/>"
                    f"<span class='sp-muted'>{len(doc['content'])} chars · Added {doc['created_at']}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with d_c2:
                if len(docs) > 1 and st.button("Delete", key=f"del_doc_{doc['id']}", width="stretch"):
                    database.delete_document(doc["id"])
                    st.session_state.rag_engine.rebuild_index()
                    st.rerun()

    with test_col:
        st.subheader("Test RAG Retrieval (TF-IDF + Cosine Similarity)")
        rag_query = st.text_input(
            "Enter a test customer query",
            value="What is the price of Growth plan and refund policy?",
        )
        top_k = st.slider("Top-K Chunks", min_value=1, max_value=5, value=3)
        if st.button("Run Similarity Search", width="stretch") or rag_query.strip():
            hits = st.session_state.rag_engine.retrieve(rag_query, top_k=top_k)
            if not hits:
                st.warning("No matching chunks found.")
            else:
                for idx, h in enumerate(hits, start=1):
                    with st.expander(
                        f"Rank #{idx} | {h['filename']} (Chunk {h['chunk_index']}) | Score: {h['score']:.4f}",
                        expanded=(idx == 1),
                    ):
                        st.write(h["text"])

        st.divider()
        st.subheader("Document Chunks Inspector")
        all_chunks = st.session_state.rag_engine.chunks
        if all_chunks:
            doc_names = sorted({c["filename"] for c in all_chunks})
            chosen_doc = st.selectbox("Filter chunks by document", doc_names)
            filtered = [c for c in all_chunks if c["filename"] == chosen_doc]
            st.caption(f"Showing {len(filtered)} chunks for `{chosen_doc}`")
            for ch in filtered:
                with st.expander(f"Chunk #{ch['chunk_index']} ({len(ch['text'])} chars)"):
                    st.write(ch["text"])


# ---------------------------------------------------------------------------
# PAGE 6: CALL ANALYTICS
# ---------------------------------------------------------------------------
elif page == "Call Analytics":
    render_hero(
        "Call & Conversion Analytics",
        "Visual insights into lead quality distribution, customer sentiment, qualification funnel, and demo bookings.",
    )

    leads = database.get_all_leads()
    convs = database.get_all_conversations()
    appts = database.get_all_appointments()
    metrics = database.get_dashboard_metrics()

    leads_df = pd.DataFrame(leads) if leads else pd.DataFrame()
    convs_df = pd.DataFrame(convs) if convs else pd.DataFrame()
    appts_df = pd.DataFrame(appts) if appts else pd.DataFrame()

    m1, m2, m3 = st.columns(3)
    qual_rate = round((metrics["qualified_leads"] / metrics["total_leads"] * 100), 1) if metrics["total_leads"] else 0.0
    booking_rate = round((metrics["booked_appointments"] / metrics["total_leads"] * 100), 1) if metrics["total_leads"] else 0.0
    m1.metric("Qualification Rate (40+)", f"{qual_rate}%")
    m2.metric("Demo Booking Rate", f"{booking_rate}%")
    m3.metric("Total Conversation Turns", metrics["total_conversations"])

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        if not leads_df.empty:
            fig_hist = px.histogram(
                leads_df,
                x="lead_score",
                nbins=10,
                title="Lead Score Distribution (0–100)",
                color_discrete_sequence=["#6366F1"],
            )
            fig_hist.update_layout(bargap=0.15)
            st.plotly_chart(_apply_plotly_theme(fig_hist), width="stretch")
        else:
            st.info("No lead data available yet.")

    with r1c2:
        if not convs_df.empty:
            s_counts = convs_df["sentiment"].value_counts().reset_index()
            s_counts.columns = ["Sentiment", "Count"]
            fig_s = px.pie(
                s_counts,
                names="Sentiment",
                values="Count",
                hole=0.52,
                title="Conversation Sentiment Breakdown",
                color="Sentiment",
                color_discrete_map={"Positive": "#10B981", "Neutral": "#64748B", "Negative": "#F43F5E"},
            )
            st.plotly_chart(_apply_plotly_theme(fig_s), width="stretch")
        else:
            st.info("No conversation sentiment data yet.")

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        if not leads_df.empty:
            leads_df["Qualified_Group"] = leads_df["lead_score"].apply(
                lambda s: "Qualified (Hot/Warm 40+)" if s >= 40 else "Unqualified (Cold <40)"
            )
            q_df = leads_df["Qualified_Group"].value_counts().reset_index()
            q_df.columns = ["Category", "Leads"]
            fig_q = px.bar(
                q_df,
                x="Category",
                y="Leads",
                title="Qualified vs Unqualified Leads",
                color="Category",
                color_discrete_map={
                    "Qualified (Hot/Warm 40+)": "#10B981",
                    "Unqualified (Cold <40)": "#3B82F6",
                },
            )
            fig_q.update_layout(showlegend=False)
            st.plotly_chart(_apply_plotly_theme(fig_q), width="stretch")

    with r2c2:
        if not appts_df.empty:
            a_df = appts_df.groupby(["appointment_date", "status"]).size().reset_index(name="Count")
            fig_a = px.bar(
                a_df,
                x="appointment_date",
                y="Count",
                color="status",
                barmode="group",
                title="Appointment Statistics by Date & Status",
                color_discrete_map={
                    "Confirmed": "#8B5CF6",
                    "Completed": "#10B981",
                    "Cancelled": "#F43F5E",
                },
            )
            st.plotly_chart(_apply_plotly_theme(fig_a), width="stretch")
        else:
            st.info("No appointment statistics yet.")

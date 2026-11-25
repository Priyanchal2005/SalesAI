"""SQLite persistence layer for SalesPilot AI."""

from contextlib import contextmanager
from datetime import datetime, timedelta
import sqlite3
from typing import Any, Optional

import config


@contextmanager
def get_connection():
    """Yield a parameterized-ready SQLite connection with foreign keys enabled."""
    conn = sqlite3.connect(config.DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create database tables if they do not exist and seed initial sample data."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT DEFAULT '',
                email TEXT DEFAULT '',
                phone TEXT DEFAULT '',
                company TEXT DEFAULT '',
                product_interest TEXT DEFAULT '',
                budget TEXT DEFAULT '',
                timeline TEXT DEFAULT '',
                requirements TEXT DEFAULT '',
                lead_score INTEGER DEFAULT 0,
                lead_status TEXT DEFAULT 'Cold',
                score_explanation TEXT DEFAULT '',
                conversation_summary TEXT DEFAULT '',
                recommended_action TEXT DEFAULT '',
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_id INTEGER NOT NULL,
                customer_message TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                sentiment TEXT DEFAULT 'Neutral',
                timestamp TEXT NOT NULL,
                FOREIGN KEY (lead_id) REFERENCES leads(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_id INTEGER NOT NULL,
                appointment_date TEXT NOT NULL,
                appointment_time TEXT NOT NULL,
                status TEXT DEFAULT 'Confirmed',
                created_at TEXT NOT NULL,
                FOREIGN KEY (lead_id) REFERENCES leads(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )

        # Ensure sample knowledge base is indexed once
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if doc_count == 0 and config.SAMPLE_KNOWLEDGE_PATH.exists():
            sample_text = config.SAMPLE_KNOWLEDGE_PATH.read_text(encoding="utf-8")
            conn.execute(
                "INSERT INTO documents (filename, content, created_at) VALUES (?, ?, ?)",
                (
                    config.SAMPLE_KNOWLEDGE_PATH.name,
                    sample_text,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )

        # Seed realistic sample CRM records only when database is fresh
        lead_count = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
        if lead_count == 0:
            _seed_sample_data(conn)


def _seed_sample_data(conn: sqlite3.Connection) -> None:
    """Populate initial demo records so analytics and CRM views are populated on first launch."""
    now = datetime.now()
    tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    day_after = (now + timedelta(days=2)).strftime("%Y-%m-%d")

    sample_leads = [
        {
            "name": "Aarav Mehta",
            "email": "aarav.mehta@finscale.in",
            "phone": "+91-9820112233",
            "company": "FinScale Technologies",
            "product_interest": "Growth Plan",
            "budget": "$150/month",
            "timeline": "Within 14 days",
            "requirements": "Bilingual English and Hindi voice agent for inbound SaaS lead qualification",
            "lead_score": 100,
            "lead_status": "Hot",
            "score_explanation": (
                "Clear product interest (+20) | Budget provided (+20) | "
                "Timeline within 30 days (+25) | Business requirements provided (+20) | "
                "Contact info provided (+15)"
            ),
            "conversation_summary": (
                "Aarav from FinScale Technologies evaluated the Growth Plan ($149/mo) for bilingual "
                "English/Hindi inbound lead qualification and requested an immediate onboarding demo."
            ),
            "recommended_action": "Conduct scheduled live product demo and send Growth Plan 14-day trial link.",
            "created_at": (now - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
            "messages": [
                (
                    "Hi, we get 400+ inbound leads weekly and need an English and Hindi voice agent. What does Growth Plan cost?",
                    "Hello! Our Growth Plan is $149/month (or 20% off on annual billing) and includes up to 3,000 AI conversations, unlimited knowledge base documents, and full English & Hindi voice + text support. May I know your name and company so I can tailor a recommendation?",
                    "Positive",
                ),
                (
                    "I am Aarav Mehta from FinScale Technologies (aarav.mehta@finscale.in, +91-9820112233). Budget is around $150/month and we want to go live within 14 days.",
                    "Great to meet you, Aarav! Growth Plan fits your $150/month budget and 14-day launch goal—teams typically launch in under 15 minutes. Would you like to book a quick demo slot to see the live bilingual workflow?",
                    "Positive",
                ),
            ],
            "appointment": (tomorrow, "11:30 AM", "Confirmed"),
        },
        {
            "name": "Priya Nair",
            "email": "priya@medcareclinic.org",
            "phone": "+91-9845098765",
            "company": "MedCare Clinics",
            "product_interest": "Enterprise Plan",
            "budget": "$500/month",
            "timeline": "Next 45 days",
            "requirements": "24/7 patient enquiry triage and appointment scheduling across 5 clinics",
            "lead_score": 85,
            "lead_status": "Hot",
            "score_explanation": (
                "Clear product interest (+20) | Budget provided (+20) | "
                "Timeline provided >30 days (+10) | Business requirements provided (+20) | "
                "Contact info provided (+15)"
            ),
            "conversation_summary": (
                "Priya Nair from MedCare Clinics is interested in the Enterprise Plan ($499/mo) for "
                "multi-clinic enquiry triage and scheduling within 45 days."
            ),
            "recommended_action": "Prepare custom multi-clinic Enterprise architecture walkthrough.",
            "created_at": (now - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S"),
            "messages": [
                (
                    "Do you have an Enterprise tier for multi-clinic appointment triage? Our budget is around $500/month.",
                    "Yes! Our Enterprise Plan is $499/month and offers unlimited conversations, multi-agent routing, dedicated vector storage, and 24/7 support. What timeline are you targeting for rollout?",
                    "Positive",
                ),
                (
                    "I'm Priya Nair (priya@medcareclinic.org, +91-9845098765) at MedCare Clinics. Looking at the next 45 days.",
                    "Thank you, Priya! A 45-day rollout gives us plenty of time for knowledge base setup and workflow testing. Feel free to pick a demo slot anytime.",
                    "Neutral",
                ),
            ],
            "appointment": (day_after, "02:00 PM", "Confirmed"),
        },
        {
            "name": "Rohan Kapoor",
            "email": "rohan.k@brightedu.co",
            "phone": "",
            "company": "BrightEdu Academy",
            "product_interest": "Starter Plan",
            "budget": "",
            "timeline": "This month",
            "requirements": "Answering student course FAQs automatically",
            "lead_score": 60,
            "lead_status": "Warm",
            "score_explanation": (
                "Clear product interest (+20) | Timeline within 30 days (+25) | "
                "Contact info provided (+15) | Missing: Budget (0), Detailed requirements partial (0)"
            ),
            "conversation_summary": (
                "Rohan from BrightEdu Academy is exploring the Starter Plan ($49/mo) to automate student FAQs this month."
            ),
            "recommended_action": "Follow up via email with Starter Plan breakdown and 14-day free trial details.",
            "created_at": (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "messages": [
                (
                    "We need something simple to answer student course FAQs this month. Email is rohan.k@brightedu.co.",
                    "Our Starter Plan at $49/month includes up to 500 AI conversations and 20 knowledge base documents—ideal for automating course FAQs. What monthly budget range did you have in mind?",
                    "Neutral",
                ),
            ],
            "appointment": None,
        },
        {
            "name": "Vikram Joshi",
            "email": "",
            "phone": "",
            "company": "Unknown",
            "product_interest": "General Inquiry",
            "budget": "",
            "timeline": "",
            "requirements": "",
            "lead_score": 20,
            "lead_status": "Cold",
            "score_explanation": "Clear product interest (+20) | Missing: Budget, Timeline, Requirements, Contact info",
            "conversation_summary": "Visitor asked about refund policy and pricing but left before sharing contact details.",
            "recommended_action": "Keep in nurture bucket; re-engage if visitor returns to chat.",
            "created_at": (now - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "messages": [
                (
                    "Your pricing feels a bit confusing and I'm worried about getting locked into a contract.",
                    "I completely understand your concern! Every new workspace starts with a 14-day free trial (no credit card required), and monthly Starter/Growth plans include a 14-day money-back guarantee with zero cancellation fees. Would you like me to clarify which plan fits your enquiry volume?",
                    "Negative",
                ),
            ],
            "appointment": None,
        },
    ]

    for idx, item in enumerate(sample_leads):
        cur = conn.execute(
            """
            INSERT INTO leads (
                name, email, phone, company, product_interest, budget, timeline,
                requirements, lead_score, lead_status, score_explanation,
                conversation_summary, recommended_action, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item["name"],
                item["email"],
                item["phone"],
                item["company"],
                item["product_interest"],
                item["budget"],
                item["timeline"],
                item["requirements"],
                item["lead_score"],
                item["lead_status"],
                item["score_explanation"],
                item["conversation_summary"],
                item["recommended_action"],
                item["created_at"],
            ),
        )
        lead_id = cur.lastrowid

        for msg_idx, (cust_msg, ai_msg, sentiment) in enumerate(item["messages"]):
            ts = (now - timedelta(hours=5 - idx, minutes=10 - msg_idx * 3)).strftime("%Y-%m-%d %H:%M:%S")
            conn.execute(
                """
                INSERT INTO conversations (lead_id, customer_message, ai_response, sentiment, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (lead_id, cust_msg, ai_msg, sentiment, ts),
            )

        if item["appointment"]:
            appt_date, appt_time, appt_status = item["appointment"]
            conn.execute(
                """
                INSERT INTO appointments (lead_id, appointment_date, appointment_time, status, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (lead_id, appt_date, appt_time, appt_status, item["created_at"]),
            )


def create_lead(**fields: Any) -> int:
    """Insert a new lead record and return its ID."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO leads (
                name, email, phone, company, product_interest, budget, timeline,
                requirements, lead_score, lead_status, score_explanation,
                conversation_summary, recommended_action, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fields.get("name", ""),
                fields.get("email", ""),
                fields.get("phone", ""),
                fields.get("company", ""),
                fields.get("product_interest", ""),
                fields.get("budget", ""),
                fields.get("timeline", ""),
                fields.get("requirements", ""),
                int(fields.get("lead_score", 0)),
                fields.get("lead_status", "Cold"),
                fields.get("score_explanation", ""),
                fields.get("conversation_summary", ""),
                fields.get("recommended_action", ""),
                now_str,
            ),
        )
        return int(cur.lastrowid)


def update_lead(lead_id: int, **fields: Any) -> None:
    """Update allowed columns for an existing lead using parameterized SQL."""
    allowed_cols = {
        "name",
        "email",
        "phone",
        "company",
        "product_interest",
        "budget",
        "timeline",
        "requirements",
        "lead_score",
        "lead_status",
        "score_explanation",
        "conversation_summary",
        "recommended_action",
    }
    updates = {k: v for k, v in fields.items() if k in allowed_cols}
    if not updates:
        return
    assignments = ", ".join(f"{col} = ?" for col in updates)
    values = list(updates.values()) + [lead_id]
    with get_connection() as conn:
        conn.execute(f"UPDATE leads SET {assignments} WHERE id = ?", values)


def get_lead(lead_id: int) -> Optional[dict[str, Any]]:
    """Fetch a single lead by ID."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM leads WHERE id = ?", (lead_id,)).fetchone()
        return dict(row) if row else None


def get_all_leads(search: str = "", status_filter: str = "All") -> list[dict[str, Any]]:
    """Fetch leads filtered by search keyword and status."""
    query = "SELECT * FROM leads WHERE 1=1"
    params: list[Any] = []
    if status_filter and status_filter != "All":
        query += " AND lead_status = ?"
        params.append(status_filter)
    if search.strip():
        like = f"%{search.strip()}%"
        query += (
            " AND (name LIKE ? OR email LIKE ? OR company LIKE ? "
            "OR product_interest LIKE ? OR requirements LIKE ?)"
        )
        params.extend([like, like, like, like, like])
    query += " ORDER BY id DESC"
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def add_conversation(lead_id: int, customer_message: str, ai_response: str, sentiment: str) -> int:
    """Persist a conversation turn linked to a lead."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO conversations (lead_id, customer_message, ai_response, sentiment, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (lead_id, customer_message, ai_response, sentiment, ts),
        )
        return int(cur.lastrowid)


def get_conversations_for_lead(lead_id: int) -> list[dict[str, Any]]:
    """Return all conversation turns for a given lead ordered chronologically."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations WHERE lead_id = ? ORDER BY id ASC",
            (lead_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_recent_conversations(limit: int = 10) -> list[dict[str, Any]]:
    """Return recent conversation turns joined with lead name and status."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT c.*, l.name AS lead_name, l.company AS lead_company, l.lead_status
            FROM conversations c
            LEFT JOIN leads l ON c.lead_id = l.id
            ORDER BY c.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_conversations() -> list[dict[str, Any]]:
    """Return all conversation turns."""
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM conversations ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]


def get_booked_slots(appointment_date: str) -> list[str]:
    """Return all non-cancelled booked times for a date."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT appointment_time FROM appointments
            WHERE appointment_date = ? AND status != 'Cancelled'
            """,
            (appointment_date,),
        ).fetchall()
        return [r["appointment_time"] for r in rows]


def book_appointment(lead_id: int, appointment_date: str, appointment_time: str) -> tuple[bool, str]:
    """Book a demo appointment if the slot is free, preventing duplicate bookings."""
    with get_connection() as conn:
        lead_exists = conn.execute("SELECT id FROM leads WHERE id = ?", (lead_id,)).fetchone()
        if not lead_exists:
            return False, "Selected lead was not found in the CRM."

        conflict = conn.execute(
            """
            SELECT id FROM appointments
            WHERE appointment_date = ? AND appointment_time = ? AND status != 'Cancelled'
            """,
            (appointment_date, appointment_time),
        ).fetchone()
        if conflict:
            return False, f"Slot {appointment_date} at {appointment_time} is already booked."

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            """
            INSERT INTO appointments (lead_id, appointment_date, appointment_time, status, created_at)
            VALUES (?, ?, ?, 'Confirmed', ?)
            """,
            (lead_id, appointment_date, appointment_time, now_str),
        )
        return True, f"Appointment confirmed for {appointment_date} at {appointment_time}."


def get_appointments_for_lead(lead_id: int) -> list[dict[str, Any]]:
    """Return appointments linked to a specific lead."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM appointments WHERE lead_id = ? ORDER BY appointment_date ASC, appointment_time ASC",
            (lead_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_appointments() -> list[dict[str, Any]]:
    """Return all appointments joined with lead contact info."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT a.*, l.name AS lead_name, l.email AS lead_email,
                   l.company AS lead_company, l.lead_status, l.lead_score
            FROM appointments a
            LEFT JOIN leads l ON a.lead_id = l.id
            ORDER BY a.appointment_date ASC, a.appointment_time ASC
            """
        ).fetchall()
        return [dict(r) for r in rows]


def update_appointment_status(appointment_id: int, status: str) -> None:
    """Update status of an appointment (Confirmed, Completed, Cancelled)."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE appointments SET status = ? WHERE id = ?",
            (status, appointment_id),
        )


def add_document(filename: str, content: str) -> int:
    """Store an uploaded knowledge base document in SQLite."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO documents (filename, content, created_at) VALUES (?, ?, ?)",
            (filename, content, now_str),
        )
        return int(cur.lastrowid)


def get_all_documents() -> list[dict[str, Any]]:
    """Return all indexed documents from SQLite."""
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY id ASC").fetchall()
        return [dict(r) for r in rows]


def delete_document(doc_id: int) -> bool:
    """Delete a document from SQLite by ID."""
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        return cur.rowcount > 0


def get_dashboard_metrics() -> dict[str, Any]:
    """Compute aggregate KPI metrics for the Overview and Analytics pages."""
    with get_connection() as conn:
        total_leads = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
        qualified_leads = conn.execute(
            "SELECT COUNT(*) FROM leads WHERE lead_score >= 40"
        ).fetchone()[0]
        hot_leads = conn.execute(
            "SELECT COUNT(*) FROM leads WHERE lead_status = 'Hot'"
        ).fetchone()[0]
        avg_score_row = conn.execute("SELECT AVG(lead_score) FROM leads").fetchone()[0]
        avg_score = round(float(avg_score_row or 0.0), 1)
        booked_appointments = conn.execute(
            "SELECT COUNT(*) FROM appointments WHERE status != 'Cancelled'"
        ).fetchone()[0]
        total_conversations = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]

        sentiment_rows = conn.execute(
            "SELECT sentiment, COUNT(*) AS cnt FROM conversations GROUP BY sentiment"
        ).fetchall()
        sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
        for r in sentiment_rows:
            sentiment_counts[r["sentiment"]] = r["cnt"]

        return {
            "total_leads": total_leads,
            "qualified_leads": qualified_leads,
            "hot_leads": hot_leads,
            "avg_score": avg_score,
            "booked_appointments": booked_appointments,
            "total_conversations": total_conversations,
            "sentiment_counts": sentiment_counts,
        }

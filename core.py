# core.py
"""
Shared config, database models, and business-logic helpers.

Nothing Flask-route-specific lives here - it's what used to sit at the top
of app.py. Kept as its own module so admin_routes.py and user_routes.py can
both import from it without a circular import with app.py (which is now
just the entrypoint that registers them as blueprints).

IMPORTANT: SessionLocal, DocumentChunk, and GROQ_API_KEY are re-exported
from app.py at module level, because ingest_documents.py and
clear_documents.py do 'from app import SessionLocal, DocumentChunk, ...'.
Don't rename these here without updating that re-export in app.py.
"""

from dotenv import load_dotenv
load_dotenv()

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timezone
from groq import Groq

import requests
import numpy as np
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from fastembed import TextEmbedding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --- Config (set these as environment variables) ---
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Optional but recommended: set this to your Meta App Secret to verify that
# incoming webhook POSTs really came from Meta (X-Hub-Signature-256 header).
# If it's not set, signature verification is skipped (fine for local dev,
# NOT recommended for a public Render URL in production).
APP_SECRET = os.getenv("WHATSAPP_APP_SECRET")
if not APP_SECRET:
    logger.warning(
        "⚠️ WHATSAPP_APP_SECRET is not set - webhook signature verification is DISABLED. "
        "Anyone who finds your webhook URL can POST fake messages. Set this env var in production."
    )

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Check your .env file.")
groq_client = Groq(api_key=GROQ_API_KEY)

# Protects all admin endpoints (reading conversation/lead data, wiping chat
# history, etc). Set this in Render's env vars and pass it as the
# "X-Admin-Key" header when calling those endpoints. Without it set, admin
# endpoints are disabled entirely.
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
if not ADMIN_API_KEY:
    logger.warning(
        "⚠️ ADMIN_API_KEY is not set - admin endpoints (reading/deleting chat "
        "history, leads, and document chunks) are DISABLED. Set this env var "
        "if you want to use them."
    )

WHATSAPP_API_URL = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"

# Render Postgres gives postgres:// URLs, SQLAlchemy 1.4+/2.0 wants postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# --- Database setup ---
Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

# --- Embedding model (for RAG over property documents) ---
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# NOTE ON MODEL NAMES: Groq periodically deprecates models. Before relying on
# any model string below, check the current list at
# https://api.groq.com/openai/v1/models (or console.groq.com/docs/models).
# As of this writing, llama-3.3-70b-versatile and llama-3.1-8b-instant are
# deprecated in favor of the openai/gpt-oss-* and qwen3.6 family.
INFO_AGENT_MODEL = "openai/gpt-oss-120b"     # higher-quality conversational replies
ANALYZER_AGENT_MODEL = "openai/gpt-oss-20b" # fast, cheap, structured extraction


# --- Models -----------------------------------------------------------

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_file = Column(String(255), nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    phone_number = Column(String(20), nullable=False)
    direction = Column(String(10), nullable=False)
    message_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ProcessedMessage(Base):
    """Tracks WhatsApp message IDs we've already handled, so retried
    webhook deliveries don't trigger duplicate LLM calls / replies."""
    __tablename__ = "processed_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(String(100), nullable=False, unique=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class LeadProfile(Base):
    """Shared conversation state for a phone number - this is what lets the
    initiator, info, and analyzer agents coordinate instead of acting blind
    of each other."""
    __tablename__ = "lead_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    phone_number = Column(String(20), nullable=False, unique=True)
    conversation_stage = Column(String(20), default="new") # new | qualifying | info_sharing | closing

    budget_range = Column(String(100), nullable=True)
    config_preference = Column(String(50), nullable=True) # e.g. "2BHK", "3BHK"
    location_preference = Column(String(100), nullable=True)
    timeline = Column(String(50), nullable=True)
    financing_status = Column(String(50), nullable=True)
    lead_quality_score = Column(Integer, default=0)
    last_updated = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


Base.metadata.create_all(engine)


# --- Webhook signature verification ---------------------------------

def verify_webhook_signature(req) -> bool:
    """Checks Meta's X-Hub-Signature-256 header against a HMAC-SHA256 of the
    raw request body, using the Meta App Secret. Returns True if verification
    is disabled (no APP_SECRET configured) or if the signature is valid."""
    if not APP_SECRET:
        return True

    signature_header = req.headers.get("X-Hub-Signature-256", "")
    if not signature_header.startswith("sha256="):
        return False

    provided_signature = signature_header.split("sha256=", 1)[1]
    expected_signature = hmac.new(
        APP_SECRET.encode("utf-8"),
        req.get_data(),
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected_signature, provided_signature)


# --- Admin auth -----------------------------------------------------

def is_authorized_admin(req) -> bool:
    """Guards every admin endpoint. Requires ADMIN_API_KEY to be set AND the
    request to send a matching X-Admin-Key header. Returns False (blocking
    the request) if ADMIN_API_KEY isn't configured at all - this is
    intentional, so admin endpoints are off by default."""
    if not ADMIN_API_KEY:
        return False
    provided = req.headers.get("X-Admin-Key", "")
    return hmac.compare_digest(provided, ADMIN_API_KEY)


# --- Message persistence & dedup ------------------------------------

def save_message(phone_number: str, direction: str, message_text: str):
    session = SessionLocal()
    try:
        record = Conversation(
            phone_number=phone_number,
            direction=direction,
            message_text=message_text
        )
        session.add(record)
        session.commit()
        logger.info("Saved %s message for %s", direction, phone_number)
    except Exception as e:
        session.rollback()
        logger.error("Failed to save message: %s", e)
    finally:
        session.close()


def is_duplicate_message(message_id: str) -> bool:
    session = SessionLocal()
    try:
        return session.query(ProcessedMessage).filter_by(message_id=message_id).first() is not None
    finally:
        session.close()


def mark_message_processed(message_id: str):
    session = SessionLocal()
    try:
        session.add(ProcessedMessage(message_id=message_id))
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error("Failed to record processed message id: %s", e)
    finally:
        session.close()


def get_conversation_history(phone_number: str, limit: int = 10):
    """Fetch last n messages from buyer, oldest first."""
    session = SessionLocal()
    try:
        records = (
            session.query(Conversation)
            .filter(Conversation.phone_number == phone_number)
            .order_by(Conversation.created_at.desc())
            .limit(limit)
            .all()
        )
        records.reverse()

        messages = []
        for r in records:
            role = "user" if r.direction == "incoming" else "assistant"
            messages.append({"role": role, "content": r.message_text})
        return messages
    finally:
        session.close()


# --- Lead profile / shared state helpers -----------------------------

def get_or_create_lead(phone_number: str) -> LeadProfile:
    session = SessionLocal()
    try:
        lead = session.query(LeadProfile).filter_by(phone_number=phone_number).first()
        if not lead:
            lead = LeadProfile(phone_number=phone_number, conversation_stage="new")
            session.add(lead)
            session.commit()
            session.refresh(lead)
        return lead
    finally:
        session.close()


def update_lead_stage(phone_number: str, stage: str):
    session = SessionLocal()
    try:
        lead = session.query(LeadProfile).filter_by(phone_number=phone_number).first()
        if lead:
            lead.conversation_stage = stage
            session.commit()
    finally:
        session.close()


# --- Orchestrator ----------------------------------------------------

def route_message(lead: LeadProfile) -> str:
    """Rule-based routing so we never waste an LLM call just to decide who
    should reply. Keep this fast - it runs on every single incoming message."""
    if lead.conversation_stage == "new":
        return "initiator"
    return "info_agent"


# --- Agent 1: Initiator ---------------------------------------------

def initiator_reply(phone_number: str) -> str:
    """Handles the very first message in a conversation. No LLM call needed -
    a templated greeting is faster, cheaper, and consistent."""
    update_lead_stage(phone_number, "qualifying")
    return (
        "👋 Hi! Thanks for reaching out. I can help you find the right property. "
        "To start - are you looking to buy for your own use or as an investment, "
        "and do you have a location or budget in mind?"
    )


# --- Agent 2: Info agent (RAG over property documents) --------------

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_relevant_chunks(query: str, top_k: int = 4) -> list[str]:
    """Find the most relevant document chunks for a buyer's question."""
    query_embedding = list(embedder.embed([query]))[0].tolist()

    session = SessionLocal()
    try:
        all_chunks = session.query(DocumentChunk).all()
        scored = []
        for chunk in all_chunks:
            chunk_embedding = json.loads(chunk.embedding)
            score = cosine_similarity(query_embedding, chunk_embedding)
            scored.append((score, chunk.chunk_text, chunk.source_file))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_chunks = scored[:top_k]

        return [f"[From {source}]: {text}" for _, text, source in top_chunks]
    finally:
        session.close()


INFO_AGENT_SYSTEM_PROMPT_TEMPLATE = """You are a helpful real estate assistant for our WhatsApp business account.
Answer buyer questions using ONLY the property information provided below.
If the answer isn't in the provided context, politely say you don't have that detail and offer to connect them with an agent.
Keep replies concise (2-4 sentences) and conversational - this is WhatsApp, not email.
Where it helps clarity, use short bullet points.

RELEVANT PROPERTY INFORMATION:
{context}
"""


def generate_llm_reply(sender: str, new_message: str) -> str:
    """Build conversation context and get a reply from Groq, grounded in
    retrieved property documents (RAG)."""
    history = get_conversation_history(sender)

    relevant_chunks = retrieve_relevant_chunks(new_message)
    context = "\n\n".join(relevant_chunks) if relevant_chunks else "No specific property data found."

    system_prompt = INFO_AGENT_SYSTEM_PROMPT_TEMPLATE.format(context=context)

    messages = [{"role": "system", "content": system_prompt}] + history + [
        {"role": "user", "content": new_message}
    ]

    try:
        response = groq_client.chat.completions.create(
            model=INFO_AGENT_MODEL,
            max_tokens=300,
            messages=messages
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error("Groq API call failed (info agent): %s", e)
        return "Sorry, I'm having trouble responding right now. Our team will get back to you shortly!"


# --- Agent 3: Analyzer (silent metadata extraction, runs in the background) ---

ANALYZER_PROMPT_TEMPLATE = """Extract any real estate lead-qualification details the customer just revealed.
You must always respond with valid JSON matching this exact schema - never leave the response empty.

Schema:
{{"budget_range": null or string, "config_preference": null or string,
"location_preference": null or string, "timeline": null or string,
"financing_status": null or string, "sentiment": "positive" or "neutral" or "negative"}}

Only fill fields that are explicitly mentioned or clearly implied by the message. Leave everything else null.
Do not guess. If the message contains no useful lead information, return all fields as null except sentiment.

Example - message: "just checking my number is still same"
Response: {{"budget_range": null, "config_preference": null, "location_preference": null, "timeline": null, "financing_status": null, "sentiment": "neutral"}}

Customer message: {message}
"""


def analyze_and_update_lead(phone_number: str, message: str):
    try:
        response = groq_client.chat.completions.create(
            model=ANALYZER_AGENT_MODEL,
            max_tokens=200,
            temperature=0,  # more deterministic JSON output
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": ANALYZER_PROMPT_TEMPLATE.format(message=message)}]
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            logger.warning("Analyzer returned empty content for message: %r", message)
            return
        extracted = json.loads(content)
    except Exception as e:
        logger.error("Analyzer agent failed: %s", e)
        return

    session = SessionLocal()
    try:
        lead = session.query(LeadProfile).filter_by(phone_number=phone_number).first()
        if not lead:
            return

        trackable_fields = [
            "budget_range", "config_preference", "location_preference",
            "timeline", "financing_status"
        ]
        newly_filled = 0
        for field in trackable_fields:
            value = extracted.get(field)
            if value and not getattr(lead, field):
                setattr(lead, field, value)
                newly_filled += 1

        if newly_filled:
            lead.lead_quality_score = (lead.lead_quality_score or 0) + (newly_filled * 10)

        session.commit()
        logger.info("Analyzer updated lead %s: %s", phone_number, extracted)
    except Exception as e:
        session.rollback()
        logger.error("Failed to save analyzer output for %s: %s", phone_number, e)
    finally:
        session.close()


# --- Send Reply ------------------------------------------------------

def send_reply(to: str, message: str):
    """Send a text message back to the buyer via WhatsApp Cloud API."""
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }
    try:
        response = requests.post(WHATSAPP_API_URL, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info("📲 Reply sent to %s", to)
        else:
            logger.error("❌ Failed to send reply: %s", response.text)
    except requests.RequestException as e:
        logger.error("❌ Network error sending reply to %s: %s", to, e)


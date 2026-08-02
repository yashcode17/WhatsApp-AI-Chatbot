# app.py

from dotenv import load_dotenv
load_dotenv()

import hashlib
import hmac
import json
import logging
import os
import threading
from datetime import datetime, timezone
from groq import Groq

import requests
from flask import Flask, jsonify, request
import numpy as np
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from fastembed import TextEmbedding

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --- Config (set these as environment variables) ---
VERIFY_TOKEN     = os.getenv("VERIFY_TOKEN")
ACCESS_TOKEN     = os.getenv("WHATSAPP_ACCESS_TOKEN")
PHONE_NUMBER_ID  = os.getenv("PHONE_NUMBER_ID")
DATABASE_URL     = os.getenv("DATABASE_URL")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")

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



# Protects destructive admin endpoints (e.g. wiping chat history). Set this
# in Render's env vars and pass it as the "X-Admin-Key" header when calling
# those endpoints. Without it set, admin endpoints are disabled entirely.
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
if not ADMIN_API_KEY:
    logger.warning(
        "⚠️ ADMIN_API_KEY is not set - admin endpoints (deleting chat history) are DISABLED. "
        "Set this env var if you want to use them."
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
INFO_AGENT_MODEL     = "openai/gpt-oss-120b"  # higher-quality conversational replies
ANALYZER_AGENT_MODEL = "openai/gpt-oss-20b"   # fast, cheap, structured extraction


# --- Models -------------------------------------------------------------------

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    source_file = Column(String(255), nullable=False)
    chunk_text  = Column(Text, nullable=False)
    embedding   = Column(Text, nullable=False)


class Conversation(Base):
    __tablename__ = "conversations"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    phone_number = Column(String(20), nullable=False)
    direction    = Column(String(10), nullable=False)
    message_text = Column(Text, nullable=False)
    created_at   = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ProcessedMessage(Base):
    """Tracks WhatsApp message IDs we've already handled, so retried
    webhook deliveries don't trigger duplicate LLM calls / replies."""
    __tablename__ = "processed_messages"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(String(100), nullable=False, unique=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class LeadProfile(Base):
    """Shared conversation state for a phone number - this is what lets the
    initiator, info, and analyzer agents coordinate instead of acting blind
    of each other."""
    __tablename__ = "lead_profiles"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    phone_number        = Column(String(20), nullable=False, unique=True)
    conversation_stage  = Column(String(20), default="new") # new | qualifying | info_sharing | closing
    budget_range        = Column(String(100), nullable=True)
    config_preference   = Column(String(50), nullable=True)  # e.g. "2BHK", "3BHK"
    location_preference = Column(String(100), nullable=True)
    timeline            = Column(String(50), nullable=True)
    financing_status    = Column(String(50), nullable=True)
    lead_quality_score  = Column(Integer, default=0)
    last_updated        = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


Base.metadata.create_all(engine)


# --- Webhook signature verification ------------------------------------------

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


# --- Message persistence & dedup ----------------------------------------------

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

def is_authorized_admin(req) -> bool:
    """Guards destructive admin endpoints. Requires ADMIN_API_KEY to be set
    AND the request to send a matching X-Admin-Key header. Returns False
    (blocking the request) if ADMIN_API_KEY isn't configured at all - this
    is intentional, so these endpoints are off by default."""
    if not ADMIN_API_KEY:
        return False
    provided = req.headers.get("X-Admin-Key", "")
    return hmac.compare_digest(provided, ADMIN_API_KEY)

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


# --- Lead profile / shared state helpers -------------------------------------

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


# --- Orchestrator -------------------------------------------------------------

def route_message(lead: LeadProfile) -> str:
    """Rule-based routing so we never waste an LLM call just to decide who
    should reply. Keep this fast - it runs on every single incoming message."""
    if lead.conversation_stage == "new":
        return "initiator"
    return "info_agent"


# --- Agent 1: Initiator -------------------------------------------------------

def initiator_reply(phone_number: str) -> str:
    """Handles the very first message in a conversation. No LLM call needed -
    a templated greeting is faster, cheaper, and consistent."""
    update_lead_stage(phone_number, "qualifying")
    return (
        "👋 Hi! Thanks for reaching out. I can help you find the right property. "
        "To start - are you looking to buy for your own use or as an investment, "
        "and do you have a location or budget in mind?"
    )


# --- Agent 2: Info agent (RAG over property documents) -----------------------

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_relevant_chunks(query: str, top_k: int = 4) -> list[str]:
    """Find the most relevant document chunks for a buyer's question."""
    query_embedding = list(embedder.embed([query])[0].tolist())

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
Return ONLY valid JSON, no prose, matching this schema exactly:
{{"budget_range": null or string, "config_preference": null or string,
"location_preference": null or string, "timeline": null or string,
"financing_status": null or string, "sentiment": "positive" or "neutral" or "negative"}}
Only fill fields that are explicitly mentioned or clearly implied by the message below.
Leave everything else null. Do not guess.

Customer message: {message}
"""


def analyze_and_update_lead(phone_number: str, message: str):
    """Runs on every customer turn, in a background thread, so it never
    delays the WhatsApp reply. Never talks to the customer directly -
    it only updates the shared lead profile."""
    try:
        response = groq_client.chat.completions.create(
            model=ANALYZER_AGENT_MODEL,
            max_tokens=200,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": ANALYZER_PROMPT_TEMPLATE.format(message=message)}]
        )
        extracted = json.loads(response.choices[0].message.content)
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


# --- Send Reply ---------------------------------------------------------------

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


# --- HEALTH CHECK -------------------------------------------------------------

@app.route("/test", methods=["GET"])
def test_route():
    logger.info("/test endpoint - server is alive")
    return jsonify({
        "status": "ok",
        "message": "Server is running"
    }), 200


# --- Webhook Verification (GET) ----------------------------------------------

@app.route("/webhook", methods=["GET"])
def verify_webhook():
    """Meta calls this once to verify your webhook URL."""
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        logger.info("✅ Webhook verified successfully")
        return challenge, 200
    else:
        logger.warning("❌ Webhook verification failed")
        return "Forbidden", 403


# --- Incoming Messages (POST) ------------------------------------------------

@app.route("/webhook", methods=["POST"])
def receive_message():
    """Meta sends all incoming messages here."""
    if not verify_webhook_signature(request):
        logger.warning("❌ Webhook signature verification failed - rejecting request")
        return "Forbidden", 403

    data = request.get_json()
    logger.info("📩 Incoming payload:\n%s", json.dumps(data, indent=2))

    try:
        entry = data["entry"][0]
        changes = entry["changes"][0]["value"]

        # Ignore status updates (delivered, read, etc.)
        if "messages" not in changes:
            return jsonify({"status": "ignored"}), 200

        message = changes["messages"][0]
        sender = message["from"]  # buyer's phone number
        msg_type = message["type"]
        msg_id = message["id"]

        # Meta may redeliver the same webhook event - skip if we've seen this message id.
        if is_duplicate_message(msg_id):
            logger.info("⏭️ Skipping already-processed message id: %s", msg_id)
            return jsonify({"status": "duplicate_ignored"}), 200
        mark_message_processed(msg_id)

        if msg_type == "text":
            text = message["text"]["body"]
            logger.info("👤 Buyer [%s] said: %s", sender, text)

            save_message(sender, "incoming", text)

            # Orchestration: decide which agent replies
            lead = get_or_create_lead(sender)
            agent = route_message(lead)

            if agent == "initiator":
                reply_text = initiator_reply(sender)
            else:
                reply_text = generate_llm_reply(sender, text)

            # Analyzer runs in the background - never blocks the reply.
            threading.Thread(
                target=analyze_and_update_lead,
                args=(sender, text),
                daemon=True
            ).start()

            send_reply(sender, reply_text)
            save_message(sender, "outgoing", reply_text)

        else:
            logger.info("ℹ️ Received non-text message of type: %s", msg_type)

    except (KeyError, IndexError) as e:
        logger.error("⚠️ Failed to parse payload: %s", e)

    return jsonify({"status": "ok"}), 200


# --- Debug / inspection routes ------------------------------------------------

@app.route("/conversations/<phone_number>", methods=["GET"])
def get_conversations(phone_number):
    session = SessionLocal()
    try:
        records = (
            session.query(Conversation)
            .filter(Conversation.phone_number == phone_number)
            .order_by(Conversation.created_at.asc())
            .all()
        )
        result = [
            {
                "direction": r.direction,
                "message": r.message_text,
                "timestamp": r.created_at.isoformat()
            }
            for r in records
        ]
        return jsonify(result), 200
    finally:
        session.close()

@app.route("/conversations", methods=["DELETE"])
def delete_all_conversations():
    """Deletes ALL chat history for ALL phone numbers. Irreversible.
    Requires header: X-Admin-Key: <ADMIN_API_KEY>
    """
    if not is_authorized_admin(request):
        logger.warning("❌ Unauthorized attempt to delete all conversations")
        return jsonify({"error": "unauthorized"}), 401

    session = SessionLocal()
    try:
        deleted = session.query(Conversation).delete()
        session.commit()
        logger.warning("🗑️ Deleted ALL conversations (%d rows)", deleted)
        return jsonify({"status": "ok", "deleted_rows": deleted}), 200
    except Exception as e:
        session.rollback()
        logger.error("Failed to delete conversations: %s", e)
        return jsonify({"error": "failed to delete conversations"}), 500
    finally:
        session.close()

@app.route("/conversations/<phone_number>", methods=["DELETE"])
def delete_conversation_for_number(phone_number):
    """Deletes chat history for a single phone number only. Irreversible.
    Requires header: X-Admin-Key: <ADMIN_API_KEY>
    """
    if not is_authorized_admin(request):
        logger.warning("❌ Unauthorized attempt to delete conversation for %s", phone_number)
        return jsonify({"error": "unauthorized"}), 401

    session = SessionLocal()
    try:
        deleted = (
            session.query(Conversation)
            .filter(Conversation.phone_number == phone_number)
            .delete()
        )
        session.commit()
        logger.warning("🗑️ Deleted %d conversation rows for %s", deleted, phone_number)
        return jsonify({"status": "ok", "phone_number": phone_number, "deleted_rows": deleted}), 200
    except Exception as e:
        session.rollback()
        logger.error("Failed to delete conversation for %s: %s", phone_number, e)
        return jsonify({"error": "failed to delete conversation"}), 500
    finally:
        session.close()

@app.route("/leads/<phone_number>", methods=["GET"])
def get_lead(phone_number):
    session = SessionLocal()
    try:
        lead = session.query(LeadProfile).filter_by(phone_number=phone_number).first()
        if not lead:
            return jsonify({"error": "not found"}), 404
        return jsonify({
            "phone_number": lead.phone_number,
            "conversation_stage": lead.conversation_stage,
            "budget_range": lead.budget_range,
            "config_preference": lead.config_preference,
            "location_preference": lead.location_preference,
            "timeline": lead.timeline,
            "financing_status": lead.financing_status,
            "lead_quality_score": lead.lead_quality_score,
            "last_updated": lead.last_updated.isoformat() if lead.last_updated else None
        }), 200
    finally:
        session.close()


@app.route("/debug/documents", methods=["GET"])
def view_documents():
    session = SessionLocal()
    try:
        chunks = session.query(DocumentChunk).all()
        result = [
            {
                "id": c.id,
                "source_file": c.source_file,
                "chunk_preview": c.chunk_text[:150],
                "embedding_length": len(json.loads(c.embedding))
            }
            for c in chunks
        ]
        return jsonify({"total_chunks": len(result), "chunks": result}), 200
    finally:
        session.close()


# --- Run ----------------------------------------------------------------------

if __name__ == "__main__":
    app.run(port=5000, debug=True)
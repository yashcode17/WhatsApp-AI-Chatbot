# app.py

import json
import logging
import os
import sys
from datetime import datetime, timezone

import requests
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --- Config (set these as environment variables or replace directly) ---
VERIFY_TOKEN   = os.getenv("VERIFY_TOKEN")
ACCESS_TOKEN   = os.getenv("WHATSAPP_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
DATABASE_URL = os.getenv("DATABASE_URL")

WHATSAPP_API_URL = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"

#Render Postgerss URL
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql1://", 1)

#Database setup
Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

class Conversation(Base):
    __tablename__ = "conversations"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    phone_number = Column(String(20), nullable=False)
    direction    = Column(String(10), nullable=False)
    message_text = Column(Text, nullable=False)
    created_at   = Column(DateTime, default=lambda: datetime.now(timezone.utc))

#Create table if not exist
Base.metadata.create_all(engine)

def save_message(phone_number: str, direction: str, message_txt: str):
    session = SessionLocal()
    try:
        record = Conversation(
            phone_number=phone_number,
            direction=direction,
            message_txt=message_txt
        )
        session.add(record)
        session.commit()
        logger.info("Saved %s message for %s", direction, phone_number)
    except Exception as e:
        session.rollback()
        logger.error("Failse to save message: %s", e)
    finally:
        session.close()

# ── HEALT CHECK  ─────────────────────────────────────────────────
@app.route("/test", methods=["GET"])
def test_route():
    logger.info("/test endpoint - server is alive")
    return jsonify({
        "status": "ok",
        "message": "Server is running"
    }), 200


# ── Webhook Verification (GET) ─────────────────────────────────────────────────
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    """Meta calls this once to verify your webhook URL."""
    mode      = request.args.get("hub.mode")
    token     = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        logger.info("✅ Webhook verified successfully")
        return challenge, 200
    else:
        logger.warning("❌ Webhook verification failed")
        return "Forbidden", 403


# ── Incoming Messages (POST) ───────────────────────────────────────────────────
@app.route("/webhook", methods=["POST"])
def receive_message():
    """Meta sends all incoming messages here."""
    data = request.get_json()
    logger.info("📩 Incoming payload:\n%s", json.dumps(data, indent=2))

    try:
        entry   = data["entry"][0]
        changes = entry["changes"][0]["value"]

        # Ignore status updates (delivered, read, etc.)
        if "messages" not in changes:
            return jsonify({"status": "ignored"}), 200

        message     = changes["messages"][0]
        sender      = message["from"]          # buyer's phone number
        msg_type    = message["type"]
        msg_id      = message["id"]

        if msg_type == "text":
            text = message["text"]["body"]
            logger.info("👤 Buyer [%s] said: %s", sender, text)

            #save incoming message:
            save_message(sender, "incoming", text)

            # ── Your business logic goes here ──────────────────────────
            reply_text = handle_message(sender, text)
            # ──────────────────────────────────────────────────────────

            send_reply(sender, reply_text)
            save_message(sender, "outgoing", reply_text)

        else:
            logger.info("📎 Received non-text message of type: %s", msg_type)

    except (KeyError, IndexError) as e:
        logger.error("⚠️  Failed to parse payload: %s", e)

    return jsonify({"status": "ok"}), 200


# ── Business Logic ─────────────────────────────────────────────────────────────
def handle_message(sender: str, text: str) -> str:
    """
    Put your chatbot logic here.
    For now it echoes back with a greeting.
    """
    text_lower = text.lower().strip()

    if any(word in text_lower for word in ["hello", "hi", "hey"]):
        return f"👋 Hello! Welcome. How can I help you today?"
    elif "price" in text_lower or "cost" in text_lower:
        return "💰 Please share the product you're interested in and I'll get you the price."
    elif "bye" in text_lower:
        return "👋 Goodbye! Feel free to message us anytime."
    else:
        return f"Thanks for your message: \"{text}\". Our team will get back to you shortly!"


# ── Send Reply ─────────────────────────────────────────────────────────────────
def send_reply(to: str, message: str):
    """Send a text message back to the buyer via WhatsApp Cloud API."""
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type":  "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to":   to,
        "type": "text",
        "text": {"body": message}
    }
    response = requests.post(WHATSAPP_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        logger.info("✅ Reply sent to %s", to)
    else:
        logger.error("❌ Failed to send reply: %s", response.text)

#view conversation for debugging
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


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(port=5000, debug=True)


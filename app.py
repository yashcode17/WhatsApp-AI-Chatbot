# app.py

from dotenv import load_dotenv
load_dotenv()

import json
import logging
import os
import sys
from datetime import datetime, timezone
from groq import Groq

import requests
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import numpy as np
import json

from sentence_transformers import SentenceTransformer

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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Check your .env file.")
groq_client = Groq(api_key=GROQ_API_KEY)

WHATSAPP_API_URL = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"

#Render Postgerss URL
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

#Database setup
Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

#Setup Embedding
embedder = SentenceTransformer("all-MiniLM-L6-v2")

SYSTEM_PROMPT = """You are a hgelpful sales assistent for our WhatsApp business account. 
                You help buyers with product questions, pricing and general inqueries.
                Keep replies concise (2-4 sentences), friendly, and conversational - this is WhatsApp, not email.
                If you don't know specific product details, politely say you'll connect them with the team."""

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    source_file = Column(String(255), nullable=False)
    chunk_text  = Column(Text, nullable=False)
    embedding   = Column(Text, nullable=False)  

# Base.metadata.create_all(engine)

class Conversation(Base):
    __tablename__ = "conversations"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    phone_number = Column(String(20), nullable=False)
    direction    = Column(String(10), nullable=False)
    message_text = Column(Text, nullable=False)
    created_at   = Column(DateTime, default=lambda: datetime.now(timezone.utc))

#Create table if not exist
Base.metadata.create_all(engine)

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
            reply_text = generate_llm_reply(sender, text)
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

def get_conversation_history(phone_number: str, limit: int = 10):
    "fetch last n messages from buyer, oldest first"
    session = SessionLocal()
    try:
        records=(
            session.query(Conversation)
            .filter(Conversation.phone_number == phone_number)
            .order_by(Conversation.created_at.desc())
            .limit(limit)
            .all()
        )
        records.reverse()

        message = []
        for r in records:
            role = "user" if r.direction == "incoming" else "assistant"
            message.append({"role": role, "content": r.message_text})
        return message
    finally:
        session.close()

# def generate_llm_reply(sender: str, new_message: str) -> str:
#     """Build conversation context and get a rely from groq"""
#     history = get_conversation_history(sender)

#     "Add newwst incoming message in conversation"
#     message = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
#     {"role": "user", "content": new_message}]

#     try:
#         response = groq_client.chat.completions.create(
#             model="llama-3.3-70b-versatile",
#             max_tokens=300,
#             messages=message
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         logger.error("❌ Groq API call failed: %s", e)
#         return "Sorry, I'm having trouble responding right now. Our team will get back to you shortly!"

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_relevant_chunks(query: str, top_k: int = 4) -> list[str]:
    "Find most relevent documnet chunk for a buyer's question."
    query_embedding = embedder.encode(query).tolist()

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

SYSTEM_PROMPT_TEMPLATE = """You are a helpful real estate assistant for our WhataApp business account.
Answer buyer questions usinf ONLY the propery information provided below.
If the answer isn't in the provided context, politely say you don't have detail and offer to connect with an agent.
Keep the replies concise (2-4 sentence) and conversational - this is WhatsApp, not email.

RELEVANT PROPERT INFORMATION:
{context}
"""

def generate_llm_reply(sender: str, new_message: str) -> str:
    """Build conversation context and get a rely from groq"""
    history = get_conversation_history(sender)

    relevant_chunks = retrieve_relevant_chunks(new_message)
    context = "\n\n".join(relevant_chunks) if relevant_chunks else "no specific propert data found."

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    messages = [{"role": "system", "context": system_prompt}] + history + [
        {"role": "user", "context": new_message}
    ]

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=300,
            messages=messages
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error("Groq API call failed: %s", e)
        return "Sorry, I'm having trouble responding right now. Our team will get back to you shortly1"


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(port=5000, debug=True)


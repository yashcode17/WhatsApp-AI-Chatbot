# app.py
import hmac
import hashlib
import json
import logging
import os
import requests
from flask import Flask, request, jsonify

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

WHATSAPP_API_URL = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"

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

    print("TOKEN IS: " + token)

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

            # ── Your business logic goes here ──────────────────────────
            reply_text = handle_message(sender, text)
            # ──────────────────────────────────────────────────────────

            send_reply(sender, reply_text)

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


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(port=5000, debug=True)


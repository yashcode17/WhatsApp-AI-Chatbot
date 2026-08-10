# user_routes.py
"""
User-facing endpoints - the ones real buyers (via Meta/WhatsApp) or an
uptime monitor actually hit. Nothing here requires the admin key, and
nothing here returns raw conversation/lead data.
"""

import json
import threading

from flask import Blueprint, jsonify, request

from core import (
    logger,
    VERIFY_TOKEN,
    verify_webhook_signature,
    is_duplicate_message,
    mark_message_processed,
    save_message,
    get_or_create_lead,
    route_message,
    initiator_reply,
    generate_llm_reply,
    analyze_and_update_lead,
    send_reply,
)

user_bp = Blueprint("user", __name__)


# --- HEALTH CHECK ---------------------------------------------------------

@user_bp.route("/test", methods=["GET"])
def test_route():
    """
    Health check
    ---
    tags:
      - User
    responses:
      200:
        description: Server is running
        schema:
          type: object
          properties:
            status:
              type: string
              example: ok
            message:
              type: string
              example: Server is running
    """
    logger.info("/test endpoint - server is alive")
    return jsonify({
        "status": "ok",
        "message": "Server is running"
    }), 200


# --- Webhook Verification (GET) -------------------------------------------

@user_bp.route("/webhook", methods=["GET"])
def verify_webhook():
    """
    WhatsApp webhook verification
    ---
    tags:
      - User
    description: >
      Meta calls this once (and whenever you update the webhook config in the
      Meta App dashboard) to verify you control this URL. Not something a
      person calls manually.
    parameters:
      - name: hub.mode
        in: query
        type: string
        example: subscribe
      - name: hub.verify_token
        in: query
        type: string
        description: Must match the VERIFY_TOKEN env var.
      - name: hub.challenge
        in: query
        type: string
    responses:
      200:
        description: Verified - echoes back hub.challenge.
      403:
        description: Token mismatch or missing.
    """
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        logger.info("✅ Webhook verified successfully")
        return challenge, 200
    else:
        logger.warning("❌ Webhook verification failed")
        return "Forbidden", 403


# --- Incoming Messages (POST) ---------------------------------------------

@user_bp.route("/webhook", methods=["POST"])
def receive_message():
    """
    Receive an incoming WhatsApp message
    ---
    tags:
      - User
    description: >
      Meta POSTs every incoming message/status event here. Verifies the
      X-Hub-Signature-256 header (if WHATSAPP_APP_SECRET is set), dedupes by
      message id, routes to the initiator or info agent, and replies via the
      WhatsApp Cloud API. Not something a person calls manually - shape of
      the body is Meta's WhatsApp webhook payload format.
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          description: Meta WhatsApp webhook payload. See Meta's Cloud API docs for the full shape.
    responses:
      200:
        description: Always 200 once parsed, even for ignored/duplicate events.
        schema:
          type: object
          properties:
            status:
              type: string
              example: ok
      403:
        description: Signature verification failed.
    """
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
            logger.info("⏩ Skipping already-processed message id: %s", msg_id)
            return jsonify({"status": "duplicate_ignored"}), 200
        mark_message_processed(msg_id)

        if msg_type == "text":
            text = message["text"]["body"]
            logger.info("💬 Buyer [%s] said: %s", sender, text)

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


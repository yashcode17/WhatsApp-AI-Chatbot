# admin_routes.py
"""
Admin/debug endpoints - reading and deleting conversation history, reading
lead profiles, and inspecting ingested document chunks.

Every route here is namespaced under /admin and requires a valid
X-Admin-Key header (checked via core.is_authorized_admin), including the
read-only ones. That's a deliberate change from the original single-file
version, where the GET routes for conversations/leads/documents had no
auth at all and only the DELETE routes were protected - conversation
content and lead data are sensitive to expose even without a destructive
verb attached, so every /admin/* route is locked the same way.

If ADMIN_API_KEY isn't set in the environment, is_authorized_admin() always
returns False, so these endpoints are effectively disabled by default.
"""

import json
from flask import Blueprint, jsonify, request

from core import (
    SessionLocal,
    Conversation,
    LeadProfile,
    DocumentChunk,
    is_authorized_admin,
    logger,
)


admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


def _require_admin():
    """Returns a (response, status) tuple to return early on if unauthorized,
    or None if the request is authorized to proceed."""
    if not is_authorized_admin(request):
        logger.warning("❌ Unauthorized admin request to %s", request.path)
        return jsonify({"error": "unauthorized"}), 401
    return None


# --- Conversations --------------------------------------------------------

@admin_bp.route("/conversations/<phone_number>", methods=["GET"])
def get_conversations(phone_number):
    """
    Get full chat history for a phone number
    ---
    tags:
      - Admin
    security:
      - AdminKey: []
    parameters:
      - name: phone_number
        in: path
        type: string
        required: true
        example: "919876543210"
    responses:
      200:
        description: Messages oldest-first
        schema:
          type: array
          items:
            type: object
            properties:
              direction:
                type: string
                enum: [incoming, outgoing]
              message:
                type: string
              timestamp:
                type: string
                format: date-time
      401:
        description: Missing or invalid X-Admin-Key.
    """
    unauthorized = _require_admin()
    if unauthorized:
        return unauthorized

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


@admin_bp.route("/conversations", methods=["DELETE"])
def delete_all_conversations():
    """
    Delete ALL chat history for ALL phone numbers
    ---
    tags:
      - Admin
    security:
      - AdminKey: []
    description: Irreversible. Wipes the entire conversations table.
    responses:
      200:
        description: Deleted
        schema:
          type: object
          properties:
            status:
              type: string
              example: ok
            deleted_rows:
              type: integer
      401:
        description: Missing or invalid X-Admin-Key.
      500:
        description: Delete failed.
    """
    unauthorized = _require_admin()
    if unauthorized:
        return unauthorized

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


@admin_bp.route("/conversations/<phone_number>", methods=["DELETE"])
def delete_conversation_for_number(phone_number):
    """
    Delete chat history for a single phone number
    ---
    tags:
      - Admin
    security:
      - AdminKey: []
    description: Irreversible.
    parameters:
      - name: phone_number
        in: path
        type: string
        required: true
        example: "919876543210"
    responses:
      200:
        description: Deleted
        schema:
          type: object
          properties:
            status:
              type: string
              example: ok
            phone_number:
              type: string
            deleted_rows:
              type: integer
      401:
        description: Missing or invalid X-Admin-Key.
      500:
        description: Delete failed.
    """
    unauthorized = _require_admin()
    if unauthorized:
        return unauthorized

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


# --- Leads ----------------------------------------------------------------

@admin_bp.route("/leads/<phone_number>", methods=["GET"])
def get_lead(phone_number):
    """
    Get the extracted lead profile for a phone number
    ---
    tags:
      - Admin
    security:
      - AdminKey: []
    parameters:
      - name: phone_number
        in: path
        type: string
        required: true
        example: "919876543210"
    responses:
      200:
        description: Lead profile
        schema:
          type: object
          properties:
            phone_number:
              type: string
            conversation_stage:
              type: string
              enum: [new, qualifying, info_sharing, closing]
            budget_range:
              type: string
            config_preference:
              type: string
            location_preference:
              type: string
            timeline:
              type: string
            financing_status:
              type: string
            lead_quality_score:
              type: integer
            last_updated:
              type: string
              format: date-time
      401:
        description: Missing or invalid X-Admin-Key.
      404:
        description: No lead found for this phone number.
    """
    unauthorized = _require_admin()
    if unauthorized:
        return unauthorized

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


# --- Documents (RAG chunk inspection) ------------------------------------

@admin_bp.route("/documents", methods=["GET"])
def view_documents():
    """
    Inspect ingested RAG document chunks
    ---
    tags:
      - Admin
    security:
      - AdminKey: []
    description: >
      Debug view over the document_chunks table populated by
      ingest_documents.py. Shows a preview of each chunk, not the full text.
    responses:
      200:
        description: Chunk inventory
        schema:
          type: object
          properties:
            total_chunks:
              type: integer
            chunks:
              type: array
              items:
                type: object
                properties:
                  id:
                    type: integer
                  source_file:
                    type: string
                  chunk_preview:
                    type: string
                  embedding_length:
                    type: integer
      401:
        description: Missing or invalid X-Admin-Key.
    """
    unauthorized = _require_admin()
    if unauthorized:
        return unauthorized

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


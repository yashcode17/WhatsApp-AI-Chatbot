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

from flask import Blueprint, jsonify, request

from core import (
    SessionLocal,
    Conversation,
    LeadProfile,
    DocumentChunk,
    is_authorized_admin,
    logger,
)
import json

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


def _require_admin():
    """Returns a (response, status) tuple to return early on if unauthorized,
    or None if the request is authorized to proceed."""
    if not is_authorized_admin(request):
        logger.warning("❌ Unauthorized admin request to %s", request.path)
        return jsonify({"error": "unauthorized"}), 401
    return None


# --- Conversations ----------------------------------------------------

@admin_bp.route("/conversations/<phone_number>", methods=["GET"])
def get_conversations(phone_number):
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
    """Deletes ALL chat history for ALL phone numbers. Irreversible.
    Requires header: X-Admin-Key: <ADMIN_API_KEY>
    """
    unauthorized = _require_admin()
    if unauthorized:
        return unauthorized

    session = SessionLocal()
    try:
        deleted = session.query(Conversation).delete()
        session.commit()
        logger.warning("🗑️ Deleted %d conversation rows", deleted)
        return jsonify({"status": "ok", "deleted_rows": deleted}), 200
    except Exception as e:
        session.rollback()
        logger.error("Failed to delete conversations: %s", e)
        return jsonify({"error": "failed to delete conversations"}), 500
    finally:
        session.close()


@admin_bp.route("/conversations/<phone_number>", methods=["DELETE"])
def delete_conversation_for_number(phone_number):
    """Deletes chat history for a single phone number only. Irreversible.
    Requires header: X-Admin-Key: <ADMIN_API_KEY>
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


# --- Leads -----------------------------------------------------------

@admin_bp.route("/leads/<phone_number>", methods=["GET"])
def get_lead(phone_number):
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


# --- Documents (RAG chunk inspection) ---------------------------------

@admin_bp.route("/documents", methods=["GET"])
def view_documents():
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


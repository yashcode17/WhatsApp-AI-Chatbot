# app.py
"""
Application entrypoint. This file's only job is to build the Flask app and
register the user-facing and admin blueprints on it - one process, one
deployment, one codebase, but the routes underneath are cleanly split:

- user_routes.py -> /test, /webhook       (buyers, Meta - no auth)
- admin_routes.py -> /admin/*              (you - requires X-Admin-Key)

Shared config, DB models, and business logic live in core.py so both
blueprint modules can import from it without a circular import back to
this file.

NOTE: SessionLocal, DocumentChunk, and GROQ_API_KEY are imported here and
re-exported at module level ON PURPOSE - ingest_documents.py and
clear_documents.py do `from app import SessionLocal, DocumentChunk, ...`
and are left untouched, so this file has to keep providing those names.
"""

from flask import Flask

from core import SessionLocal, DocumentChunk, GROQ_API_KEY  # re-exported - do not remove, see note above
from user_routes import user_bp
from admin_routes import admin_bp

app = Flask(__name__)
app.register_blueprint(user_bp)
app.register_blueprint(admin_bp)

if __name__ == "__main__":
    app.run(port=5000, debug=True)


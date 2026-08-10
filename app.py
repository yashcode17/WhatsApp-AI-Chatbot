# app.py
"""
Application entrypoint. This file's only job is to build the Flask app and
register the user-facing and admin blueprints on it - one process, one
deployment, one codebase, but the routes underneath are cleanly split:

  - user_routes.py -> /test, /webhook         (buyers, Meta - no auth)
  - admin_routes.py -> /admin/*               (you - requires X-Admin-Key)

Shared config, DB models, and business logic live in core.py so both
blueprint modules can import from it without a circular import back to
this file.

NOTE: SessionLocal, DocumentChunk, and GROQ_API_KEY are imported here and
re-exported at module level ON PURPOSE - ingest_documents.py and
clear_documents.py do `from app import SessionLocal, DocumentChunk, ...`
and are left untouched, so this file has to keep providing those names.
"""

from flask import Flask
from flasgger import Swagger

from core import SessionLocal, DocumentChunk, GROQ_API_KEY  # re-exported - do not remove, see note above
from user_routes import user_bp
from admin_routes import admin_bp

app = Flask(__name__)
app.register_blueprint(user_bp)
app.register_blueprint(admin_bp)

# --- Swagger / OpenAPI docs ----------------------------------------------
# Renders an interactive UI at /apidocs. Each route documents itself via a
# YAML docstring (see user_routes.py / admin_routes.py) - nothing else to
# register here. "User" routes need no auth; "Admin" routes are marked with
# the AdminKey security scheme, which maps to the X-Admin-Key header your
# admin endpoints already require.
app.config["SWAGGER"] = {
    "title": "Real Estate WhatsApp Bot API",
    "uiversion": 3,
    "specs_route": "/apidocs/",
}

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Real Estate WhatsApp Bot API",
        "description": (
            "Backend for the WhatsApp buyer-chat bot. 'User' endpoints are "
            "what Meta/buyers hit (no auth). 'Admin' endpoints are what the "
            "agent-facing tooling hits, and all require the X-Admin-Key header."
        ),
        "version": "1.0.0",
    },
    "tags": [
        {"name": "User", "description": "Public-facing: WhatsApp webhook + health check. No auth."},
        {"name": "Admin", "description": "Requires X-Admin-Key header. Conversation/lead/document data."},
    ],
    "securityDefinitions": {
        "AdminKey": {
            "type": "apiKey",
            "name": "X-Admin-Key",
            "in": "header",
            "description": "Required for all /admin/* routes. Set via the ADMIN_API_KEY env var.",
        }
    },
}

Swagger(app, template=swagger_template)


if __name__ == "__main__":
    app.run(port=5000, debug=True)


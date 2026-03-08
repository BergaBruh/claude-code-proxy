from proxy.routes.messages import register_messages_routes
from proxy.routes.token_count import register_token_count_routes
from proxy.routes.health import register_health_routes


def register_routes(app):
    """Register all route handlers on the FastAPI app."""
    register_messages_routes(app)
    register_token_count_routes(app)
    register_health_routes(app)

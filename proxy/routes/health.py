def register_health_routes(app):
    """Register the health check endpoint."""

    @app.get("/")
    async def root():
        return {"message": "Anthropic Proxy for LiteLLM"}

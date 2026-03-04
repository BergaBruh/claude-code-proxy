import logging

from fastapi import Request

logger = logging.getLogger("proxy")


def register_middleware(app):
    """Register HTTP middleware on the FastAPI app."""

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        # Get request details
        method = request.method
        path = request.url.path

        # Log only basic request details at debug level
        logger.debug(f"Request: {method} {path}")

        # Process the request and get the response
        response = await call_next(request)

        return response

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, APIRouter, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .base import create_table_router
from .. import local_async

# Configure logging
logger = logging.getLogger(__name__)


# Create routers for each table
def create_all_routers() -> list[APIRouter]:
    """Create all table routers.
    
    Returns
    -------
    list[APIRouter]
        List of all FastAPI routers to include in the app
    """
    
    routers = [
        create_table_router(
            "algorithms",
            local_async.algorithm,
        ),
        create_table_router(
            "bands",
            local_async.band,
        ),
        create_table_router(
            "catalog_band_assocs",
            local_async.catalog_band_assoc,
        ),
        create_table_router(
            "catalog_tags",
            local_async.catalog_tag,
        ),
        create_table_router(
            "datasets",
            local_async.dataset,
        ),
        create_table_router(
            "estimates",
            local_async.estimates,
        ),
        create_table_router(
            "estimators",
            local_async.estimator,
        ),
        create_table_router(
            "models",
            local_async.model,
        ),
    ]
    
    return routers


def register_all_routers(app: FastAPI, prefix: str = "/api/v1") -> None:
    """Register all routers with a FastAPI app.
    
    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance
    prefix : str
        URL prefix for all API routes (default: "/api/v1")
    """
    for router in create_all_routers():
        # Include router with the API version prefix
        app.include_router(router, prefix=prefix)
        logger.info(f"Registered router: {router.prefix} at {prefix}{router.prefix}")


def add_rate_limiting(app: FastAPI, default_limits: list[str] = None) -> Limiter:
    """Add rate limiting to the FastAPI app.
    
    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance
    default_limits : list[str]
        Default rate limits (e.g., ["200 per day", "50 per hour"])
    
    Returns
    -------
    Limiter
        The limiter instance
    
    Example
    -------
    >>> from fastapi import FastAPI
    >>> app = FastAPI()
    >>> limiter = add_rate_limiting(app, ["1000 per day", "100 per hour"])
    
    Note
    ----
    Requires: pip install slowapi
    For production, use Redis: storage_uri="redis://localhost:6379"
    """
    try:
        if default_limits is None:
            default_limits = ["1000 per day", "100 per hour"]
        
        limiter = Limiter(
            key_func=get_remote_address,
            default_limits=default_limits,
            storage_uri="memory://",  # Use Redis in production: redis://localhost:6379
        )
        
        # Add limiter to app state so it's accessible in routes
        app.state.limiter = limiter
        
        # Add exception handler for rate limit exceeded
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        
        logger.info(f"Rate limiting enabled: {default_limits}")
        return limiter
    except ImportError:
        logger.warning("slowapi not installed. Rate limiting not available.")
        logger.warning("Install with: pip install slowapi")
        return None


def add_health_check(app: FastAPI) -> None:
    """Add a health check endpoint to the app.
    
    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance
    """
    @app.get("/health", tags=["health"])
    async def health_check():
        """Health check endpoint.
        
        Returns:
            200: Service is healthy
            503: Service is unhealthy
        """
        try:
            # Add any health checks here (database connection, etc.)
            return {
                "status": "healthy",
                "service": "api",
                "version": "1.0.0"
            }
        except Exception as e:
            logger.exception("Health check failed")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "unhealthy",
                    "error": str(e) if app.debug else "Service unavailable"
                }
            )
    
    logger.info("Health check endpoint added at /health")


def add_error_handlers(app: FastAPI) -> None:
    """Add global error handlers to the app.
    
    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance
    """
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        """Handle 404 errors."""
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "Endpoint not found"}
        )
    
    @app.exception_handler(405)
    async def method_not_allowed_handler(request: Request, exc):
        """Handle 405 errors."""
        return JSONResponse(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
            content={"error": "Method not allowed"}
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors."""
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation error",
                "details": exc.errors()
            }
        )
    
    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc):
        """Handle 500 errors."""
        logger.exception("Internal server error")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "details": str(exc) if app.debug else None
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions."""
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "details": str(exc) if app.debug else "An unexpected error occurred"
            }
        )
    
    logger.info("Global error handlers added")


def add_cors_middleware(
    app: FastAPI,
    allow_origins: list[str] = None,
    allow_credentials: bool = True,
    allow_methods: list[str] = None,
    allow_headers: list[str] = None,
) -> None:
    """Add CORS middleware to the app.
    
    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance
    allow_origins : list[str]
        Allowed origins (default: ["*"])
    allow_credentials : bool
        Whether to allow credentials (default: True)
    allow_methods : list[str]
        Allowed HTTP methods (default: ["*"])
    allow_headers : list[str]
        Allowed headers (default: ["*"])
    """
    if allow_origins is None:
        allow_origins = ["*"]  # In production, specify exact origins
    
    if allow_methods is None:
        allow_methods = ["*"]
    
    if allow_headers is None:
        allow_headers = ["*"]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=allow_methods,
        allow_headers=allow_headers,
    )
    
    logger.info(f"CORS middleware added with origins: {allow_origins}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events.
    
    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance
    """
    # Startup
    logger.info("Starting up application...")
    # Add any startup logic here (database connections, etc.)
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    # Add any cleanup logic here (close database connections, etc.)


def create_fastapi_app(
    title: str = "API",
    description: str = "FastAPI application",
    version: str = "1.0.0",
    enable_rate_limiting: bool = False,
    rate_limits: list[str] = None,
    enable_cors: bool = False,
    cors_origins: list[str] = None,
    debug: bool = False,
) -> FastAPI:
    """Create and configure a FastAPI application.
    
    Parameters
    ----------
    title : str
        Application title
    description : str
        Application description
    version : str
        Application version
    enable_rate_limiting : bool
        Whether to enable rate limiting (default: False)
    rate_limits : list[str]
        Rate limit rules (default: ["1000 per day", "100 per hour"])
    enable_cors : bool
        Whether to enable CORS (default: False)
    cors_origins : list[str]
        Allowed CORS origins (default: ["*"])
    debug : bool
        Debug mode (default: False)
    
    Returns
    -------
    FastAPI
        Configured FastAPI application instance
    
    Example
    -------
    >>> app = create_fastapi_app(
    ...     title="My API",
    ...     enable_rate_limiting=True,
    ...     enable_cors=True,
    ...     debug=True
    ... )
    """
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        lifespan=lifespan,
        debug=debug,
    )
    
    # Add CORS if enabled
    if enable_cors:
        add_cors_middleware(app, allow_origins=cors_origins)
    
    # Register all routers
    register_all_routers(app)
    
    # Add health check
    add_health_check(app)
    
    # Add error handlers
    add_error_handlers(app)
    
    # Add rate limiting if enabled
    if enable_rate_limiting:
        add_rate_limiting(app, default_limits=rate_limits)
    
    logger.info("FastAPI app setup complete")
    
    return app


def setup_fastapi_app(
    app: FastAPI,
    enable_rate_limiting: bool = False,
    rate_limits: list[str] = None,
    enable_cors: bool = False,
    cors_origins: list[str] = None,
) -> None:
    """Complete setup for an existing FastAPI app with all routers and optional features.
    
    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance
    enable_rate_limiting : bool
        Whether to enable rate limiting (default: False)
    rate_limits : list[str]
        Rate limit rules (default: ["1000 per day", "100 per hour"])
    enable_cors : bool
        Whether to enable CORS (default: False)
    cors_origins : list[str]
        Allowed CORS origins (default: ["*"])
    
    Example
    -------
    >>> from fastapi import FastAPI
    >>> app = FastAPI()
    >>> setup_fastapi_app(app, enable_rate_limiting=True, enable_cors=True)
    """
    # Add CORS if enabled
    if enable_cors:
        add_cors_middleware(app, allow_origins=cors_origins)
    
    # Register all routers
    register_all_routers(app)
    
    # Add health check
    add_health_check(app)
    
    # Add error handlers
    add_error_handlers(app)
    
    # Optional features
    if enable_rate_limiting:
        add_rate_limiting(app, default_limits=rate_limits)
    
    logger.info("FastAPI app setup complete")


# Create the default app instance
fastapi_app = create_fastapi_app(
    title="Database API",
    description="RESTful API for database operations",
    version="1.0.0",
    enable_rate_limiting=True,
    enable_cors=True,
    debug=False,
)



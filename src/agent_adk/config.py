"""
Configuration module for the ADK agent application.

Loads and validates environment variables required for the application to run.
Handles configuration for LLM gateway, observability (OpenTelemetry, Pyroscope),
and tracing (Langfuse).

Raises:
    OSError: When required environment variables are not set.
"""

import json
import os

from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    """
    Retrieve a required environment variable.

    Args:
        key: The environment variable name.

    Returns:
        str: The environment variable value.

    Raises:
    """
    value = os.getenv(key)
    if not value:
        raise OSError(f"Required environment variable '{key}' is not set.")
    return value


def _optional(key: str, default: str = "") -> str:
    """
    Retrieve an optional environment variable or the default if not set.

    Args:
        key: The environment variable name.
        default: The default value if the environment variable is not set.
                 Defaults to empty string.

    Returns:
        str: The environment variable value, or the default if not set.
    """
    return os.getenv(key, default) or default


# Bifrost / LLM gateway
BIFROST_GATEWAY_URL = _require("BIFROST_GATEWAY_URL")
BIFROST_API_KEY = _require("BIFROST_API_KEY")
BIFROST_MODEL = _require("BIFROST_MODEL")

# SSL
SSL_CERT_PATH = _require("SSL_CERT_PATH")
OTEL_CLIENT_CERT = _require("OTEL_CLIENT_CERT_PATH")
OTEL_CLIENT_KEY = _require("OTEL_CLIENT_KEY_PATH")

# OpenTelemetry endpoints
OTEL_METRICS_ENDPOINT = _require("OTEL_METRICS_ENDPOINT")
OTEL_TRACES_ENDPOINT = _require("OTEL_TRACES_ENDPOINT")
OTEL_LOGS_ENDPOINT = _require("OTEL_LOGS_ENDPOINT")
OTEL_SERVICE_NAME = _require("OTEL_SERVICE_NAME")
OTEL_SERVICE_VERSION = _require("OTEL_SERVICE_VERSION")
OTEL_USER = _require("LANGFUSE_PUBLIC_KEY")
OTEL_KEY = _require("LANGFUSE_SECRET_KEY")

# Pyroscope
PYROSCOPE_SERVER_URL = _require("PYROSCOPE_SERVER_URL")

# Langfuse prompt and feedback settings
LANGFUSE_PROMPT_NAME = _optional("LANGFUSE_PROMPT_NAME")
LANGFUSE_PROMPT_LABEL = _optional("LANGFUSE_PROMPT_LABEL")
LANGFUSE_USER_ID = _optional("LANGFUSE_USER_ID")
LANGFUSE_TRACE_NAME = _optional("LANGFUSE_TRACE_NAME")
LANGFUSE_TAGS = json.loads(_optional("LANGFUSE_TAGS", "[]"))

# Application metadata for tracing
APP_ENV = _optional("APP_ENV", "development")
APP_VERSION = _optional("APP_VERSION", "0.1.0")
MODEL_PROVIDER = _optional("MODEL_PROVIDER", "bifrost")
AGENT_ROUTE = _optional("AGENT_ROUTE", "cli-chat")

"""Interactive Google ADK agent with Langfuse and OpenTelemetry telemetry."""

import asyncio
import json
import logging
import time
import uuid
from typing import Any

import pyroscope
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from langfuse import Langfuse
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider

from agent_adk import config
from agent_adk.telemetry import (
    AgentMetrics,
    setup_logs,
    setup_metrics,
    setup_openlit,
    setup_profiling,
    setup_traces,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# These objects are initialized explicitly by bootstap()
_langfuse: Langfuse | None = None
_meter_provider: MeterProvider | None = None
_logger_provider: LoggerProvider | None = None
_agent_metrics: AgentMetrics | None = None
_langfuse_prompt_obj = None
_system_prompt: str | None = None
_bootstrapped = False


def _setup_telemetry() -> tuple[MeterProvider, LoggerProvider]:
    """Configure metrics, traces, logs, instrumentation, and profiling.

    Returns:
        A tuple containing the configured meter and logger providers.

    Raises:
        Exception: Propagates telemetry configuration failures.
    """
    meter_provider = setup_metrics(
        endpoint=config.OTEL_METRICS_ENDPOINT,
        ssl_cert=config.SSL_CERT_PATH,
        client_cert=config.OTEL_CLIENT_CERT,
        client_key=config.OTEL_CLIENT_KEY,
    )

    logger_provider = setup_logs(
       endpoint=config.OTEL_LOGS_ENDPOINT,
       ssl_cert=config.SSL_CERT_PATH,
       client_cert=config.OTEL_CLIENT_CERT,
       client_key=config.OTEL_CLIENT_KEY,
    )

    # Langfuse must register its tracer provider before OpenLIT is configured.
    setup_traces(
        endpoint=config.OTEL_TRACES_ENDPOINT,
        ssl_cert=config.SSL_CERT_PATH,
        user = config.OTEL_USER,
        key= config.OTEL_KEY,
        client_cert=config.OTEL_CLIENT_CERT,
        client_key=config.OTEL_CLIENT_KEY,
    )

    setup_openlit()

    setup_profiling(
        server_url=config.PYROSCOPE_SERVER_URL,
        tags={"environment": config.APP_ENV, "app_version": config.APP_VERSION},
    )

    return meter_provider, logger_provider


def _latency_bucket(latency_ms: float) -> str:
    """Convert latency in milliseconds into a reporting bucket.

    Args:
        latency_ms: Request latency in milliseconds.

    Returns:
        The categorical latency bucket name.
    """
    if latency_ms < 1_000:
        return "lt_1s"
    if latency_ms < 3_000:
        return "1s_3s"
    return "gte_3s"


def _log_event(
    event: str,
    status: str,
    *,
    level: int = logging.INFO,
    error: Exception | None = None,
    **context: Any,
) -> None:
    """Emit a structured JSON event to the application logger.

    Args:
        event: Event name.
        status: Event status.
        level: Logging level.
        error: Optional exception associated with the event.
        **context: Additional structured context.
    """
    payload: dict[str, Any] = {
        "event": event,
        "service_name": config.OTEL_SERVICE_NAME,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "context": context,
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)

    logger.log(level, json.dumps(payload, sort_keys=True, default=str))


def _langfuse_prompt():
    """Retrieve the configured prompt from Langfuse.

    Returns:
        The Langfuse prompt object.

    Raises:
        AttributeError: If Langfuse has not been initialized.
    """
    if _langfuse is None:
        raise RuntimeError("Langfuse has not been initialized")

    return _langfuse.get_prompt(
        config.LANGFUSE_PROMPT_NAME,
        label=config.LANGFUSE_PROMPT_LABEL,
    )


def bootstrap() -> None:
    """Initialize Langfuse, OTel telemetry, and the system prompt.

    This function intentionally performs initialization explicitly rather than
    during module import, preventing network calls when the module is imported.
    """
    global _langfuse
    global _meter_provider
    global _logger_provider
    global _agent_metrics
    global _system_prompt
    global _langfuse_prompt_obj
    global _bootstrapped

    _langfuse = Langfuse(
        environment=config.APP_ENV, release=config.APP_VERSION, should_export_span=lambda _span: False
    )
    _meter_provider, _logger_provider = _setup_telemetry()
    _agent_metrics = AgentMetrics()

    _langfuse_prompt_obj = _langfuse_prompt()
    _system_prompt = _langfuse_prompt_obj.prompt
    _bootstrapped = True


def create_adk_agent(model_name: str | None = None) -> tuple[Runner, InMemorySessionService] | None:
    """Create an ADK runner backed by the configured OpenAI-compatible model.

    Args:
        model_name: Optional model override.

    Returns:
        An ADK runner and its in-memory session service, or None if creation fails.
    """
    try:
        model = model_name or config.BIFROST_MODEL
        logger.info("Creating agent with model: %s", model)

        model_client = LiteLlm(
            model=f"openai/{model}",
            api_base=config.BIFROST_GATEWAY_URL,
            api_key=config.BIFROST_API_KEY,
        )
        agent = LlmAgent(
            name="bifrost_agent",
            model=model_client,
        )
        session_service = InMemorySessionService()
        runner = Runner(
            app_name=config.OTEL_SERVICE_NAME,
            agent=agent,
            session_service=session_service,
        )

        logger.info("Agent created successfully")
        return runner, session_service

    except Exception as error:
        logger.error("Failed to create agent: %s", error, exc_info=True)
        return None


async def _ensure_session(session_service: InMemorySessionService, session_id: str) -> None:
    """Create the ADK session used to retain conversation history."""
    await session_service.create_session(
        app_name=config.OTEL_SERVICE_NAME,
        user_id=config.LANGFUSE_USER_ID or "cli-user",
        session_id=session_id,
    )


async def _run_adk_turn(
    runner: Runner,
    session_service: InMemorySessionService,
    user_input: str,
    session_id: str,
) -> str | None:
    """Send one message through ADK and return its final response text."""
    user_id = config.LANGFUSE_USER_ID or "cli-user"
    session = await session_service.get_session(
        app_name=config.OTEL_SERVICE_NAME,
        user_id=user_id,
        session_id=session_id,
    )
    if session is None:
        await _ensure_session(session_service, session_id)

    message = types.Content(role="user", parts=[types.Part(text=user_input)])
    final_text = None
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text
    return final_text


def _process_turn(
    runner: Runner,
    session_service: InMemorySessionService,
    user_input: str,
    session_id: str,
    base_trace_metadata: dict[str, Any],
    attrs: dict[str, Any],
) -> None:
    """Process one user turn, record telemetry, and collect feedback.

    Args:
        runner: ADK runner.
        user_input: User's input text.
        session_id: Conversation identifier.
        base_trace_metadata: Metadata shared by all turns.
        attrs: Metric attributes for the current session.
    """
    if _agent_metrics is None or _langfuse is None:
        raise RuntimeError("Telemetry has not been initialized")

    request_id = str(uuid.uuid4())
    trace_metadata = {**base_trace_metadata, "request_id": request_id}
    _agent_metrics.request_counter.add(1, attrs)
    start = time.monotonic()
    trace_id = ""

    try:
        # Use a Langfuse observation so the turn appears in Langfuse sessions.
        with _langfuse.start_as_current_observation(
            name=config.LANGFUSE_TRACE_NAME,
            as_type="span",
            input=user_input,
        ) as turn_span:
            trace_id = turn_span.trace_id
            turn_span.update(metadata=trace_metadata)
            try:
                response = asyncio.run(_run_adk_turn(runner, session_service, user_input, session_id))
            except Exception as error:
                _log_event(
                    "agent.invoke.error",
                    "error",
                    level=logging.ERROR,
                    error=error,
                    trace_name=config.LANGFUSE_TRACE_NAME,
                    request_id=request_id,
                    session_id=session_id,
                    model=config.BIFROST_MODEL,
                )
                # Flag the span itself as failed, not just the trace-level score
                turn_span.update(level="ERROR", status_message=str(error))
                raise
            
            last_message = response
            if last_message is not None:
                turn_span.update(output=last_message)

        latency_ms = (time.monotonic() - start) * 1000
        _agent_metrics.request_duration.record(latency_ms, attrs)

        _log_event(
            "agent.invoke.complete",
            "success",
            trace_name=config.LANGFUSE_TRACE_NAME,
            request_id=request_id,
            session_id=session_id,
            model=config.BIFROST_MODEL,
        )

        if last_message is not None:
            print(f"\nAgent: {last_message}\n")
            _langfuse.create_score(trace_id=trace_id, name="turn_success", value=1.0, data_type="NUMERIC")

            feedback = input("Rate this response (y/n, Enter to skip): ").strip().lower()
            if feedback in {"y", "n"}:
                _langfuse.create_score(
                    trace_id=trace_id,
                    name="user_feedback",
                    value=feedback == "y",
                    data_type="BOOLEAN",
                )
        else:
            logger.error("Unexpected response structure: %s", response)
            _langfuse.create_score(
                trace_id=trace_id,
                name="turn_success",
                value=0.0,
                data_type="NUMERIC",
                comment="empty response",
            )

        _langfuse.create_score(
            trace_id=trace_id,
            name="latency_bucket",
            value=_latency_bucket(latency_ms),
            data_type="CATEGORICAL",
        )

    except Exception as error:
        latency_ms = (time.monotonic() - start) * 1000
        _agent_metrics.request_duration.record(latency_ms, attrs)
        _agent_metrics.error_counter.add(1, attrs)

        _log_event(
            "agent.invoke.failed",
            "error",
            level=logging.ERROR,
            error=error,
            trace_name=config.LANGFUSE_TRACE_NAME,
            request_id=request_id,
            session_id=session_id,
            model=config.BIFROST_MODEL,
        )

        if trace_id:
            _langfuse.create_score(
                trace_id=trace_id,
                name="turn_success",
                value=0.0,
                data_type="NUMERIC",
                comment=str(error),
            )


def _shutdown_telemetry() -> None:
    """Flush and shut down telemetry providers without masking errors."""
    if _langfuse is not None:
        try:
            _langfuse.flush()
        except Exception:
            logger.exception("Failed to flush Langfuse telemetry")

    if _meter_provider is not None:
        try:
            _meter_provider.shutdown()
        except Exception:
            logger.exception("Failed to shut down the meter provider")

    if _logger_provider is not None:
        try:
            _logger_provider.shutdown()
        except Exception:
            logger.exception("Failed to shut down the logger provider")

    try:
        pyroscope.shutdown()
    except Exception:
        logger.exception("Failed to shut down Pyroscope")


def main() -> None:
    """Run the interactive chat loop until the user exits."""
    bootstrap()

    adk_runtime = create_adk_agent()
    if adk_runtime is None:
        logger.error("Failed to create ADK agent, exiting")
        return
    runner, session_service = adk_runtime

    session_id = str(uuid.uuid4())
    attrs = {"model": config.BIFROST_MODEL, "session_id": session_id}
    base_trace_metadata: dict[str, Any] = {
        "environment": config.APP_ENV,
        "app_version": config.APP_VERSION,
        "model_provider": config.MODEL_PROVIDER,
        "route": config.AGENT_ROUTE,
        "langfuse_session_id": session_id,
        "langfuse_user_id": config.LANGFUSE_USER_ID,
    }

    if _agent_metrics is None:
        raise RuntimeError("Agent metrics have not been initialized")

    _agent_metrics.active_sessions.add(1, {"session_id": session_id})
    _log_event(
        "agent.invoke.start",
        "start",
        trace_name=config.LANGFUSE_TRACE_NAME,
        session_id=session_id,
        model=config.BIFROST_MODEL,
    )

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            _agent_metrics.active_sessions.add(-1, {"session_id": session_id})
            break
        if not user_input:
            continue

        _process_turn(runner, session_service, user_input, session_id, base_trace_metadata, attrs)


if __name__ == "__main__":
    try:
        main()
    finally:
        if _bootstrapped:
            _shutdown_telemetry()

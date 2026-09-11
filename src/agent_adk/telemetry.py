"""Telemetry setup for OpenTelemetry, Pyroscope, and OpenLIT integrations.

Provides configuration functions for distributed tracing (Tempo/Grafana),
metrics collection (OTLP), logging (Loki), CPU/memory profiling (Pyroscope),
and automatic LLM instrumentation (OpenLit).
"""

import base64
import logging
import openlit
import pyroscope
from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from agent_adk import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_metrics(
    endpoint: str,
    ssl_cert: str,
    client_cert: str | None = None,
    client_key: str | None = None,
    export_interval_ms: int = 10_000,
) -> MeterProvider:
    """Initialize OpenTelemetry metrics exporter with OTLP protocol.

    Args:
        endpoint: OTLP HTTP endpoint URL.
        ssl_cert: Path to CA certificate file for server verification.
        client_cert: Path to client TLS certificate for mTLS (optional).
        client_key: Path to client TLS key for mTLS (optional).
        export_interval_ms: Metrics export interval in milliseconds. Defaults to 10s.

    Returns:
        Configured MeterProvider instance.
    """
    exporter = OTLPMetricExporter(
        endpoint=endpoint,
        certificate_file=ssl_cert,
        client_certificate_file=client_cert,
        client_key_file=client_key,
    )
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=export_interval_ms)
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)
    return provider


def setup_traces(
    endpoint: str,
    ssl_cert: str,
    user: str,
    key: str,
    client_cert: str | None = None,
    client_key: str | None = None,
) -> None:
    """Add an OTLP span exporter to the active TracerProvider (Tempo/Grafana).

    Args:
        endpoint: OTLP HTTP endpoint URL for span submission.
        ssl_cert: Path to CA certificate file for server verification.
        client_cert: Path to client TLS certificate for mTLS.
        client_key: Path to client TLS key for mTLS.
    """

    headers = _otlp_headers(user, key)

    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        certificate_file=ssl_cert,
        client_certificate_file=client_cert,
        client_key_file=client_key,
        headers=headers,
    )
    tracer_provider = trace.get_tracer_provider()

    if not isinstance(tracer_provider, SDKTracerProvider):
        raise RuntimeError(
            "setup_traces() requires an SDK TracerProvider to already be registered. "
            f"{type(tracer_provider).__name__}."
        )
    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))


def setup_logs(
    endpoint: str,
    ssl_cert: str,
    client_cert: str | None = None,
    client_key: str | None = None,
    log_level: int = logging.INFO,
) -> LoggerProvider:
    """Export Python logging records to Loki via OTLP.

    Attaches an OTel LoggingHandler to the root logger so every
    logger.info() / logger.warning() / logger.error() call is
    forwarded to Loki automatically.

    Args:
        endpoint: OTLP HTTP endpoint URL for log submission.
        ssl_cert: Path to CA certificate file for server verification.
        client_cert: Path to client TLS certificate for mTLS (optional).
        client_key: Path to client TLS key for mTLS (optional).
        log_level: Minimum logging level to export. Defaults to logging.INFO.

    Returns:
        Configured LoggerProvider instance.
    """
    exporter = OTLPLogExporter(
        endpoint=endpoint,
        certificate_file=ssl_cert,
        client_certificate_file=client_cert,
        client_key_file=client_key,
    )
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    set_logger_provider(logger_provider)

    # Bridge Python's standard logging → OTEL
    handler = LoggingHandler(level=log_level, logger_provider=logger_provider)
    logging.getLogger().addHandler(handler)

    return logger_provider


def _otlp_headers(user: str, key: str) -> dict[str, str] | None:
    """Return Basic auth headers for OTLP endpoint"""
    token = base64.b64encode(f"{user}:{key}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def setup_profiling(
    server_url: str,
    app_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> None:
    """Start continuous CPU + memory profiling and push to Pyroscope.

    Runs a background thread — no instrumentation needed in application code.

    Args:
        server_url: Pyroscope server URL.
        app_name: Application name for Pyroscope.
        tags: Optional tags dictionary to attach to profiling data.
    """
    pyroscope.configure(
        application_name=app_name or config.OTEL_SERVICE_NAME,
        server_address=server_url,
        cpu_enabled=True,
        mem_enabled=True,
        tags=tags or {},
        # enable_logging=True,
    )


def setup_openlit() -> None:
    """Auto-instrument LLM calls using OpenLit.

    Reuses the global TracerProvider and MeterProvider."""
    logging.getLogger("openlit").setLevel(logging.WARNING)
    openlit.init(collect_system_metrics=True, collect_gpu_stats=False, max_content_length=2000)
    # pricing_json="pricing.json"


class AgentMetrics:
    """Manages custom metrics for agent invocations and session tracking.

    Metrics tracked:
        - agent.requests.total: Total invocation count
        - agent.errors.total: Total error count
        - agent.request.duration: Invocation latency (ms)
        - agent.active_sessions: Current active session count
    """

    def __init__(self) -> None:
        meter = metrics.get_meter(config.OTEL_SERVICE_NAME, version=config.OTEL_SERVICE_VERSION)

        self.request_counter = meter.create_counter(
            "agent.requests.total",
            unit="1",
            description="Total number of agent invocations",
        )
        self.error_counter = meter.create_counter(
            "agent.errors.total",
            unit="1",
            description="Total number of failed invocations",
        )
        self.request_duration = meter.create_histogram(
            "agent.request.duration",
            unit="ms",
            description="Agent invocation latency",
        )
        self.active_sessions = meter.create_up_down_counter(
            "agent.active_sessions",
            unit="1",
            description="Number of active conversation sessions",
        )

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml README.md uv.lock ./
COPY src ./src
COPY certs ./certs

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates
RUN update-ca-certificates

ENV UV_SYSTEM_CERTS=true
ENV SSL_CERT_DIR=/etc/ssl/certs

RUN uv sync
RUN uv pip install --system-certs requests

EXPOSE 8080

CMD ["uv", "run", "src/agent_adk/agent.py"] 
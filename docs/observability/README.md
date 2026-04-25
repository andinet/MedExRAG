# Observability

MedExRAG ships with a four-tier observability stack. Each tier is independently toggleable and the application degrades gracefully if a backend is unavailable.

| Tier | Tool | Purpose | Enable Flag |
|------|------|---------|-------------|
| M1 | LangSmith | LLM/agent call tracing (prompts, tokens, chains) | `LANGSMITH_ENABLED` |
| M2 | Prometheus | Time-series metrics (latency, throughput, errors) | `PROMETHEUS_ENABLED` |
| M3 | OpenTelemetry / Jaeger | End-to-end distributed tracing across pipeline stages | `OTEL_ENABLED` |
| M4 | Grafana | Dashboards and alert visualization on top of Prometheus | (always on with stack) |

## Service URLs

| Service | URL | Notes |
|---------|-----|-------|
| Streamlit UI | http://localhost:8501 | Application |
| Grafana | http://localhost:3000 | Default credentials `admin` / `admin` — **change for production** |
| Prometheus | http://localhost:9090 | Query UI and alert state |
| Jaeger UI | http://localhost:16686 | Trace search and waterfall views |
| Metrics endpoint | http://localhost:8002/metrics | Raw Prometheus exposition (scraped by Prometheus) |
| OTEL collector (gRPC) | localhost:4317 | OTLP ingest from app |
| OTEL collector health | http://localhost:13133 | Liveness check |

## Starting the Stack

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml \
  up streamlit prometheus grafana jaeger otel-collector
```

CPU mode: drop `-f docker/docker-compose.gpu.yml`.

## Environment Variables

```bash
# LangSmith (M1)
LANGSMITH_ENABLED=true
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_xxxxxxxx       # from https://smith.langchain.com/
LANGCHAIN_PROJECT=medical-xray-rag

# Prometheus (M2)
PROMETHEUS_ENABLED=true
METRICS_PORT=8002

# OpenTelemetry / Jaeger (M3)
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=otel-collector:4317
OTEL_SERVICE_NAME=medical-rag-streamlit

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

Set any flag to `false` to disable that tier; the app will log a warning and continue.

## Exported Prometheus Metrics

Defined in `src/medexrag/observability/metrics_config.py`. All metrics are prefixed `medical_rag_`.

### Counters

| Name | Labels | Meaning |
|------|--------|---------|
| `medical_rag_analysis_requests_total` | `use_rag`, `status` | Total X-ray analysis requests |
| `medical_rag_vlm_requests_total` | `operation`, `status` | VLM inference calls |
| `medical_rag_literature_searches_total` | `has_threshold` | Vector store search calls |
| `medical_rag_literature_ingestion_total` | `status` | Documents ingested |
| `medical_rag_tool_invocations_total` | `tool_name`, `status` | Agent tool calls |

### Histograms

| Name | Labels | Buckets (s) |
|------|--------|-------------|
| `medical_rag_vlm_inference_seconds` | `operation` | 0.5 – 120 |
| `medical_rag_vector_search_seconds` | — | 0.01 – 5 |
| `medical_rag_pipeline_total_seconds` | `use_rag` | 1 – 300 |
| `medical_rag_literature_retrieval_seconds` | — | 0.1 – 10 |
| `medical_rag_agent_step_seconds` | `agent_name` | 0.5 – 120 |

### Gauges

| Name | Meaning |
|------|---------|
| `medical_rag_active_requests` | In-flight analysis requests |
| `medical_rag_vector_store_chunks_total` | Current KB size |
| `medical_rag_model_loaded` | VLM ready (1) or not (0) |
| `medical_rag_sources_retrieved_last` | Sources returned by last search |

### Info

| Name | Meaning |
|------|---------|
| `medical_rag_service` | Service name, version, model |

## Grafana Dashboard

A pre-built dashboard ships with the repo and is auto-provisioned on container start.

- Dashboard JSON: `config/grafana/dashboards/medical-rag-overview.json`
- Provisioning: `config/grafana/provisioning/dashboards/dashboards.yaml`
- Datasource: `config/grafana/provisioning/datasources/datasources.yaml` (points to `http://prometheus:9090`)

The dashboard groups panels by request overview, VLM latency (p50/p95/p99), multi-agent step duration, and RAG vs non-RAG breakdown.

## Alerting

Prometheus alert rules: `config/observability/alerting/prometheus-alerts.yml`. Rule path is wired in `config/observability/prometheus.yml`. Reload after edits:

```bash
docker compose -f docker/docker-compose.yml restart prometheus
```

Default rules cover high VLM latency, high error rate, empty vector store, and Streamlit service down.

## OpenTelemetry Spans

Spans emitted by the application (see `src/medexrag/observability/tracing.py`):

- `streamlit_analysis_request` — root span per UI request (`src/medexrag/app.py`)
- `analyze_xray_pipeline`, `load_image`, `initial_vlm_analysis`, `vlm_generate`, `literature_retrieval`, `enhanced_vlm_analysis` — pipeline stages (`src/medexrag/pipeline.py`)
- `agent_analyst`, `agent_researcher`, `agent_diagnostician`, `agent_reporter` — multi-agent steps (`src/medexrag/agents.py`)

OTEL collector config: `config/observability/otel-collector-config.yaml` (OTLP gRPC receiver -> Jaeger exporter).

## References

- Prometheus query language: https://prometheus.io/docs/prometheus/latest/querying/basics/
- Jaeger UI guide: https://www.jaegertracing.io/docs/latest/frontend-ui/
- Grafana dashboards: https://grafana.com/docs/grafana/latest/dashboards/
- LangSmith: https://docs.smith.langchain.com/

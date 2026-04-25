# Agents Guide

LangGraph-based 4-agent workflow for evidence-based X-ray interpretation. An alternative to the direct two-stage RAG in `pipeline.py`, used when you want explicit, traceable reasoning steps.

Source: [src/medexrag/agents.py](../../src/medexrag/agents.py).

## Workflow

```
analyst -> researcher -> diagnostician -> reporter -> END
```

Graph definition: [agents.py:413](../../src/medexrag/agents.py#L413).

| Node | Role | Tool used |
|---|---|---|
| `analyst` | Examine X-ray, list findings | `ImageAnalysisTool` |
| `researcher` | Retrieve supporting literature | `LiteratureSearchTool` |
| `diagnostician` | Synthesize findings + literature | `DiagnosticReasoningTool` |
| `reporter` | Format final radiology report | (VLM directly) |

State is a `TypedDict` carrying `messages`, `image`, `initial_findings`, `literature_context`, `diagnostic_reasoning`, and `final_report`. See `AgentState` in [agents.py](../../src/medexrag/agents.py).

## Tools

All three tools subclass `langchain.tools.BaseTool`:

| Tool | Purpose |
|---|---|
| `LiteratureSearchTool` | Wraps `MedicalVectorStore.search()` for ChromaDB retrieval |
| `ImageAnalysisTool` | Calls the VLM on the current image with an instruction string |
| `DiagnosticReasoningTool` | Combines findings + literature into differential diagnoses |

The VLM is exposed to LangChain via a custom `LLM` subclass (`MedExRAGLLM`) that holds the current image in instance state.

## Invoking the Multi-Agent System

```python
from medexrag.agents import create_medical_agent
from PIL import Image

agent = create_medical_agent()  # builds graph + loads VLM + vector store
image = Image.open("data/xray_images/chest.dcm")

result = agent.analyze(image, clinical_question="Evaluate for pneumonia")

print(result["findings"])    # analyst output
print(result["literature"])  # researcher output (with sources)
print(result["diagnosis"])   # diagnostician output
print(result["report"])      # final formatted report
```

## When to Use Which

| Use case | Choose |
|---|---|
| Quick analysis, single VLM round | `MedicalRAGPipeline.analyze_xray()` |
| Need agent-level traceability, modular reasoning, audit trail | `create_medical_agent()` |
| Adding new diagnostic steps as nodes | `create_medical_agent()` |

The simple pipeline calls the VLM twice (initial + enhanced). The agent system calls the VLM 3-4 times (once per node that uses it), so it is slower but produces explicit intermediate outputs at every stage.

## Adding a New Agent

1. Define a node function `def my_agent(state: AgentState) -> AgentState:` that reads/writes state keys.
2. Optionally wrap a tool by subclassing `BaseTool` (see existing tools in [agents.py](../../src/medexrag/agents.py)).
3. Register and wire it into the graph builder near [agents.py:413](../../src/medexrag/agents.py#L413):

```python
workflow.add_node("my_agent", my_agent)
workflow.add_edge("diagnostician", "my_agent")
workflow.add_edge("my_agent", "reporter")
```

4. Extend `AgentState` with any new fields the node populates.

## Conditional Routing

The current graph is linear. For branching (e.g., skip literature search on a clean image), use `workflow.add_conditional_edges()`:

```python
workflow.add_conditional_edges(
    "analyst",
    lambda s: "researcher" if s["initial_findings"] else "reporter",
)
```

## Troubleshooting

- **Tool call returns empty literature** - vector store is empty; run `python -m medexrag.cli ingest data/medical_literature/`.
- **Agent system slow** - it runs the VLM 3-4 times; for quick checks use `MedicalRAGPipeline.analyze_xray()` instead.
- **State key missing in downstream node** - confirm the upstream node returns the key, and that it is declared on `AgentState`.

## References

- LangGraph: https://langchain-ai.github.io/langgraph/
- LangChain tools: https://python.langchain.com/docs/concepts/tools/

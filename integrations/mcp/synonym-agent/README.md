## Quick Start

```
# 1. Run Memgraph with a dataset loaded.
# 2. `business_rules.yaml` should contain data in the following format:
#        configuration:
#          business_rules:
#            - label: "label"
#              prop: "property"
#              explanation: "Explain to LLM when and how to use the label+property pair."
# 3.
export OPENAI_API_KEY=...
export PYTHONPATH={{script_dir}}/../../langgraph/synonym-agent
./init.bash
```

Add the implementation time (2025-03-16), MCP did NOT have framework support
for workflows. `langgraph`-based workflow was used instead.

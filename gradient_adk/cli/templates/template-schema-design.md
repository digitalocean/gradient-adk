# ADK Template Schema Design

## Problem

The current [gradient-adk-templates](https://github.com/digitalocean/gradient-adk-templates) repository contains example agents, not true reusable templates. Users must clone folders and manually modify code or config values to create agents.

## Goal

Define a schema-driven template system that allows users to create ADK agents without writing code, consumed consistently by:

- **ADK CLI** — `gradient agent create --template <name>` prompts for parameters in the terminal
- **Console UI** — renders the schema as a web form
- **Genie / MCP** — exposes templates as tool parameters

The schema is a single source of truth: one definition, three surfaces.

## Schema Location

The canonical home for template schemas is the **public API schema** (Jenni API). Template definitions and user-provided values will be stored in the database, served via API endpoints. The ADK CLI and Console will consume templates through these endpoints.

## Template Schema Specification

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | string | yes | Schema format version (e.g. `v1`) |
| `template_id` | string | yes | Unique identifier for the template |
| `name` | string | yes | Human-readable template name |
| `description` | string | yes | What this template does |
| `category` | string | yes | Template category (e.g. `rag`, `function-calling`, `multi-agent`) |
| `parameters` | array | yes | User-configurable inputs |
| `secrets` | array | no | Required credentials and environment variables |
| `agent_config` | object | yes | Agent configuration with `${parameter}` substitution |
| `metadata` | object | no | Tags, ownership, and other metadata |

### Parameter Fields

Each entry in `parameters` describes one user-facing input:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Parameter identifier (used in `${...}` substitution) |
| `label` | string | yes | Display label for CLI prompt or form field |
| `type` | string | yes | Input type (see supported types below) |
| `required` | boolean | yes | Whether the user must provide a value |
| `multiple` | boolean | no | If true, the parameter accepts zero, one, or many values (e.g. multiple KBs, multiple tools). Default false. |
| `default` | any | no | Default value if user doesn't provide one |
| `description` | string | no | Help text shown to the user |
| `source` | string | no | Dynamic data source (e.g. `serverless_inference_models`) |
| `resource_type` | string | no | For `resource_select` — the DO resource type to list |
| `options` | array | no | For `select` — static list of allowed values |

### Supported Parameter Types

| Type | Description | Rendered as |
|------|-------------|-------------|
| `string` | Single-line text input | Text field |
| `textarea` | Multi-line text input | Text area |
| `select` | Pick one from a list | Dropdown |
| `resource_select` | Pick a DigitalOcean resource from the user's account | Resource picker |
| `tool_select` | Select function tools for the agent | Tool picker (TBD — pending clarification on UX) |

### Secrets

Templates require credentials that are handled separately from regular parameters — they are masked, stored securely, and never displayed back to the user.

Each entry in `secrets` declares a required environment variable:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `env` | string | yes | Environment variable name |
| `label` | string | yes | Display label for the user |
| `description` | string | no | Help text explaining what the secret is for |
| `required` | boolean | yes | Whether the secret must be provided |

### `agent_config`

A normalized representation of the agent's configuration, independent of the underlying framework or code. Parameters are injected via `${parameter_name}` substitution.

Consumers (CLI, Console, renderer) interpret this config to produce the appropriate code and deployment artifacts. The rendering layer is a separate design step.

## Example: Knowledge Base RAG Template



```yaml
schema_version: v1
template_id: knowledge-base-rag
name: Knowledge Base RAG Agent
description: Answer questions using selected knowledge bases.
category: rag

parameters:
  - name: model
    label: Model
    type: select
    required: true
    source: serverless_inference_models
    description: Model used for inference through Serverless Inference.

  - name: knowledge_base_ids
    label: Knowledge Bases
    type: resource_select
    resource_type: knowledge_base
    multiple: true
    required: false
    description: Knowledge bases this agent can query.

  - name: system_prompt
    label: System Prompt
    type: textarea
    required: false
    default: You are a helpful assistant.
    description: System instructions for the agent.

  - name: tools
    label: Function Tools
    type: tool_select
    multiple: true
    required: false
    description: Tools the agent can call during execution.

secrets:
  - env: GRADIENT_MODEL_ACCESS_KEY
    label: Gradient Model Access Key
    required: true
    description: API key for Serverless Inference model access.

  - env: DIGITALOCEAN_API_TOKEN
    label: DigitalOcean API Token
    required: true
    description: API token for accessing DigitalOcean resources.

agent_config:
  model: ${model}
  prompt:
    system: ${system_prompt}
  retrieval:
    knowledge_base_ids: ${knowledge_base_ids}
  tools: ${tools}

metadata:
  owner: adk
  tags:
    - rag
    - knowledge-base
```

## Open Questions

- **`tool_select` UX**: Should users select from existing tools on the platform, or define new tools inline (URL, name, description, parameters)? Pending clarification.
- **Validation rules**: Parameters like agent names need constraints (e.g. `^[a-zA-Z0-9_-]+$`, max length). A `validation` field on parameters could support this.
- **Agent-to-Agent (A2A) routing**: Deferred to phase 2. Data gathering on console agent-to-agent usage is in progress to determine if this capability needs to be carried forward.




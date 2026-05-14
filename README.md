# otter-docs

A polyglot codebase inspection library for agent-driven development.

> **Status (2026-05-14): namespace reserved; public release in progress.**

## What it will be

otter-docs builds a queryable model of a codebase — modules, functions,
classes, calls, imports — augmented with LLM-generated description
embeddings, and emits structured **findings** (redundancy, drift, dead
code, architectural smells) that an agent can act on. Each finding can
carry a recommendation with rationale and an apply-ready diff.

The library is designed for agents to consume, not humans to read. The
human operates the agent.

## What's here today

This release (`0.0.0`) reserves the PyPI namespace. It contains no
functional code yet.

## What's coming in v0.1

- Polyglot AST via tree-sitter (Python, Go, TypeScript day one)
- Precise name resolution via tree-sitter-stack-graphs
- Three-vector indexing per symbol (description, code, docstring)
- Static-tier detectors wrapping `similarity-py`, `vulture`, `pydeps`, `radon`
- Embedding-augmented redundancy + drift detectors
- Agent harness (MCP-style tools, prompts, schemas, runner)
- SQLite + sqlite-vec graph backend; Neo4j adapter for heavy graphs
- Git hook + systemd timer integration

Roadmap and design notes live in the repository (link below) once
public.

## Why

AI-assisted development produces working code fast and produces
redundancy fast. otter-docs is the biopsy layer — deterministic where
it can be, LLM-augmented where it must be, agent-readable everywhere.

## License

MIT.

## Links

- Repository: <https://github.com/blong-dev/otter-docs>
- Issues: <https://github.com/blong-dev/otter-docs/issues>

## Watch this space

Follow the PyPI page or the GitHub repository for v0.1 release notes.

# AxiomMesh Lab

AxiomMesh Lab is the research and sandbox repository behind **AxiomMesh SDK**.

This repository gathers the original experiments, prototypes, standalone components, draft implementations, and conceptual files that led to the public SDK.

## What this repository is

- a **research sandbox**
- a place to inspect the **original ideas and evolution** of EDP, MEP, SAVOIR, and the drone-first direction
- a repository for **experiments, prototypes, and concept files**
- a companion archive for people who want to understand the project’s origin

## What this repository is not

- not the stable production-ready SDK
- not the canonical public package to install from PyPI
- not guaranteed to preserve backwards compatibility
- not curated for clean API usage

For the stable SDK, use **AxiomMesh SDK**:
- GitHub: https://github.com/ONE-ORIGINE/axiomesh-sdk

## Repository layout

```text
axiomesh-lab/
├── article/
├── axiom/
│   ├── analytics/
│   ├── contextualization/
│   ├── core/
│   ├── drone/
│   ├── edp/
│   ├── examples/
│   ├── mep/
│   ├── protocol/
│   └── savoir/
├── notes/
├── LICENSE_NOTICE.md
└── README.md
```

## Included components

- **EDP**: mathematical and structural environment design layer
- **MEP**: protocol and agent-side experiments, including multiple Ollama/OpenAI/Anthropic iterations
- **SAVOIR**: certainty layer for known vs estimated state
- **Drone-first experiments**: autonomous drone environment and swarm reasoning
- **Contextualizer**: raw-world to contextualized-world transformation
- **Impact Matrix**: post-causal analytics and session impact modeling
- **Article draft**: conceptual and theoretical framing of the work

## Suggested use

Use this repository to:
- inspect the project’s conceptual origin
- study alternative iterations and design directions
- compare experiments against the public SDK
- extract ideas for future AxiomMesh versions

## Publication note

This repository is intentionally presented as a **lab/sandbox repository**.
If you want a stable installation target, documentation-first experience, and publication-ready package metadata, use the main SDK repository instead.

## Quick start

This repository is not packaged as a stable Python distribution.
Treat it as source material.

You can explore files directly, for example:

```bash
git clone https://github.com/ONE-ORIGINE/axiomesh-lab.git
cd axiomesh-lab
```

## Relationship to AxiomMesh SDK

- **AxiomMesh SDK** = public, stable, package-oriented release
- **AxiomMesh Lab** = experiments, research files, prototype lineage, conceptual substrate


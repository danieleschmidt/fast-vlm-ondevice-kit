# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for FastVLM On-Device Kit.

## About ADRs

Architecture Decision Records capture important architectural decisions along with their context and consequences. They help teams understand why decisions were made and provide historical context for future changes.

## ADR Format

We use the following template for ADRs:

```markdown
# ADR-XXXX: [Brief Decision Title]

## Status
[Proposed | Accepted | Deprecated | Superseded]

## Context
[What is the issue that we're seeing that is motivating this decision or change?]

## Decision
[What is the change that we're proposing and/or doing?]

## Consequences
[What becomes easier or more difficult to do because of this change?]

## Alternatives Considered
[What other options were considered?]

## Related Documents
[Links to related ADRs, issues, or documentation]
```

## ADR Index

| Number | Title | Status | Date |
|--------|-------|--------|------|
| [ADR-0001](./0001-model-quantization-strategy.md) | Model Quantization Strategy | Accepted | 2025-01-15 |
| [ADR-0002](./0002-apple-neural-engine-optimization.md) | Apple Neural Engine Optimization | Accepted | 2025-01-15 |
| [ADR-0003](./0003-swift-api-design.md) | Swift API Design Principles | Accepted | 2025-01-15 |

## Creating New ADRs

1. Copy the template above
2. Use the next available number (ADR-XXXX)
3. Use a descriptive title in kebab-case
4. Start with status "Proposed"
5. Get team review before marking "Accepted"
6. Update this index with the new ADR

## ADR Lifecycle

- **Proposed**: Initial draft for team discussion
- **Accepted**: Decision approved and implemented
- **Deprecated**: Decision no longer applies but kept for history
- **Superseded**: Replaced by a newer ADR (link to replacement)
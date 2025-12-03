# Documentation Principles

## Core Structure

```
README.md (≤100 lines)     → "What is this" + doc links
QUICKSTART.md (≤150 lines) → "Run now" (commands only)
docs/                      → "Why, detailed" (no limit)
```

## 3 Principles

### 1. QUICKSTART = Executable Immediately
- Minimal explanation, command-focused
- "Why" explanations → move to docs/

### 2. No Duplication
- One piece of info in one place only
- Others link to it

### 3. Link Everything
- All docs include "Related Documents" table
- Backlinks required

## Content Location Guide

| Content | Location |
|---------|----------|
| How to run (simple) | QUICKSTART.md |
| How to run (detailed) | docs/guides/ |
| Theory/background | docs/guides/ or docs/reports/ |
| Paper/config comparison | docs/reports/ |
| Troubleshooting | docs/troubleshooting/ |

## Document Templates

### README.md Template
```markdown
# Project Name
One-line description.

## What is this?
2-3 sentences max.

## Quick Links
| Doc | Description |
|-----|-------------|
| [QUICKSTART](QUICKSTART.md) | Run in 5 min |
| [Experiments](docs/guides/experiments.md) | All configs |
| [Output Guide](docs/guides/output.md) | Result files |

## Installation
\`\`\`bash
# 3-5 lines max
\`\`\`

## Basic Usage
\`\`\`bash
# 3-5 lines max
\`\`\`

→ See [QUICKSTART.md](QUICKSTART.md) for more
```

### QUICKSTART.md Template
```markdown
# Quick Start

## Prerequisites
\`\`\`bash
# environment setup
\`\`\`

## Run
\`\`\`bash
# main command
\`\`\`

## All Commands
| Task | Command |
|------|---------|
| ... | ... |

→ Details: [docs/guides/...](docs/guides/...)
```

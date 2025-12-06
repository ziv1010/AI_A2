# Agentic AI Framework: Findings and Experience Report

## Overview

This document presents the findings, challenges, and insights from building an agentic AI 
predictive analytics framework with multi-layered guardrails for hallucination mitigation.

---

## Framework Architecture

### Multi-Agent System
The framework implements three specialized agents orchestrated via LangGraph:

1. **Profiler Agent**: Discovers and analyzes datasets, identifying data quality issues
2. **Modeler Agent**: Builds and evaluates ML models (LogisticRegression, RandomForest, GradientBoosting)
3. **Action Agent**: Generates insights and business recommendations

### 5-Layer Guardrail Pipeline
Based on the Medium article recommendations, we implemented defense-in-depth:

| Layer | Guardrail | Purpose |
|-------|-----------|---------|
| 1 | Input | Prompt injection, PII, dangerous patterns |
| 2 | Planning | Agent intent validation before execution |
| 3 | Code | AST analysis, blocked functions, imports |
| 4 | Output | Sensitive info detection, confidence scoring |
| 5 | Hallucination | Multi-strategy claim verification |

---

## Key Features from Tredence Article

### ✅ Implemented Features

1. **Autonomous Agents**: Self-directed agents that reason about data and models
2. **Tool-Based Execution**: Agents use tools for data access, code execution, artifacts
3. **Multi-Agent Orchestration**: LangGraph coordinates handoffs between agents
4. **Adaptive Learning**: Retry logic and error recovery for robustness
5. **Natural Language Interface**: Chat-based interaction for data analysis

### Additional Agentic Capabilities

- **Context Preservation**: State management across agent transitions
- **Rate Limiting**: Handles free-tier API constraints gracefully
- **Artifact Persistence**: Saves models, reports, and analysis results

---

## Guardrail Evaluation Results

### Hallucination Detection Strategies

1. **Factual Consistency**: Checks claims against tool outputs
2. **Self-Contradiction**: Detects inconsistent statements within response
3. **Numerical Validation**: Verifies metrics match source data
4. **Source Grounding**: Categorizes claims as grounded/ungrounded
5. **Confidence Calibration**: Flags overconfident language patterns

### Planning Guardrail (Layer 2)

Key innovation addressing the Medium article's recommendation for intent validation:

- **Intent Classification**: safe / risky / blocked
- **Tool Sequence Validation**: Checks for dangerous tool combinations
- **Resource Access Checking**: Prevents unauthorized file/network access
- **Risk Escalation Detection**: Monitors patterns across actions

---

## Challenges Encountered

### 1. Token Limits (Groq Free Tier)
**Challenge**: 6,000 TPM limit caused frequent rate limiting
**Solution**: 
- Concise prompts with truncated outputs
- Built-in delays between API calls
- Retry logic with exponential backoff

### 2. State Management
**Challenge**: Passing context between agents without bloat
**Solution**:
- Truncate long outputs to 1,000-3,000 characters
- Extract key information for next agent

### 3. Code Execution Safety
**Challenge**: Agents generating potentially dangerous code
**Solution**:
- Multi-layer validation (AST + regex + blocklists)
- Sandboxed execution environment
- Restricted import allowlist

### 4. Hallucination Detection
**Challenge**: Distinguishing approximate vs. fabricated claims
**Solution**:
- Multiple detection strategies with weighted scoring
- Context-aware verification against tool outputs
- Grounded vs. ungrounded claim categorization

### 5. Prompt Injection
**Challenge**: Malicious inputs attempting to override instructions
**Solution**:
- Regex patterns for known injection techniques
- Input sanitization before processing
- Role-hijacking detection

---

## Evaluation Methodology

The framework includes a comprehensive evaluation suite (`evaluation.py`) that measures:

- **Precision**: Correctly identifying harmful inputs/outputs
- **Recall**: Not missing actual threats
- **F1 Score**: Balanced measure of precision and recall
- **Accuracy**: Overall correctness

### Test Categories
- Input validation: 10 test cases (normal + adversarial)
- Code validation: 10 test cases (safe + dangerous)
- Output validation: 10 test cases (valid + leaky)
- Hallucination detection: 5 test cases (grounded + fabricated)
- Planning validation: 5 test cases (safe + blocked intents)

---

## Lessons Learned

### What Worked Well

1. **Multi-agent architecture** enabled specialization and cleaner code
2. **5-layer guardrails** caught issues at different stages
3. **LangGraph orchestration** made state transitions explicit
4. **Robust retry logic** handled API variability gracefully

### Areas for Improvement

1. **Semantic similarity** could use embeddings for better grounding
2. **RAG integration** would improve factual accuracy
3. **Multi-turn memory** could enhance conversation quality
4. **Streaming responses** would improve UX for long operations

---

## Conclusion

This framework demonstrates that agentic AI systems can be built with:
- **Autonomous reasoning** capabilities for predictive analytics
- **Robust safety measures** through multi-layered guardrails
- **Practical constraints** handled (free-tier limits, token budgets)

The multi-layered guardrail approach significantly reduces hallucination risk while 
maintaining the flexibility needed for autonomous agent operations.

---

## How to Run

```bash
# Full pipeline
python main.py --mode pipeline

# Interactive mode
python main.py --mode interactive

# Evaluation
python evaluation.py --run-all

# Demo
python main.py --mode demo
```

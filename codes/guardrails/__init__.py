"""
Guardrails module for the Agentic AI Framework.
Implements multi-layered validation to reduce hallucinations and mitigate risks.

Pipeline Layers:
1. Input Guardrail - Validates user inputs for safety
2. Planning Guardrail - Validates agent intent before execution
3. Code Guardrail - Checks generated code before running
4. Output Guardrail - Validates LLM responses
5. Hallucination Guardrail - Detects fabricated claims
"""

from .input_guardrail import InputGuardrail, validate_input
from .output_guardrail import OutputGuardrail, validate_output
from .code_guardrail import CodeGuardrail, validate_code
from .hallucination_guardrail import HallucinationGuardrail, check_hallucination
from .planning_guardrail import PlanningGuardrail, validate_plan
from .pipeline import GuardrailPipeline

__all__ = [
    "InputGuardrail",
    "PlanningGuardrail",
    "OutputGuardrail",
    "CodeGuardrail",
    "HallucinationGuardrail",
    "GuardrailPipeline",
    "validate_input",
    "validate_plan",
    "validate_output",
    "validate_code",
    "check_hallucination",
]

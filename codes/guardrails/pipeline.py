"""
Guardrail Pipeline for the Agentic AI Framework.
Orchestrates all guardrails in a multi-layered pipeline.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .input_guardrail import InputGuardrail, InputValidationResult
from .planning_guardrail import PlanningGuardrail, PlanValidationResult
from .output_guardrail import OutputGuardrail, OutputValidationResult
from .code_guardrail import CodeGuardrail, CodeValidationResult
from .hallucination_guardrail import HallucinationGuardrail, HallucinationCheckResult


@dataclass
class GuardrailPipelineResult:
    """Complete result from the guardrail pipeline."""
    passed: bool
    overall_score: float  # 0.0 to 1.0 (higher = safer)
    input_result: Optional[InputValidationResult] = None
    planning_result: Optional[PlanValidationResult] = None
    output_result: Optional[OutputValidationResult] = None
    code_result: Optional[CodeValidationResult] = None
    hallucination_result: Optional[HallucinationCheckResult] = None
    all_issues: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class GuardrailPipeline:
    """
    Multi-layered guardrail pipeline that orchestrates all safety checks.

    Pipeline Flow (5 Layers):
    1. Input Validation → Sanitize and validate user input
    2. Planning Validation → Check agent intent before execution
    3. Code Validation → Check generated code before execution
    4. Output Validation → Validate LLM responses
    5. Hallucination Check → Verify claims against evidence

    Each layer can pass, warn, or block based on severity.
    """

    def __init__(
        self,
        enable_input: bool = True,
        enable_planning: bool = True,
        enable_code: bool = True,
        enable_output: bool = True,
        enable_hallucination: bool = True,
        strict_mode: bool = False
    ):
        """
        Initialize the guardrail pipeline.

        Args:
            enable_input: Enable input validation.
            enable_planning: Enable planning/intent validation.
            enable_code: Enable code validation.
            enable_output: Enable output validation.
            enable_hallucination: Enable hallucination detection.
            strict_mode: If True, any warning causes failure.
        """
        self.enable_input = enable_input
        self.enable_planning = enable_planning
        self.enable_code = enable_code
        self.enable_output = enable_output
        self.enable_hallucination = enable_hallucination
        self.strict_mode = strict_mode

        # Initialize guardrails
        self.input_guardrail = InputGuardrail()
        self.planning_guardrail = PlanningGuardrail()
        self.code_guardrail = CodeGuardrail()
        self.output_guardrail = OutputGuardrail()

    def validate_input(self, input_text: str) -> Tuple[bool, str, InputValidationResult]:
        """
        Run input validation layer.

        Returns:
            Tuple of (passed, sanitized_input, result)
        """
        if not self.enable_input:
            return True, input_text, None

        result = self.input_guardrail.validate(input_text)
        passed = result.is_valid if not self.strict_mode else len(result.issues) == 0
        return passed, result.sanitized_input, result

    def validate_plan(
        self,
        plan_text: str,
        proposed_tools: Optional[List[str]] = None,
        context: Optional[Dict] = None
    ) -> Tuple[bool, PlanValidationResult]:
        """
        Run planning validation layer (Layer 2).
        Validates agent intent before tool execution.

        Returns:
            Tuple of (passed, result)
        """
        if not self.enable_planning:
            return True, None

        result = self.planning_guardrail.validate_plan(plan_text, proposed_tools, context)
        passed = result.is_valid if not self.strict_mode else result.intent_classification == "safe"
        return passed, result

    def validate_code(self, code: str) -> Tuple[bool, CodeValidationResult]:
        """
        Run code validation layer.

        Returns:
            Tuple of (passed, result)
        """
        if not self.enable_code:
            return True, None

        result = self.code_guardrail.validate(code)
        passed = result.is_valid if not self.strict_mode else result.risk_level == "safe"
        return passed, result

    def validate_output(
        self,
        output: str,
        context: Optional[Dict] = None
    ) -> Tuple[bool, str, OutputValidationResult]:
        """
        Run output validation layer.

        Returns:
            Tuple of (passed, sanitized_output, result)
        """
        if not self.enable_output:
            return True, output, None

        result = self.output_guardrail.validate(output, context)
        passed = result.is_valid if not self.strict_mode else result.confidence_score >= 0.8
        return passed, result.sanitized_output, result

    def check_hallucination(
        self,
        output: str,
        tool_outputs: Optional[List[str]] = None,
        context: Optional[Dict] = None
    ) -> Tuple[bool, HallucinationCheckResult]:
        """
        Run hallucination detection layer.

        Returns:
            Tuple of (passed, result)
        """
        if not self.enable_hallucination:
            return True, None

        guardrail = HallucinationGuardrail(context)
        result = guardrail.check(output, tool_outputs)

        if self.strict_mode:
            passed = result.recommendation == "accept"
        else:
            passed = result.recommendation != "reject"

        return passed, result

    def run_full_pipeline(
        self,
        input_text: Optional[str] = None,
        code: Optional[str] = None,
        output: Optional[str] = None,
        tool_outputs: Optional[List[str]] = None,
        context: Optional[Dict] = None
    ) -> GuardrailPipelineResult:
        """
        Run the complete guardrail pipeline.

        Args:
            input_text: User input to validate.
            code: Generated code to validate.
            output: LLM output to validate.
            tool_outputs: Tool outputs for hallucination checking.
            context: Context for consistency checking.

        Returns:
            GuardrailPipelineResult with all check results.
        """
        all_issues = []
        scores = []
        results = {}

        # Layer 1: Input Validation
        if input_text:
            passed, sanitized, result = self.validate_input(input_text)
            results["input"] = result
            if result:
                all_issues.extend(result.issues)
                scores.append(1.0 if passed else 0.5 if result.risk_level == "medium" else 0.0)

        # Layer 2: Code Validation
        if code:
            passed, result = self.validate_code(code)
            results["code"] = result
            if result:
                all_issues.extend(result.issues)
                risk_scores = {"safe": 1.0, "caution": 0.6, "dangerous": 0.0}
                scores.append(risk_scores.get(result.risk_level, 0.5))

        # Layer 3: Output Validation
        if output:
            passed, sanitized, result = self.validate_output(output, context)
            results["output"] = result
            if result:
                all_issues.extend(result.issues)
                scores.append(result.confidence_score)

        # Layer 4: Hallucination Check
        if output:
            passed, result = self.check_hallucination(output, tool_outputs, context)
            results["hallucination"] = result
            if result:
                all_issues.extend(result.issues)
                scores.append(1.0 - result.hallucination_score)

        # Calculate overall score
        overall_score = sum(scores) / len(scores) if scores else 1.0

        # Determine if pipeline passed
        pipeline_passed = overall_score >= (0.8 if self.strict_mode else 0.5)

        return GuardrailPipelineResult(
            passed=pipeline_passed,
            overall_score=overall_score,
            input_result=results.get("input"),
            output_result=results.get("output"),
            code_result=results.get("code"),
            hallucination_result=results.get("hallucination"),
            all_issues=all_issues,
            metadata={
                "strict_mode": self.strict_mode,
                "layers_run": list(results.keys()),
            }
        )

    def get_summary(self, result: GuardrailPipelineResult) -> str:
        """Generate a human-readable summary of guardrail results."""
        lines = [
            "=" * 50,
            "GUARDRAIL PIPELINE SUMMARY",
            "=" * 50,
            f"Status: {'PASSED' if result.passed else 'FAILED'}",
            f"Overall Safety Score: {result.overall_score:.2f}",
            f"Timestamp: {result.timestamp}",
            "",
        ]

        if result.input_result:
            lines.append(f"Input Validation: {result.input_result.risk_level.upper()}")

        if result.code_result:
            lines.append(f"Code Validation: {result.code_result.risk_level.upper()}")
            if result.code_result.blocked_operations:
                lines.append(f"  Blocked: {result.code_result.blocked_operations}")

        if result.output_result:
            lines.append(f"Output Validation: Confidence {result.output_result.confidence_score:.2f}")

        if result.hallucination_result:
            lines.append(f"Hallucination Score: {result.hallucination_result.hallucination_score:.2f}")
            lines.append(f"Recommendation: {result.hallucination_result.recommendation.upper()}")

        if result.all_issues:
            lines.append("")
            lines.append("Issues Found:")
            for issue in result.all_issues[:10]:  # Limit to 10
                lines.append(f"  - {issue}")

        lines.append("=" * 50)
        return "\n".join(lines)

"""
Hallucination Guardrail for the Agentic AI Framework.
Detects and mitigates LLM hallucinations using multiple strategies.
"""

import re
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class HallucinationCheckResult:
    """Result of hallucination check."""
    hallucination_score: float  # 0.0 (none) to 1.0 (definite hallucination)
    issues: List[str]
    grounded_claims: List[str]
    ungrounded_claims: List[str]
    recommendation: str  # 'accept', 'verify', 'reject'


class HallucinationGuardrail:
    """
    Multi-strategy hallucination detection guardrail.

    Strategies:
    1. Factual consistency checking
    2. Self-contradiction detection
    3. Numerical consistency validation
    4. Source grounding verification
    5. Confidence calibration
    """

    # Phrases that often precede hallucinations
    HALLUCINATION_INDICATORS = [
        "it is well known that",
        "everyone knows",
        "obviously",
        "clearly",
        "definitely",
        "without a doubt",
        "100%",
        "always",
        "never",
        "all studies show",
        "research proves",
    ]

    # Phrases indicating uncertainty (good sign)
    GROUNDED_INDICATORS = [
        "based on the data",
        "the results show",
        "according to",
        "the model predicts",
        "with accuracy of",
        "the metrics indicate",
        "from the analysis",
    ]

    def __init__(self, context: Optional[Dict] = None):
        """
        Initialize hallucination guardrail.

        Args:
            context: Dictionary with ground truth information for verification.
        """
        self.context = context or {}

    def check(
        self,
        output: str,
        tool_outputs: Optional[List[str]] = None
    ) -> HallucinationCheckResult:
        """
        Check output for potential hallucinations.

        Args:
            output: LLM output to check.
            tool_outputs: List of actual tool outputs to verify against.

        Returns:
            HallucinationCheckResult with detailed analysis.
        """
        issues = []
        grounded = []
        ungrounded = []
        score = 0.0

        # Strategy 1: Check for hallucination indicators
        indicator_score = self._check_indicators(output)
        score += indicator_score * 0.2

        # Strategy 2: Self-contradiction check
        contradictions = self._find_contradictions(output)
        if contradictions:
            issues.extend(contradictions)
            score += 0.2 * len(contradictions)

        # Strategy 3: Numerical consistency
        if tool_outputs:
            num_issues = self._check_numerical_consistency(output, tool_outputs)
            issues.extend(num_issues)
            score += 0.15 * len(num_issues)

        # Strategy 4: Claim grounding
        grounded, ungrounded = self._analyze_claims(output, tool_outputs)
        if ungrounded:
            issues.append(f"Found {len(ungrounded)} potentially ungrounded claims")
            score += 0.1 * min(len(ungrounded), 3)

        # Strategy 5: Context verification
        if self.context:
            context_issues = self._verify_against_context(output)
            issues.extend(context_issues)
            score += 0.15 * len(context_issues)

        # Clamp score
        score = min(1.0, score)

        # Determine recommendation
        if score < 0.2:
            recommendation = "accept"
        elif score < 0.5:
            recommendation = "verify"
        else:
            recommendation = "reject"

        return HallucinationCheckResult(
            hallucination_score=score,
            issues=issues,
            grounded_claims=grounded,
            ungrounded_claims=ungrounded,
            recommendation=recommendation
        )

    def _check_indicators(self, text: str) -> float:
        """Check for phrases that often indicate hallucinations."""
        text_lower = text.lower()
        count = sum(
            1 for phrase in self.HALLUCINATION_INDICATORS
            if phrase in text_lower
        )
        grounded_count = sum(
            1 for phrase in self.GROUNDED_INDICATORS
            if phrase in text_lower
        )
        # Net score: more hallucination indicators = higher score
        return max(0, (count - grounded_count) / 5)

    def _find_contradictions(self, text: str) -> List[str]:
        """Detect self-contradictions in the text."""
        contradictions = []

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        # Look for opposing claims
        opposites = [
            (r"accuracy\s+of\s+(\d+)", r"accuracy\s+of\s+(\d+)"),
            (r"(\d+)\s+features", r"(\d+)\s+features"),
            (r"best\s+model\s+is\s+(\w+)", r"best\s+model\s+is\s+(\w+)"),
        ]

        for pattern1, pattern2 in opposites:
            matches = re.findall(pattern1, text, re.IGNORECASE)
            if len(set(matches)) > 1:
                contradictions.append(f"Inconsistent values found for pattern: {pattern1}")

        return contradictions

    def _check_numerical_consistency(
        self,
        output: str,
        tool_outputs: List[str]
    ) -> List[str]:
        """Verify numbers in output match tool outputs."""
        issues = []

        # Extract numbers from output
        output_numbers = set(re.findall(r'\b\d+\.?\d*\b', output))

        # Extract numbers from tool outputs
        tool_numbers = set()
        for tool_output in tool_outputs:
            tool_numbers.update(re.findall(r'\b\d+\.?\d*\b', str(tool_output)))

        # Check for fabricated numbers (numbers in output not in tool outputs)
        # Only flag significant numbers (likely metrics, not counts)
        for num in output_numbers:
            try:
                val = float(num)
                # Only check numbers that look like metrics (0.x or xx.x%)
                if 0 < val < 1 or (0 < val < 100 and '.' in num):
                    if num not in tool_numbers:
                        # Check if a close number exists
                        close_match = any(
                            abs(float(tn) - val) < 0.01
                            for tn in tool_numbers
                            if self._is_float(tn)
                        )
                        if not close_match:
                            issues.append(f"Number {num} not found in tool outputs")
            except ValueError:
                pass

        return issues[:3]  # Limit issues

    def _is_float(self, s: str) -> bool:
        """Check if string is a valid float."""
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _analyze_claims(
        self,
        output: str,
        tool_outputs: Optional[List[str]]
    ) -> Tuple[List[str], List[str]]:
        """Analyze claims and categorize as grounded or ungrounded."""
        grounded = []
        ungrounded = []

        # Simple claim extraction (sentences with assertions)
        claim_patterns = [
            r"the\s+(\w+)\s+is\s+(\d+\.?\d*)",
            r"(\w+)\s+achieved\s+(\d+\.?\d*)",
            r"accuracy\s+(?:of|is)\s+(\d+\.?\d*)",
            r"f1[- ]?score\s+(?:of|is)\s+(\d+\.?\d*)",
        ]

        for pattern in claim_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            for match in matches:
                claim = f"{pattern}: {match}"
                # Check if claim appears in tool outputs
                if tool_outputs:
                    found = any(
                        str(m) in str(to) if isinstance(m, str) else
                        str(m[-1]) in str(to)  # Check numeric part
                        for to in tool_outputs
                        for m in [match]
                    )
                    if found:
                        grounded.append(claim)
                    else:
                        ungrounded.append(claim)

        return grounded, ungrounded

    def _verify_against_context(self, output: str) -> List[str]:
        """Verify output against known context."""
        issues = []

        # Check dataset name
        if "dataset_name" in self.context:
            expected = self.context["dataset_name"]
            # If output mentions a different dataset
            dataset_mentions = re.findall(r"dataset:\s*(\S+)", output, re.IGNORECASE)
            for mention in dataset_mentions:
                if expected.lower() not in mention.lower():
                    issues.append(f"Dataset mismatch: expected {expected}, found {mention}")

        # Check target column
        if "target_column" in self.context:
            expected = self.context["target_column"]
            if "target" in output.lower() and expected.lower() not in output.lower():
                # Only flag if a different target is mentioned
                target_mentions = re.findall(r"target[:\s]+(\w+)", output, re.IGNORECASE)
                for mention in target_mentions:
                    if mention.lower() != expected.lower():
                        issues.append(f"Target column mismatch: expected {expected}")

        return issues


def check_hallucination(
    output: str,
    tool_outputs: Optional[List[str]] = None,
    context: Optional[Dict] = None
) -> Tuple[float, List[str], str]:
    """
    Convenience function to check for hallucinations.

    Args:
        output: LLM output to check.
        tool_outputs: Optional list of tool outputs.
        context: Optional context dictionary.

    Returns:
        Tuple of (hallucination_score, issues, recommendation)
    """
    guardrail = HallucinationGuardrail(context)
    result = guardrail.check(output, tool_outputs)
    return result.hallucination_score, result.issues, result.recommendation

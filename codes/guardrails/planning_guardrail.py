"""
Planning/Reasoning Guardrail for the Agentic AI Framework.
Validates agent intent and action plans before execution.

This implements Layer 2 of the multi-layered guardrail pipeline as described
in the Medium article on reducing hallucinations and mitigating risk.
"""

import re
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class PlanValidationResult:
    """Result of planning validation."""
    is_valid: bool
    intent_classification: str  # 'safe', 'risky', 'blocked'
    issues: List[str]
    risk_score: float  # 0.0 (safe) to 1.0 (dangerous)
    recommended_actions: List[str]
    blocked_actions: List[str]


class PlanningGuardrail:
    """
    Planning/Reasoning guardrail that validates agent intent before execution.
    
    This guardrail implements a key concept from the Medium article:
    "Layer 2: Planning/Reasoning Guardrails - Validates the AI agent's internal 
    action plan to block risky or non-compliant intentions before execution."
    
    Checks:
    1. Intent classification (data analysis, model building, dangerous ops)
    2. Action sequence validation
    3. Resource access patterns
    4. Multi-step risk detection
    5. Goal alignment verification
    """
    
    # Safe intent patterns
    SAFE_INTENTS = [
        r"analyz\w*\s+(data|dataset|column|feature)",
        r"load\s+(csv|data|file)",
        r"train\s+(model|classifier|regressor)",
        r"evaluat\w*\s+(model|performance|metrics)",
        r"predict\s+(churn|outcome|class)",
        r"calculate\s+(accuracy|precision|recall|f1)",
        r"visualiz\w*\s+(data|distribution|correlation)",
        r"save\s+(model|artifact|report)",
        r"preprocess\s+(data|features)",
    ]
    
    # Risky intent patterns (require caution)
    RISKY_INTENTS = [
        r"delet\w*\s+(file|data|record)",
        r"modif\w*\s+(original|source)",
        r"overwrite",
        r"drop\s+all",
        r"truncate",
        r"external\s+(api|service|url)",
        r"download\s+from",
        r"send\s+(email|message|request)",
    ]
    
    # Blocked intent patterns (will not execute)
    BLOCKED_INTENTS = [
        r"execute\s+shell",
        r"system\s+command",
        r"access\s+(password|credential|secret)",
        r"bypass\s+security",
        r"ignore\s+guardrail",
        r"disable\s+validation",
        r"root\s+access",
        r"sudo",
        r"admin\s+privilege",
    ]
    
    # Valid tool sequences
    VALID_TOOL_SEQUENCES = {
        "data_analysis": ["list_data_files", "get_dataframe_info", "run_python"],
        "model_building": ["run_python", "save_artifact"],
        "prediction": ["run_python", "save_artifact"],
        "reporting": ["read_artifact_text", "save_artifact"],
    }
    
    def __init__(self):
        """Initialize planning guardrail."""
        self.action_history = []
    
    def validate_plan(
        self,
        plan_text: str,
        proposed_tools: Optional[List[str]] = None,
        context: Optional[Dict] = None
    ) -> PlanValidationResult:
        """
        Validate an agent's plan before execution.
        
        Args:
            plan_text: Natural language description of the plan.
            proposed_tools: List of tools the agent intends to use.
            context: Additional context about the current state.
            
        Returns:
            PlanValidationResult with detailed analysis.
        """
        issues = []
        blocked = []
        recommended = []
        risk_score = 0.0
        
        # Step 1: Intent Classification
        intent, intent_issues = self._classify_intent(plan_text)
        issues.extend(intent_issues)
        
        if intent == "blocked":
            risk_score = 1.0
            blocked.append("Plan contains blocked operations")
        elif intent == "risky":
            risk_score = 0.5
            recommended.append("Review plan carefully before proceeding")
        
        # Step 2: Tool Sequence Validation
        if proposed_tools:
            seq_issues, seq_blocked = self._validate_tool_sequence(proposed_tools)
            issues.extend(seq_issues)
            blocked.extend(seq_blocked)
            if seq_blocked:
                risk_score = min(1.0, risk_score + 0.3)
        
        # Step 3: Resource Access Check
        resource_issues = self._check_resource_access(plan_text, context)
        issues.extend(resource_issues)
        risk_score = min(1.0, risk_score + 0.1 * len(resource_issues))
        
        # Step 4: Multi-step Risk Detection
        if len(self.action_history) > 0:
            escalation_issues = self._detect_risk_escalation(plan_text)
            issues.extend(escalation_issues)
            if escalation_issues:
                risk_score = min(1.0, risk_score + 0.2)
        
        # Step 5: Goal Alignment
        alignment_issues = self._check_goal_alignment(plan_text, context)
        issues.extend(alignment_issues)
        
        # Determine final classification
        if blocked or risk_score >= 0.8:
            intent_classification = "blocked"
        elif risk_score >= 0.4:
            intent_classification = "risky"
        else:
            intent_classification = "safe"
        
        is_valid = intent_classification != "blocked"
        
        # Track this plan
        self.action_history.append({
            "plan": plan_text[:200],
            "classification": intent_classification,
            "risk_score": risk_score
        })
        
        return PlanValidationResult(
            is_valid=is_valid,
            intent_classification=intent_classification,
            issues=issues,
            risk_score=risk_score,
            recommended_actions=recommended,
            blocked_actions=blocked
        )
    
    def _classify_intent(self, plan_text: str) -> Tuple[str, List[str]]:
        """Classify the intent of the plan."""
        text_lower = plan_text.lower()
        issues = []
        
        # Check for blocked intents first
        for pattern in self.BLOCKED_INTENTS:
            if re.search(pattern, text_lower):
                issues.append(f"Blocked intent detected: {pattern}")
                return "blocked", issues
        
        # Check for risky intents
        risky_count = 0
        for pattern in self.RISKY_INTENTS:
            if re.search(pattern, text_lower):
                risky_count += 1
                issues.append(f"Risky intent: {pattern}")
        
        if risky_count > 0:
            return "risky", issues
        
        # Check for safe intents
        safe_count = sum(
            1 for pattern in self.SAFE_INTENTS
            if re.search(pattern, text_lower)
        )
        
        if safe_count > 0:
            return "safe", issues
        
        # Unknown intent - treat as risky
        issues.append("Intent could not be clearly classified")
        return "risky", issues
    
    def _validate_tool_sequence(
        self,
        tools: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Validate the proposed tool sequence."""
        issues = []
        blocked = []
        
        # Check for dangerous tool combinations
        dangerous_combos = [
            ({"run_python", "save_artifact"}, "external"),  # code + save after external access
        ]
        
        tool_set = set(tools)
        
        # Check if tools are in valid sequence categories
        valid_sequence = False
        for category, valid_tools in self.VALID_TOOL_SEQUENCES.items():
            if all(t in valid_tools for t in tools):
                valid_sequence = True
                break
        
        if not valid_sequence and len(tools) > 2:
            issues.append("Tool sequence does not match known safe patterns")
        
        return issues, blocked
    
    def _check_resource_access(
        self,
        plan_text: str,
        context: Optional[Dict]
    ) -> List[str]:
        """Check for unauthorized resource access."""
        issues = []
        text_lower = plan_text.lower()
        
        # Check for file system access outside allowed directories
        path_patterns = [
            r"\/etc\/",
            r"\/root\/",
            r"\/home\/[^\/]+\/\.",  # Hidden files in home
            r"\.\.\/",  # Parent directory traversal
            r"~\/\.",  # Hidden files in home via tilde
        ]
        
        for pattern in path_patterns:
            if re.search(pattern, plan_text):
                issues.append(f"Suspicious path access pattern: {pattern}")
        
        # Check for network access
        network_patterns = [
            r"http[s]?:\/\/",
            r"ftp:\/\/",
            r"api\.([\w]+)\.(com|io|org)",
        ]
        
        for pattern in network_patterns:
            if re.search(pattern, text_lower):
                issues.append("External network access detected in plan")
                break
        
        return issues
    
    def _detect_risk_escalation(self, plan_text: str) -> List[str]:
        """Detect if current plan escalates risk from previous actions."""
        issues = []
        
        # Check if we're seeing a pattern of increasing risk
        if len(self.action_history) >= 2:
            recent_scores = [h["risk_score"] for h in self.action_history[-3:]]
            if all(s > 0 for s in recent_scores):
                if len(set(recent_scores)) == len(recent_scores) and recent_scores == sorted(recent_scores):
                    issues.append("Detected escalating risk pattern across actions")
        
        return issues
    
    def _check_goal_alignment(
        self,
        plan_text: str,
        context: Optional[Dict]
    ) -> List[str]:
        """Check if plan aligns with stated goals."""
        issues = []
        text_lower = plan_text.lower()
        
        # Expected goals for predictive analytics
        expected_keywords = [
            "churn", "predict", "model", "analysis", "data",
            "feature", "train", "evaluate", "customer"
        ]
        
        keyword_count = sum(1 for kw in expected_keywords if kw in text_lower)
        
        if keyword_count == 0 and len(plan_text) > 50:
            issues.append("Plan may not align with predictive analytics goals")
        
        return issues
    
    def reset_history(self):
        """Reset action history."""
        self.action_history = []


def validate_plan(
    plan_text: str,
    proposed_tools: Optional[List[str]] = None,
    context: Optional[Dict] = None
) -> Tuple[bool, str, float]:
    """
    Convenience function to validate a plan.
    
    Args:
        plan_text: Plan description to validate.
        proposed_tools: Optional list of proposed tools.
        context: Optional context dictionary.
        
    Returns:
        Tuple of (is_valid, intent_classification, risk_score)
    """
    guardrail = PlanningGuardrail()
    result = guardrail.validate_plan(plan_text, proposed_tools, context)
    return result.is_valid, result.intent_classification, result.risk_score

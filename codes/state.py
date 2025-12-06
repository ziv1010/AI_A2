"""
State Management for the Agentic AI Framework.
Defines the graph state and state transitions for LangGraph.
"""

from typing import TypedDict, Annotated, List, Optional, Dict, Any, Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Main state object that flows through the LangGraph pipeline.

    Attributes:
        messages: Conversation history with the LLM (managed by add_messages)
        current_phase: Current pipeline phase (profiling, modeling, action, evaluation)
        dataset_info: Information about the loaded dataset
        target_column: The column to predict
        feature_columns: List of feature column names
        model_results: Results from model training and evaluation
        recommendations: Generated recommendations/insights
        guardrail_status: Status of guardrail checks
        error_log: Any errors encountered
        iteration_count: Number of iterations in current phase
    """
    # Core message state (accumulates messages)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Pipeline state
    current_phase: str
    task_description: str

    # Data state
    dataset_name: Optional[str]
    dataset_info: Optional[Dict[str, Any]]
    target_column: Optional[str]
    feature_columns: Optional[List[str]]

    # Model state
    model_results: Optional[Dict[str, Any]]
    best_model_name: Optional[str]
    best_model_score: Optional[float]

    # Output state
    recommendations: Optional[List[str]]
    final_report: Optional[str]

    # Control state
    guardrail_status: Dict[str, Any]
    error_log: List[str]
    iteration_count: int
    max_iterations: int


def create_initial_state(task: str = "") -> AgentState:
    """Create an initial state for a new pipeline run."""
    return AgentState(
        messages=[],
        current_phase="init",
        task_description=task,
        dataset_name=None,
        dataset_info=None,
        target_column=None,
        feature_columns=None,
        model_results=None,
        best_model_name=None,
        best_model_score=None,
        recommendations=None,
        final_report=None,
        guardrail_status={"passed": True, "checks": []},
        error_log=[],
        iteration_count=0,
        max_iterations=10,
    )


class ProfilerState(TypedDict):
    """State for the Profiler Agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    dataset_name: Optional[str]
    dataset_info: Optional[Dict[str, Any]]
    target_column: Optional[str]
    feature_columns: Optional[List[str]]
    profiling_complete: bool


class ModelerState(TypedDict):
    """State for the Modeler Agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    dataset_name: str
    target_column: str
    feature_columns: List[str]
    model_results: Optional[Dict[str, Any]]
    best_model_name: Optional[str]
    best_model_score: Optional[float]
    modeling_complete: bool


class ActionState(TypedDict):
    """State for the Action/Recommendation Agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    model_results: Dict[str, Any]
    dataset_info: Dict[str, Any]
    recommendations: Optional[List[str]]
    final_report: Optional[str]
    action_complete: bool


class GuardrailState(TypedDict):
    """State for Guardrail evaluation."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    input_to_check: str
    output_to_check: str
    check_type: str  # 'input', 'output', 'code', 'hallucination'
    passed: bool
    issues: List[str]
    confidence_score: float


# State transition helpers
def update_phase(state: AgentState, new_phase: str) -> AgentState:
    """Update the current phase of the pipeline."""
    return {**state, "current_phase": new_phase}


def add_error(state: AgentState, error: str) -> AgentState:
    """Add an error to the error log."""
    new_log = state.get("error_log", []) + [error]
    return {**state, "error_log": new_log}


def increment_iteration(state: AgentState) -> AgentState:
    """Increment the iteration counter."""
    return {**state, "iteration_count": state.get("iteration_count", 0) + 1}


def check_iteration_limit(state: AgentState) -> bool:
    """Check if we've exceeded the maximum iterations."""
    return state.get("iteration_count", 0) >= state.get("max_iterations", 10)

"""
Action Agent for the Agentic AI Framework.
Responsible for generating insights, recommendations, and prescriptive actions.
"""

from typing import List, Dict, Any
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.code_executor import execute_python_code
from tools.artifact_tools import save_artifact, list_artifacts, read_artifact_text


# Token-efficient prompt
ACTION_PROMPT = """You are an Action/Insights Agent. Your job is to generate business recommendations based on ML model results.

CONTEXT:
Dataset: {dataset_name}
Target: {target_column}
Best Model: {best_model_name} (Score: {best_model_score:.4f})

MODEL RESULTS:
{model_results}

WORKFLOW:
1. Analyze model results and feature importance
2. Identify key drivers of the prediction (e.g., what causes churn)
3. Segment customers/entities based on risk scores
4. Generate actionable recommendations

TASKS using execute_python_code:
1. Load the saved model and make predictions on full dataset
2. Create risk segments (High/Medium/Low)
3. Analyze characteristics of each segment
4. Generate feature importance visualization (save to artifacts)

OUTPUT REQUIREMENTS:
Provide:

## Key Findings
- Top 3-5 factors influencing {target_column}
- Statistical backing for each finding

## Risk Segmentation
- High Risk: characteristics and count
- Medium Risk: characteristics and count
- Low Risk: characteristics and count

## Recommendations
For each segment, provide:
1. Specific action to take
2. Expected impact
3. Priority level

## Model Performance Summary
- Best model and key metrics
- Confidence in predictions

RULES:
- Be specific and actionable
- Use data to back recommendations
- Keep response concise
- Save final report as artifact"""


def get_action_tools() -> List[BaseTool]:
    """Get the tools available to the action agent."""
    return [
        execute_python_code,
        save_artifact,
        list_artifacts,
        read_artifact_text,
    ]


def create_action_agent(llm):
    """
    Create an action agent using LangGraph's ReAct pattern.

    Args:
        llm: The language model to use.

    Returns:
        A compiled LangGraph agent.
    """
    tools = get_action_tools()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=ACTION_PROMPT,
    )

    return agent


def get_action_prompt_template() -> ChatPromptTemplate:
    """Get a prompt template for the action agent."""
    return ChatPromptTemplate.from_messages([
        ("system", ACTION_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ])


def format_action_prompt(
    dataset_name: str,
    target_column: str,
    best_model_name: str,
    best_model_score: float,
    model_results: Dict[str, Any]
) -> str:
    """Format the action prompt with context."""
    # Format model results as string
    results_str = ""
    if isinstance(model_results, dict):
        for model_name, metrics in model_results.items():
            results_str += f"\n{model_name}:\n"
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        results_str += f"  - {metric}: {value:.4f}\n"
                    else:
                        results_str += f"  - {metric}: {value}\n"
            else:
                results_str += f"  {metrics}\n"
    else:
        results_str = str(model_results)

    return ACTION_PROMPT.format(
        dataset_name=dataset_name,
        target_column=target_column,
        best_model_name=best_model_name,
        best_model_score=best_model_score,
        model_results=results_str
    )

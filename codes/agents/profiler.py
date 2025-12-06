"""
Profiler Agent for the Agentic AI Framework.
Responsible for data discovery, exploration, and profiling.
"""

from typing import List
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.data_tools import list_data_files, load_dataframe_head, get_dataframe_info, get_column_statistics
from tools.code_executor import execute_python_code
from tools.artifact_tools import save_artifact, list_artifacts


# Token-efficient prompt (concise but comprehensive)
PROFILER_PROMPT = """You are a Data Profiler Agent. Your job is to explore and understand datasets.

TASK: {task}

WORKFLOW:
1. Use list_data_files to discover available datasets
2. Use load_dataframe_head to see sample data
3. Use get_dataframe_info for schema and statistics
4. Use execute_python_code for deeper analysis (missing values, distributions, correlations)
5. Identify the TARGET column for prediction (usually binary like 'Exited', 'Churn', etc.)
6. Identify useful FEATURE columns (exclude IDs, names, row numbers)

OUTPUT REQUIREMENTS:
When done, provide a CONCISE summary with:
- Dataset name and shape
- Target column name
- List of feature columns (categorized as numeric/categorical)
- Key data quality issues found
- Any preprocessing needed

RULES:
- Be concise - avoid verbose explanations
- Focus on actionable insights
- Always identify the target variable
- Flag columns to drop (IDs, leakage)"""


def get_profiler_tools() -> List[BaseTool]:
    """Get the tools available to the profiler agent."""
    return [
        list_data_files,
        load_dataframe_head,
        get_dataframe_info,
        get_column_statistics,
        execute_python_code,
        save_artifact,
        list_artifacts,
    ]


def create_profiler_agent(llm):
    """
    Create a profiler agent using LangGraph's ReAct pattern.

    Args:
        llm: The language model to use.

    Returns:
        A compiled LangGraph agent.
    """
    tools = get_profiler_tools()

    # Create the agent with the system prompt
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=PROFILER_PROMPT,
    )

    return agent


def get_profiler_prompt_template() -> ChatPromptTemplate:
    """Get a prompt template for the profiler agent."""
    return ChatPromptTemplate.from_messages([
        ("system", PROFILER_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ])

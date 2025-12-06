"""
Interactive Chat Agent for the Agentic AI Framework.
Handles conversations about the dataset and makes predictions.
"""

import time
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, ARTIFACTS_DIR, RATE_LIMIT_CONFIG
from tools.data_tools import list_data_files, load_dataframe_head, get_dataframe_info, get_column_statistics
from tools.code_executor import run_python, run_code, reset_persistent_env
from tools.artifact_tools import save_artifact, list_artifacts, read_artifact_text
from tools.prediction_tools import (
    predict_single_customer,
    get_customer_risk_profile,
    get_high_risk_customers,
    analyze_churn_factors
)


# All available tools for the chat agent
CHAT_TOOLS = [
    list_data_files,
    load_dataframe_head,
    get_dataframe_info,
    get_column_statistics,
    run_python,
    list_artifacts,
    read_artifact_text,
    predict_single_customer,
    get_customer_risk_profile,
    get_high_risk_customers,
    analyze_churn_factors,
]


CHAT_SYSTEM_PROMPT = f"""You are an AI Data Analyst assistant specialized in customer churn analysis.

DATA LOCATION: {DATA_DIR}
ARTIFACTS: {ARTIFACTS_DIR}
DATASET: Customer-Churn-Records.csv (10,000 bank customers)

AVAILABLE TOOLS:
1. list_data_files - List available datasets
2. load_dataframe_head - Preview data
3. get_dataframe_info - Get schema and statistics
4. get_column_statistics - Detailed column analysis
5. run_python - Execute Python code for analysis
6. list_artifacts - See saved models/reports
7. predict_single_customer - Predict churn for new customer
8. get_customer_risk_profile - Analyze existing customer by ID
9. get_high_risk_customers - List customers most likely to churn
10. analyze_churn_factors - Understand what drives churn

GUIDELINES:
- Answer questions about the dataset using tools
- For statistics/analysis, use run_python with pandas code
- For predictions, use the prediction tools
- Be concise but informative
- If asked to predict, use predict_single_customer or get_customer_risk_profile
- Always explain your findings in plain language

When using run_python, write simple code like:
```python
df = pd.read_csv(DATA_DIR / "Customer-Churn-Records.csv")
print(df['column'].describe())
```

DATASET COLUMNS:
- CustomerId, Surname: Customer identifiers
- CreditScore: Credit rating (300-850)
- Geography: France, Spain, Germany
- Gender: Male, Female
- Age: Customer age
- Tenure: Years as customer
- Balance: Account balance
- NumOfProducts: Products owned (1-4)
- HasCrCard: Has credit card (0/1)
- IsActiveMember: Active status (0/1)
- EstimatedSalary: Annual salary
- Exited: TARGET - Churned (1) or not (0)
- Complain: Has complained (0/1)
- Satisfaction Score: 1-5
- Card Type: DIAMOND, GOLD, SILVER, PLATINUM
- Point Earned: Loyalty points"""


class ChatAgent:
    """Interactive chat agent for data analysis and predictions."""

    def __init__(self, llm, max_iterations: int = 5):
        self.llm = llm
        self.tools = CHAT_TOOLS
        self.tool_map = {t.name: t for t in self.tools}
        self.max_iterations = max_iterations
        self.conversation_history: List = []
        self.system_message = SystemMessage(content=CHAT_SYSTEM_PROMPT)

    def reset(self):
        """Reset conversation history."""
        self.conversation_history = []
        reset_persistent_env()

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return the agent's response.

        Args:
            user_message: The user's question or request

        Returns:
            The agent's response
        """
        # Add user message to history
        self.conversation_history.append(HumanMessage(content=user_message))

        # Build messages for LLM
        messages = [self.system_message] + self.conversation_history

        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(self.tools)

        iteration = 0
        final_response = ""

        while iteration < self.max_iterations:
            iteration += 1

            # Rate limiting
            time.sleep(RATE_LIMIT_CONFIG.get("retry_delay", 1.5))

            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                error_msg = str(e)
                if "tool_use_failed" in error_msg:
                    # Retry with simpler prompt
                    messages.append(HumanMessage(content="Please try a simpler approach."))
                    continue
                else:
                    final_response = f"Error: {error_msg[:200]}"
                    break

            messages.append(response)

            # Check for tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id", "")

                    if tool_name in self.tool_map:
                        try:
                            result = self.tool_map[tool_name].invoke(tool_args)
                            # Truncate if too long
                            if len(str(result)) > 4000:
                                result = str(result)[:4000] + "\n...[TRUNCATED]"
                        except Exception as e:
                            result = f"Tool error: {str(e)}"

                        messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))
                    else:
                        messages.append(ToolMessage(content=f"Unknown tool: {tool_name}", tool_call_id=tool_id))
            else:
                # No tool calls - final response
                final_response = response.content
                break

        # Add assistant response to history
        if final_response:
            self.conversation_history.append(AIMessage(content=final_response))

        return final_response if final_response else "I couldn't generate a response. Please try rephrasing your question."

    def get_quick_stats(self) -> str:
        """Get quick dataset statistics without LLM."""
        code = '''
df = pd.read_csv(DATA_DIR / "Customer-Churn-Records.csv")
print(f"Total Customers: {len(df):,}")
print(f"Churned: {df['Exited'].sum():,} ({df['Exited'].mean()*100:.1f}%)")
print(f"Active Members: {df['IsActiveMember'].sum():,}")
print(f"Avg Age: {df['Age'].mean():.1f}")
print(f"Avg Balance: ${df['Balance'].mean():,.0f}")
print(f"Countries: {', '.join(df['Geography'].unique())}")
'''
        return run_code(code)

    def predict_for_customer(self, customer_data: Dict) -> str:
        """Make prediction for customer data."""
        try:
            return predict_single_customer.invoke(customer_data)
        except Exception as e:
            return f"Prediction error: {str(e)}"


def create_chat_agent(llm) -> ChatAgent:
    """Create and return a chat agent instance."""
    return ChatAgent(llm)

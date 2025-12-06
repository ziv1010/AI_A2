"""
LangGraph Orchestration for the Agentic AI Framework.
Simplified implementation optimized for Groq's free tier.
"""

import time
import json
from typing import Dict, Any, List, Literal, Annotated, Sequence, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import AGENT_CONFIG, RATE_LIMIT_CONFIG, DATA_DIR, ARTIFACTS_DIR
from tools.data_tools import list_data_files, load_dataframe_head, get_dataframe_info, get_column_statistics
from tools.code_executor import run_python, execute_code, run_code, reset_persistent_env
from tools.artifact_tools import save_artifact, list_artifacts, read_artifact_text
from tools.prediction_tools import get_high_risk_customers, get_low_risk_customers, analyze_churn_factors, predict_single_customer
from guardrails.pipeline import GuardrailPipeline


# =============================================================================
# SIMPLIFIED TOOL SETS
# =============================================================================

PROFILER_TOOLS = [list_data_files, load_dataframe_head, get_dataframe_info, run_python]
MODELER_TOOLS = [run_python, save_artifact, list_artifacts]
ACTION_TOOLS = [run_python, save_artifact, list_artifacts, read_artifact_text, get_high_risk_customers, get_low_risk_customers, analyze_churn_factors]


# =============================================================================
# SIMPLIFIED PROMPTS - Shorter and clearer for Groq
# =============================================================================

PROFILER_SYSTEM = f"""You are a Data Profiler. Your ONLY job is to analyze and profile datasets for ML modeling.

DATA LOCATION: {DATA_DIR}
DATASET: Customer-Churn-Records.csv
TARGET COLUMN: Exited (1 = churned, 0 = stayed)

IMPORTANT: You are NOT responsible for making predictions or answering user questions about predictions.
Your job is ONLY to profile the data so the Modeler can build a model.
DO NOT try to identify specific customers who will churn - that is the Action agent's job AFTER the model is trained.

IMPORTANT COLUMN MEANINGS:
- RowNumber, CustomerId, Surname: ID columns (drop for modeling)
- CreditScore: Higher = less likely to churn
- Geography: France, Spain, Germany
- Gender: Male/Female
- Age: Older customers more loyal
- Tenure: Years as customer (higher = more loyal)
- Balance: Higher balance = less likely to churn
- NumOfProducts: Products purchased (1-4)
- HasCrCard: 1 = has credit card
- IsActiveMember: 1 = active (less likely to churn)
- EstimatedSalary: Annual salary
- Complain: 1 = has complaint (STRONG churn indicator)
- Satisfaction Score: 1-5 rating
- Card Type: DIAMOND, GOLD, SILVER, PLATINUM
- Point Earned: Loyalty points

STEPS:
1. First use list_data_files to see available files
2. Use get_dataframe_info to understand the data structure
3. Use run_python to analyze data quality and class distribution

When using run_python, write simple Python code:
```python
df = pd.read_csv(DATA_DIR / "Customer-Churn-Records.csv")
print(f"Dataset Name: Customer-Churn-Records.csv")
print(f"Shape: {{df.shape}}")
print(f"Target Column: Exited")
print(f"Churn Rate: {{df['Exited'].mean()*100:.2f}}%")
print(f"Feature Columns: {{[c for c in df.columns if c not in ['RowNumber', 'CustomerId', 'Surname', 'Exited']]}}")
```

OUTPUT: Provide dataset name, shape, target column (Exited), feature columns, and data issues.
DO NOT list specific customers or make predictions."""


MODELER_SYSTEM = f"""You are an ML Modeler. Build and evaluate models for churn prediction.

DATA LOCATION: {DATA_DIR}
ARTIFACTS: {ARTIFACTS_DIR}

YOUR TASK: Train machine learning models to predict customer churn.

STEPS:
1. Use run_python to train and evaluate models
2. Save the best model to artifacts

SIMPLE CODE TO USE with run_python tool:
- Load data from DATA_DIR / "Customer-Churn-Records.csv"
- Target column is "Exited" (1=churned, 0=stayed)
- Drop columns: RowNumber, CustomerId, Surname
- Encode categorical columns (Geography, Gender, Card Type) using LabelEncoder
- Train LogisticRegression, RandomForest, GradientBoosting
- Evaluate each model with accuracy and F1 score
- Save best model with: joblib.dump(best_model, ARTIFACTS_DIR / "best_model.joblib")

Keep your code simple and short. Avoid special characters.
Print results clearly: model name, accuracy, F1 score.
Save the best performing model to artifacts.

OUTPUT: Report model comparison and which model was saved."""


ACTION_SYSTEM = f"""You are an Insights Generator. Your PRIMARY job is to ANSWER THE USER'S SPECIFIC QUESTION using the trained model.

DATA: {DATA_DIR}
ARTIFACTS: {ARTIFACTS_DIR}

AVAILABLE TOOLS:
- get_high_risk_customers: Pass top_n as integer (e.g., 5) to get N customers most likely to churn
- get_low_risk_customers: Pass top_n as integer (e.g., 5) to get N customers least likely to churn
- analyze_churn_factors: No parameters needed. Returns feature importances and what drives churn
- run_python: Run custom Python code for analysis
- save_artifact: Save reports to artifacts folder

CRITICAL: You MUST answer the user's specific question. If they ask for "top 5 most likely to churn and 5 least likely", use the tools to get BOTH lists.

WORKFLOW:
1. FIRST understand what the user is asking for
2. Use the appropriate tools to get the specific data requested
3. ANSWER THE SPECIFIC QUESTION with actual customer data
4. Use analyze_churn_factors to explain what features drive the predictions
5. Save a comprehensive final report

EXAMPLE: If user asks "top 5 most likely and 5 least likely to churn + what factors":
1. Call get_high_risk_customers with top_n=5
2. Call get_low_risk_customers with top_n=5
3. Call analyze_churn_factors

Then SUMMARIZE the results clearly for the user:
- List each high-risk customer with ID, name, churn probability, key attributes
- List each low-risk customer with same details
- Explain the top factors driving these predictions (e.g., "Complain is the #1 factor at 45%")

OUTPUT REQUIREMENTS:
1. DIRECTLY ANSWER the user's question with SPECIFIC customer data
2. Show actual customer IDs, names, probabilities, and details
3. Explain what factors drove these predictions with actual percentages
4. Save a comprehensive final report with all details to artifacts"""


# =============================================================================
# CUSTOM AGENT WITH RETRY LOGIC
# =============================================================================

class RobustAgent:
    """Agent with retry logic for handling Groq API errors."""

    def __init__(self, llm, tools: List[BaseTool], system_prompt: str, name: str = "agent"):
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.name = name
        self.max_iterations = 8
        self.max_retries = 3

    def invoke(self, task: str) -> Dict[str, Any]:
        """Run the agent with retry logic."""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=task)
        ]

        llm_with_tools = self.llm.bind_tools(self.tools)
        tool_map = {t.name: t for t in self.tools}

        iteration = 0
        final_response = ""

        while iteration < self.max_iterations:
            iteration += 1

            # Rate limiting
            time.sleep(RATE_LIMIT_CONFIG.get("retry_delay", 2.0))

            # Call LLM with retries
            response = None
            for retry in range(self.max_retries):
                try:
                    response = llm_with_tools.invoke(messages)
                    break
                except Exception as e:
                    error_msg = str(e)
                    print(f"  Retry {retry + 1}/{self.max_retries}: {error_msg[:100]}")

                    # If it's a tool format error, simplify the prompt
                    if "tool_use_failed" in error_msg or "Failed to call" in error_msg:
                        # Add a hint about proper tool usage
                        messages.append(HumanMessage(content="Please use simpler code without special characters. Use double quotes only."))
                        time.sleep(2)
                    else:
                        time.sleep(2 ** retry)

            if response is None:
                print(f"  Failed after {self.max_retries} retries")
                break

            messages.append(response)

            # Check if we have tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id", "")

                    print(f"  Tool: {tool_name}")

                    if tool_name in tool_map:
                        try:
                            result = tool_map[tool_name].invoke(tool_args)
                            # Truncate long results
                            if len(str(result)) > 3000:
                                result = str(result)[:3000] + "\n...[TRUNCATED]"
                            print(f"  Result: {str(result)[:200]}...")
                        except Exception as e:
                            result = f"Tool error: {str(e)}"
                            print(f"  Error: {result}")

                        messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))
                    else:
                        messages.append(ToolMessage(content=f"Unknown tool: {tool_name}", tool_call_id=tool_id))
            else:
                # No tool calls - this is the final response
                final_response = response.content
                break

        return {
            "messages": messages,
            "response": final_response,
            "iterations": iteration
        }


# =============================================================================
# SIMPLE PIPELINE BUILDER
# =============================================================================

def build_simple_pipeline(llm):
    """Build simple agents without LangGraph complexity."""
    return {
        "profiler": RobustAgent(llm, PROFILER_TOOLS, PROFILER_SYSTEM, "profiler"),
        "modeler": RobustAgent(llm, MODELER_TOOLS, MODELER_SYSTEM, "modeler"),
        "action": RobustAgent(llm, ACTION_TOOLS, ACTION_SYSTEM, "action"),
    }


def build_react_agents(llm):
    """Build LangGraph ReAct agents."""
    profiler = create_react_agent(
        model=llm,
        tools=PROFILER_TOOLS,
        state_modifier=PROFILER_SYSTEM,
    )

    modeler = create_react_agent(
        model=llm,
        tools=MODELER_TOOLS,
        state_modifier=MODELER_SYSTEM,
    )

    action = create_react_agent(
        model=llm,
        tools=ACTION_TOOLS,
        state_modifier=ACTION_SYSTEM,
    )

    return {
        "profiler": profiler,
        "modeler": modeler,
        "action": action,
    }


# =============================================================================
# PIPELINE RUNNER
# =============================================================================

class AgenticPipeline:
    """High-level pipeline runner."""

    def __init__(self, llm, use_guardrails: bool = True, use_robust_agents: bool = True):
        self.llm = llm
        self.use_guardrails = use_guardrails

        if use_robust_agents:
            self.agents = build_simple_pipeline(llm)
            self.use_robust = True
        else:
            self.agents = build_react_agents(llm)
            self.use_robust = False

        self.guardrail_pipeline = GuardrailPipeline() if use_guardrails else None
        self.results = {}

    def _run_robust_agent(self, agent: RobustAgent, task: str) -> str:
        """Run a RobustAgent and return the response."""
        result = agent.invoke(task)
        return result.get("response", "")

    def _run_react_agent(self, agent, task: str, thread_id: str) -> str:
        """Run a LangGraph ReAct agent."""
        config = {"configurable": {"thread_id": thread_id}}
        result = agent.invoke(
            {"messages": [HumanMessage(content=task)]},
            config=config
        )
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content
        return ""

    def run_profiler(self, task: str = None) -> str:
        """Run the profiler agent.
        
        Note: The profiler always uses a standard profiling task, regardless of user query.
        The user's specific question is handled by the Action agent AFTER model training.
        """
        # Always use standard profiling task - user questions are handled in Phase 3
        profiler_task = "Analyze the Customer-Churn-Records.csv dataset. List files, check data structure, identify target and features. DO NOT make predictions or identify specific customers."

        print("\n" + "="*60)
        print("PHASE 1: DATA PROFILING")
        print("="*60)

        reset_persistent_env()

        if self.use_robust:
            output = self._run_robust_agent(self.agents["profiler"], profiler_task)
        else:
            output = self._run_react_agent(self.agents["profiler"], profiler_task, "profiler-1")

        self.results["profiler"] = output
        print(f"\nProfiler Output:\n{output[:1500]}...")
        return output

    def run_modeler(self, profiler_context: str) -> str:
        """Run the modeler agent with fallback to direct training."""
        print("\n" + "="*60)
        print("PHASE 2: MODEL BUILDING")
        print("="*60)

        task = f"""Based on profiler analysis:
{profiler_context[:1000]}

Build predictive models for customer churn (target: Exited column).
Train LogisticRegression, RandomForest, GradientBoosting.
Evaluate and compare models."""

        if self.use_robust:
            output = self._run_robust_agent(self.agents["modeler"], task)
        else:
            output = self._run_react_agent(self.agents["modeler"], task, "modeler-1")

        # Check if model was saved - if not, fallback to direct training
        model_path = ARTIFACTS_DIR / "best_model.joblib"
        if not model_path.exists() or not output or output.strip() == "":
            print("\n  LLM modeler failed or incomplete. Falling back to direct model training...")
            output = run_modeling_directly()
            print("  Direct training complete.")

        self.results["modeler"] = output
        print(f"\nModeler Output:\n{output[:1500]}...")
        return output

    def run_action(self, modeler_context: str, user_query: str = None) -> str:
        """Run the action agent."""
        print("\n" + "="*60)
        print("PHASE 3: INSIGHTS & RECOMMENDATIONS")
        print("="*60)

        # If user provided a specific query, include it prominently
        if user_query:
            task = f"""USER'S ORIGINAL QUESTION: {user_query}

YOU MUST ANSWER THIS SPECIFIC QUESTION WITH ACTUAL DATA FROM THE MODEL.

Context from modeling phase:
{modeler_context[:800]}

REQUIRED ACTIONS:
1. Load the trained model from artifacts
2. Run predictions on the full dataset
3. ANSWER THE USER'S QUESTION with specific customer data
4. Show the factors/features driving these predictions
5. Save a comprehensive final report to artifacts

DO NOT give generic insights. ANSWER THE SPECIFIC QUESTION."""
        else:
            task = f"""Based on modeling results:
{modeler_context[:1000]}

Generate:
1. Key factors driving churn
2. Customer risk segments
3. Specific recommendations
4. Save a final report"""

        if self.use_robust:
            output = self._run_robust_agent(self.agents["action"], task)
        else:
            output = self._run_react_agent(self.agents["action"], task, "action-1")

        self.results["action"] = output
        print(f"\nAction Output:\n{output[:1500]}...")
        return output

    def run_full_pipeline(self, task: str = None) -> Dict[str, Any]:
        """Run the complete pipeline."""
        print("\n" + "#"*60)
        print("AGENTIC AI PREDICTIVE ANALYTICS PIPELINE")
        print("#"*60)

        # Store the original user query to pass to the action agent
        user_query = task

        profiler_output = self.run_profiler(task)
        modeler_output = self.run_modeler(profiler_output)
        action_output = self.run_action(modeler_output, user_query)

        if self.use_guardrails and action_output:
            print("\n" + "="*60)
            print("GUARDRAIL EVALUATION")
            print("="*60)

            result = self.guardrail_pipeline.run_full_pipeline(
                input_text=task,
                output=action_output,
                tool_outputs=[profiler_output, modeler_output],
            )
            print(self.guardrail_pipeline.get_summary(result))
            self.results["guardrails"] = result

            # Save comprehensive guardrails report to artifacts
            self._save_guardrails_artifact(result, task)

        print("\n" + "#"*60)
        print("PIPELINE COMPLETE")
        print("#"*60)

        return self.results

    def _save_guardrails_artifact(self, result, task: str = None):
        """Save comprehensive guardrails evaluation to artifacts."""
        import json
        from datetime import datetime

        guardrails_report = {
            "timestamp": datetime.now().isoformat(),
            "user_query": task,
            "overall_status": "PASSED" if result.passed else "FAILED",
            "overall_safety_score": result.overall_score,
            "layers_evaluated": result.metadata.get("layers_run", []),
            "strict_mode": result.metadata.get("strict_mode", False),
            "input_validation": None,
            "output_validation": None,
            "hallucination_check": None,
            "all_issues": result.all_issues,
        }

        # Add input validation details
        if result.input_result:
            guardrails_report["input_validation"] = {
                "is_valid": result.input_result.is_valid,
                "risk_level": result.input_result.risk_level,
                "issues": result.input_result.issues,
            }

        # Add output validation details
        if result.output_result:
            guardrails_report["output_validation"] = {
                "is_valid": result.output_result.is_valid,
                "confidence_score": result.output_result.confidence_score,
                "issues": result.output_result.issues,
            }

        # Add hallucination check details
        if result.hallucination_result:
            guardrails_report["hallucination_check"] = {
                "hallucination_score": result.hallucination_result.hallucination_score,
                "recommendation": result.hallucination_result.recommendation,
                "grounded_claims": result.hallucination_result.grounded_claims,
                "ungrounded_claims": result.hallucination_result.ungrounded_claims,
                "issues": result.hallucination_result.issues,
            }

        # Add code validation details if present
        if result.code_result:
            guardrails_report["code_validation"] = {
                "is_valid": result.code_result.is_valid,
                "risk_level": result.code_result.risk_level,
                "issues": result.code_result.issues,
            }

        # Save to artifacts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = ARTIFACTS_DIR / f"guardrails_report_{timestamp}.json"
        with open(filepath, 'w') as f:
            json.dump(guardrails_report, f, indent=2, default=str)

        print(f"\nGuardrails report saved to: {filepath}")


# =============================================================================
# DIRECT EXECUTION FUNCTIONS (No LLM, for testing)
# =============================================================================

def run_profiling_directly() -> str:
    """Run data profiling without LLM."""
    code = '''
import pandas as pd

df = pd.read_csv(DATA_DIR / "Customer-Churn-Records.csv")
print("=== DATASET PROFILE ===")
print(f"Shape: {df.shape}")
print(f"\\nColumns: {list(df.columns)}")
print(f"\\nData Types:\\n{df.dtypes}")
print(f"\\nMissing Values:\\n{df.isnull().sum()}")
print(f"\\nTarget Distribution (Exited):\\n{df['Exited'].value_counts()}")
print(f"\\nChurn Rate: {df['Exited'].mean()*100:.2f}%")
'''
    return run_code(code)


def run_modeling_directly() -> str:
    """Run model building without LLM."""
    code = '''
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv(DATA_DIR / "Customer-Churn-Records.csv")

# Prepare features
X = df.drop(['Exited', 'RowNumber', 'CustomerId', 'Surname'], axis=1)
y = df['Exited']

# Encode categorical columns
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate models
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}

print("=== MODEL COMPARISON ===")
results = {}
best_model = None
best_f1 = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    print(f"{name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name

# Save best model
joblib.dump(best_model, ARTIFACTS_DIR / "best_model.joblib")
print(f"\\nBest Model: {best_name} (F1={best_f1:.4f})")
print(f"Model saved to {ARTIFACTS_DIR / 'best_model.joblib'}")

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"\\nTop 5 Features:\\n{importance.head()}")
'''
    return run_code(code)


def run_action_directly() -> str:
    """Generate insights without LLM."""
    code = '''
import pandas as pd
import numpy as np

# Load model and data
model = joblib.load(ARTIFACTS_DIR / "best_model.joblib")
df = pd.read_csv(DATA_DIR / "Customer-Churn-Records.csv")

# Prepare data
X = df.drop(['Exited', 'RowNumber', 'CustomerId', 'Surname'], axis=1)
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Get predictions
if hasattr(model, 'predict_proba'):
    probs = model.predict_proba(X)[:, 1]
else:
    probs = model.predict(X)

df['ChurnRisk'] = probs

# Create segments
df['Segment'] = pd.cut(df['ChurnRisk'],
                       bins=[0, 0.3, 0.6, 1.0],
                       labels=['Low Risk', 'Medium Risk', 'High Risk'])

print("=== RISK SEGMENTATION ===")
print(df['Segment'].value_counts())

print("\\n=== SEGMENT PROFILES ===")
for segment in ['High Risk', 'Medium Risk', 'Low Risk']:
    seg_data = df[df['Segment'] == segment]
    print(f"\\n{segment}:")
    print(f"  Count: {len(seg_data)}")
    print(f"  Avg Age: {seg_data['Age'].mean():.1f}")
    print(f"  Avg Balance: ${seg_data['Balance'].mean():,.0f}")
    print(f"  Active Rate: {seg_data['IsActiveMember'].mean()*100:.1f}%")
    print(f"  Complaint Rate: {seg_data['Complain'].mean()*100:.1f}%")

print("\\n=== RECOMMENDATIONS ===")
print("High Risk: Immediate intervention - personalized retention offers")
print("Medium Risk: Proactive engagement - loyalty programs")
print("Low Risk: Maintain satisfaction - regular check-ins")
'''
    return run_code(code)

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
from guardrails.pipeline import GuardrailPipeline


# =============================================================================
# SIMPLIFIED TOOL SETS
# =============================================================================

PROFILER_TOOLS = [list_data_files, load_dataframe_head, get_dataframe_info, run_python]
MODELER_TOOLS = [run_python, save_artifact, list_artifacts]
ACTION_TOOLS = [run_python, save_artifact, list_artifacts, read_artifact_text]


# =============================================================================
# SIMPLIFIED PROMPTS - Shorter and clearer for Groq
# =============================================================================

PROFILER_SYSTEM = f"""You are a Data Profiler. Analyze datasets for ML.

DATA LOCATION: {DATA_DIR}
DATASET: Customer-Churn-Records.csv

STEPS:
1. First use list_data_files to see available files
2. Use get_dataframe_info to understand the data structure
3. Use run_python to analyze data quality

When using run_python, write simple Python code:
```python
df = pd.read_csv(DATA_DIR / "Customer-Churn-Records.csv")
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
```

OUTPUT: Provide dataset name, shape, target column (Exited), feature columns, and data issues."""


MODELER_SYSTEM = f"""You are an ML Modeler. Build and evaluate models.

DATA LOCATION: {DATA_DIR}
ARTIFACTS: {ARTIFACTS_DIR}

STEPS:
1. Load and preprocess data
2. Train 3 models: LogisticRegression, RandomForest, GradientBoosting
3. Evaluate with accuracy, precision, recall, F1
4. Save best model

Use run_python with code like:
```python
df = pd.read_csv(DATA_DIR / "Customer-Churn-Records.csv")
# preprocessing
X = df.drop(['Exited', 'RowNumber', 'CustomerId', 'Surname'], axis=1)
y = df['Exited']
# encode categoricals
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col])
# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {{accuracy_score(y_test, y_pred):.4f}}")
print(f"F1: {{f1_score(y_test, y_pred):.4f}}")
```

OUTPUT: Model comparison table, best model name and metrics."""


ACTION_SYSTEM = f"""You are an Insights Generator. Create business recommendations from ML results.

ARTIFACTS: {ARTIFACTS_DIR}

Based on model results, provide:
1. Key factors driving churn
2. Risk segments (High/Medium/Low)
3. Specific recommendations per segment
4. Business impact

Use run_python to analyze feature importance and create segments.

Save final report using save_artifact tool."""


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
        """Run the profiler agent."""
        if task is None:
            task = "Analyze the Customer-Churn-Records.csv dataset. List files, check data structure, identify target and features."

        print("\n" + "="*60)
        print("PHASE 1: DATA PROFILING")
        print("="*60)

        reset_persistent_env()

        if self.use_robust:
            output = self._run_robust_agent(self.agents["profiler"], task)
        else:
            output = self._run_react_agent(self.agents["profiler"], task, "profiler-1")

        self.results["profiler"] = output
        print(f"\nProfiler Output:\n{output[:1500]}...")
        return output

    def run_modeler(self, profiler_context: str) -> str:
        """Run the modeler agent."""
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

        self.results["modeler"] = output
        print(f"\nModeler Output:\n{output[:1500]}...")
        return output

    def run_action(self, modeler_context: str) -> str:
        """Run the action agent."""
        print("\n" + "="*60)
        print("PHASE 3: INSIGHTS & RECOMMENDATIONS")
        print("="*60)

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

        profiler_output = self.run_profiler(task)
        modeler_output = self.run_modeler(profiler_output)
        action_output = self.run_action(modeler_output)

        if self.use_guardrails and action_output:
            print("\n" + "="*60)
            print("GUARDRAIL EVALUATION")
            print("="*60)

            result = self.guardrail_pipeline.run_full_pipeline(
                output=action_output,
                tool_outputs=[profiler_output, modeler_output],
            )
            print(self.guardrail_pipeline.get_summary(result))
            self.results["guardrails"] = result

        print("\n" + "#"*60)
        print("PIPELINE COMPLETE")
        print("#"*60)

        return self.results


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

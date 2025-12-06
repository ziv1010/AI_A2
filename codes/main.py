"""
Main Entry Point for the Agentic AI Predictive Analytics Framework.

This script demonstrates the complete pipeline for:
1. Data profiling and exploration
2. Model building and evaluation
3. Insights and recommendations generation
4. Guardrail evaluation

Usage:
    python main.py --api-key YOUR_GROQ_API_KEY

    Or set environment variable:
    export GROQ_API_KEY=your_key
    python main.py
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add codes directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import set_api_key, MODEL_CONFIG, RATE_LIMIT_CONFIG
from graph import AgenticPipeline, build_simple_pipeline
from guardrails.pipeline import GuardrailPipeline


def create_llm(api_key: str):
    """Create the Groq LLM instance with rate limiting."""
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        api_key=api_key,
        model=MODEL_CONFIG["model_name"],
        temperature=MODEL_CONFIG["temperature"],
        max_tokens=MODEL_CONFIG["max_tokens"],
        timeout=MODEL_CONFIG.get("timeout", 60),
    )

    return llm


def run_interactive_mode(llm):
    """Run in interactive mode where user can chat with agents."""
    from langchain_core.messages import HumanMessage
    from graph import PROFILER_TOOLS, MODELER_TOOLS, ACTION_TOOLS
    from langgraph.prebuilt import create_react_agent

    print("\n" + "="*60)
    print("INTERACTIVE AGENTIC AI MODE")
    print("="*60)
    print("Commands:")
    print("  /profile  - Run data profiling")
    print("  /model    - Run model building")
    print("  /action   - Run action/insights")
    print("  /full     - Run full pipeline")
    print("  /quit     - Exit")
    print("="*60 + "\n")

    agents = build_simple_pipeline(llm)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "/quit":
                print("Goodbye!")
                break

            elif user_input.lower() == "/profile":
                task = input("Enter profiling task (or press Enter for default): ").strip()
                if not task:
                    task = "Analyze the available dataset for predictive modeling"

                print("\nRunning profiler agent...")
                result = agents["profiler"].invoke(
                    {"messages": [HumanMessage(content=task)]},
                    config={"configurable": {"thread_id": "interactive-profile"}}
                )
                print_agent_response(result)

            elif user_input.lower() == "/model":
                task = input("Enter modeling task (or press Enter for default): ").strip()
                if not task:
                    task = "Build predictive models using the loaded dataset"

                print("\nRunning modeler agent...")
                result = agents["modeler"].invoke(
                    {"messages": [HumanMessage(content=task)]},
                    config={"configurable": {"thread_id": "interactive-model"}}
                )
                print_agent_response(result)

            elif user_input.lower() == "/action":
                task = input("Enter action task (or press Enter for default): ").strip()
                if not task:
                    task = "Generate insights and recommendations from model results"

                print("\nRunning action agent...")
                result = agents["action"].invoke(
                    {"messages": [HumanMessage(content=task)]},
                    config={"configurable": {"thread_id": "interactive-action"}}
                )
                print_agent_response(result)

            elif user_input.lower() == "/full":
                print("\nRunning full pipeline...")
                pipeline = AgenticPipeline(llm, use_guardrails=True)
                results = pipeline.run_full_pipeline()
                print("\nPipeline completed. Results saved in artifacts folder.")

            else:
                # Default: send to profiler agent
                print("\nSending to profiler agent...")
                result = agents["profiler"].invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config={"configurable": {"thread_id": "interactive-default"}}
                )
                print_agent_response(result)

            # Rate limiting delay
            time.sleep(RATE_LIMIT_CONFIG["retry_delay"])

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


def print_agent_response(result):
    """Print the agent's response from the result."""
    from langchain_core.messages import AIMessage

    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            print(f"\nAgent: {msg.content}")
            break


def run_pipeline_mode(llm, task: str = None):
    """Run the full pipeline automatically."""
    if task is None:
        task = "Analyze the customer churn dataset, build predictive models, and generate actionable recommendations"

    pipeline = AgenticPipeline(llm, use_guardrails=True)
    results = pipeline.run_full_pipeline(task)

    return results


def run_demo_mode(llm):
    """Run a demonstration of all components."""
    print("\n" + "="*60)
    print("DEMONSTRATION MODE")
    print("="*60)

    # Demo 1: Tool testing
    print("\n--- Testing Tools ---")
    from tools.data_tools import list_data_files, get_dataframe_info
    from tools.code_executor import execute_code

    print("\n1. Listing data files:")
    print(list_data_files.invoke({}))

    print("\n2. Getting dataset info:")
    print(get_dataframe_info.invoke({"filename": "Customer-Churn-Records.csv"}))

    print("\n3. Executing Python code:")
    code = """
import pandas as pd
df = pd.read_csv(DATA_DIR / "Customer-Churn-Records.csv")
print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\\n{df['Exited'].value_counts()}")
"""
    print(execute_code.invoke({"code": code}))

    # Demo 2: Guardrails testing
    print("\n--- Testing Guardrails ---")
    pipeline = GuardrailPipeline()

    test_output = """
Based on the analysis, the best model is RandomForest with accuracy of 0.87.
The top features are Age, Balance, and NumOfProducts.
Customers with high balance and low activity are at high risk of churn.
"""

    result = pipeline.run_full_pipeline(output=test_output)
    print(pipeline.get_summary(result))

    # Demo 3: Run simple profiling
    print("\n--- Running Profiler Agent ---")
    from langchain_core.messages import HumanMessage
    agents = build_simple_pipeline(llm)

    result = agents["profiler"].invoke(
        "List available datasets and show basic info about the churn dataset"
    )
    print(f"\nAgent Response:\n{result.get('response', 'No response')[:1000]}")

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Agentic AI Predictive Analytics Framework"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Groq API key (or set GROQ_API_KEY env var)"
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "pipeline", "demo"],
        default="pipeline",
        help="Execution mode"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task description for pipeline mode"
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: Please provide Groq API key via --api-key or GROQ_API_KEY env var")
        print("\nTo get an API key:")
        print("1. Go to https://console.groq.com")
        print("2. Sign up/login and create an API key")
        print("3. Run: export GROQ_API_KEY=your_key")
        sys.exit(1)

    set_api_key(api_key)

    # Create LLM
    print("Initializing Groq LLM...")
    llm = create_llm(api_key)

    # Run based on mode
    if args.mode == "interactive":
        run_interactive_mode(llm)
    elif args.mode == "demo":
        run_demo_mode(llm)
    else:  # pipeline
        results = run_pipeline_mode(llm, args.task)
        print("\n\nFinal Results Summary:")
        print("-" * 40)
        for phase, output in results.items():
            if phase != "guardrails":
                print(f"\n{phase.upper()}:")
                if isinstance(output, str):
                    print(output[:500] + "..." if len(output) > 500 else output)


if __name__ == "__main__":
    main()

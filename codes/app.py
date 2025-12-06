"""
Interactive Gradio UI for the Agentic AI Framework.
Compatible with Gradio 6.x
"""

import os
import sys
from pathlib import Path

# Add codes directory to path
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from config import set_api_key, MODEL_CONFIG, DATA_DIR, ARTIFACTS_DIR
from tools.code_executor import run_code, reset_persistent_env
from tools.prediction_tools import predict_single_customer, get_high_risk_customers, analyze_churn_factors
from graph import run_modeling_directly

# Global variables
chat_agent = None
llm = None


def initialize_agent(api_key: str) -> str:
    """Initialize the LLM and chat agent."""
    global chat_agent, llm

    if not api_key or len(api_key) < 10:
        return "Please enter a valid Groq API key. Get one free at https://console.groq.com"

    try:
        set_api_key(api_key)
        os.environ["GROQ_API_KEY"] = api_key

        from langchain_groq import ChatGroq
        from chat_agent import ChatAgent

        llm = ChatGroq(
            api_key=api_key,
            model=MODEL_CONFIG["model_name"],
            temperature=MODEL_CONFIG["temperature"],
            max_tokens=MODEL_CONFIG["max_tokens"],
        )

        chat_agent = ChatAgent(llm)
        reset_persistent_env()

        return "✅ Agent initialized successfully! You can now chat with the AI."

    except Exception as e:
        return f"❌ Error: {str(e)}"


def chat_fn(message: str, history: list) -> tuple:
    """Chat function for Gradio."""
    global chat_agent

    if not message.strip():
        return history, ""

    if chat_agent is None:
        history.append([message, "Please initialize the agent first by entering your API key."])
        return history, ""

    try:
        response = chat_agent.chat(message)
        history.append([message, response])
    except Exception as e:
        history.append([message, f"Error: {str(e)}"])

    return history, ""


def clear_chat():
    """Clear chat history."""
    global chat_agent
    if chat_agent:
        chat_agent.reset()
    return []


def get_dataset_overview():
    """Get dataset overview."""
    reset_persistent_env()
    code = '''
df = pd.read_csv(DATA_DIR / "Customer-Churn-Records.csv")
print("=" * 50)
print("CUSTOMER CHURN DATASET OVERVIEW")
print("=" * 50)
print(f"Total Records: {len(df):,}")
print(f"Total Features: {len(df.columns)}")
print(f"Churn Rate: {df['Exited'].mean()*100:.2f}%")
print(f"  - Churned: {df['Exited'].sum():,}")
print(f"  - Retained: {(df['Exited']==0).sum():,}")
print(f"\\nGeography Distribution:")
for geo, count in df['Geography'].value_counts().items():
    print(f"  - {geo}: {count:,} ({count/len(df)*100:.1f}%)")
print(f"\\nKey Statistics:")
print(f"  - Avg Age: {df['Age'].mean():.1f} years")
print(f"  - Avg Balance: ${df['Balance'].mean():,.0f}")
print(f"  - Avg Credit Score: {df['CreditScore'].mean():.0f}")
'''
    return run_code(code)


def train_models():
    """Train ML models."""
    reset_persistent_env()
    return run_modeling_directly()


def get_high_risk():
    """Get high risk customers."""
    try:
        return get_high_risk_customers.invoke({"top_n": 10})
    except Exception as e:
        return f"Error: {str(e)}. Train the model first!"


def get_factors():
    """Get churn factors."""
    try:
        return analyze_churn_factors.invoke({})
    except Exception as e:
        return f"Error: {str(e)}. Train the model first!"


def make_prediction(credit_score, geography, gender, age, tenure, balance,
                   num_products, has_card, is_active, salary, complain,
                   satisfaction, card_type, points):
    """Make a prediction."""
    try:
        return predict_single_customer.invoke({
            "credit_score": int(credit_score),
            "geography": geography,
            "gender": gender,
            "age": int(age),
            "tenure": int(tenure),
            "balance": float(balance),
            "num_of_products": int(num_products),
            "has_cr_card": int(has_card),
            "is_active_member": int(is_active),
            "estimated_salary": float(salary),
            "complain": int(complain),
            "satisfaction_score": int(satisfaction),
            "card_type": card_type,
            "point_earned": int(points)
        })
    except Exception as e:
        return f"Error: {str(e)}. Train the model first!"


def run_custom_code(code: str):
    """Run custom Python code."""
    if not code.strip():
        return "Enter some code to run."
    safe_code = f'''
import pandas as pd
import numpy as np
df = pd.read_csv(DATA_DIR / "Customer-Churn-Records.csv")
{code}
'''
    return run_code(safe_code)


# Build the UI
with gr.Blocks(title="Agentic AI - Churn Analysis") as demo:
    gr.Markdown("# 🤖 Agentic AI - Customer Churn Prediction")
    gr.Markdown("An intelligent system for analyzing customer churn and making predictions.")

    # API Key
    with gr.Row():
        api_key = gr.Textbox(label="Groq API Key", type="password", scale=3)
        init_btn = gr.Button("Initialize", variant="primary", scale=1)
    status = gr.Textbox(label="Status", interactive=False)
    init_btn.click(initialize_agent, inputs=[api_key], outputs=[status])

    with gr.Tab("💬 Chat"):
        chatbot = gr.Chatbot(label="Chat with AI", height=400)
        msg = gr.Textbox(label="Message", placeholder="Ask about the data...")
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear")

        send.click(chat_fn, [msg, chatbot], [chatbot, msg])
        msg.submit(chat_fn, [msg, chatbot], [chatbot, msg])
        clear.click(clear_chat, outputs=[chatbot])

    with gr.Tab("📊 Data Overview"):
        overview_btn = gr.Button("Load Overview", variant="primary")
        overview_out = gr.Textbox(label="Overview", lines=20)
        overview_btn.click(get_dataset_overview, outputs=[overview_out])

    with gr.Tab("🎯 Train Model"):
        train_btn = gr.Button("Train Models", variant="primary")
        train_out = gr.Textbox(label="Results", lines=20)
        train_btn.click(train_models, outputs=[train_out])

    with gr.Tab("🔮 Predict"):
        with gr.Row():
            with gr.Column():
                p_credit = gr.Slider(300, 850, value=650, label="Credit Score")
                p_geo = gr.Dropdown(["France", "Spain", "Germany"], value="France", label="Geography")
                p_gender = gr.Dropdown(["Male", "Female"], value="Male", label="Gender")
                p_age = gr.Slider(18, 100, value=35, label="Age")
                p_tenure = gr.Slider(0, 10, value=5, label="Tenure")
                p_balance = gr.Number(value=50000, label="Balance")
                p_products = gr.Slider(1, 4, value=1, label="Products")
            with gr.Column():
                p_card = gr.Radio([0, 1], value=1, label="Has Card")
                p_active = gr.Radio([0, 1], value=1, label="Active")
                p_salary = gr.Number(value=75000, label="Salary")
                p_complain = gr.Radio([0, 1], value=0, label="Complained")
                p_satisfaction = gr.Slider(1, 5, value=3, label="Satisfaction")
                p_cardtype = gr.Dropdown(["DIAMOND", "GOLD", "SILVER", "PLATINUM"], value="GOLD", label="Card Type")
                p_points = gr.Number(value=500, label="Points")

        predict_btn = gr.Button("Predict Churn", variant="primary")
        predict_out = gr.Textbox(label="Prediction", lines=8)
        predict_btn.click(make_prediction,
                         inputs=[p_credit, p_geo, p_gender, p_age, p_tenure, p_balance,
                                p_products, p_card, p_active, p_salary, p_complain,
                                p_satisfaction, p_cardtype, p_points],
                         outputs=[predict_out])

    with gr.Tab("⚠️ Risk Analysis"):
        with gr.Row():
            with gr.Column():
                risk_btn = gr.Button("High Risk Customers")
                risk_out = gr.Textbox(label="High Risk", lines=15)
                risk_btn.click(get_high_risk, outputs=[risk_out])
            with gr.Column():
                factor_btn = gr.Button("Churn Factors")
                factor_out = gr.Textbox(label="Factors", lines=15)
                factor_btn.click(get_factors, outputs=[factor_out])

    with gr.Tab("🔧 Custom Code"):
        code_in = gr.Textbox(label="Python Code", lines=8,
                            value='print(df["Exited"].value_counts())')
        code_btn = gr.Button("Run", variant="primary")
        code_out = gr.Textbox(label="Output", lines=10)
        code_btn.click(run_custom_code, inputs=[code_in], outputs=[code_out])


if __name__ == "__main__":
    print("Starting app...")
    print(f"Data: {DATA_DIR}")
    demo.launch(server_name="127.0.0.1", server_port=7860)

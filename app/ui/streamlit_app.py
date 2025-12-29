import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Local MCP Agents", layout="centered")
st.title("Local MCP Agents")

AGENTS = ["fastfinance_ads", "school_of_hard_knocks", "default"]

agent_name = st.selectbox("Agent", AGENTS, index=0)

if "run_id" not in st.session_state:
    st.session_state.run_id = None

def build_message_from_params(agent: str, params: dict) -> str:
    """
    Keep backend compatibility by encoding params into a structured message.
    The agent prompts already handle key:value style inputs well.
    """
    lines = []
    for k, v in params.items():
        if v is None:
            continue
        v_str = str(v).strip()
        if v_str == "":
            continue
        lines.append(f"{k}: {v_str}")
    return "\n".join(lines).strip()

def render_output(output):
    """
    Prefer human-friendly text, but keep raw JSON available.
    """
    if isinstance(output, str):
        text = output.strip()
        if not text:
            st.info("Empty response.")
            return
        st.markdown(text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\n\n"))
        return

    if isinstance(output, dict):
        for key in ["text", "ad_copy", "advice", "output", "answer", "message"]:
            val = output.get(key)
            if isinstance(val, str) and val.strip():
                st.markdown(val.strip().replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\n\n"))
                with st.expander("Raw JSON"):
                    st.json(output)
                return

        st.json(output)
        return

    if isinstance(output, list):
        st.json(output)
        return

    st.write(output)

params = {}

if agent_name == "fastfinance_ads":
    st.subheader("Parameters")
    params["brand"] = st.selectbox("Brand", ["FastFinance", "SME_Business"], index=0)
    params["channel"] = st.selectbox("Channel", ["META", "SEARCH", "LINKEDIN"], index=0)
    params["country"] = st.text_input("Country (e.g. NG, KE, ZA)", value="NG")
    params["product_id"] = st.text_input("Product ID (optional)", value="")
    params["prompt"] = st.text_area(
        "Campaign prompt",
        height=140,
        placeholder="Describe the campaign goal, offer, audience, constraints, etc.",
    )

    msg = build_message_from_params(agent_name, params)

elif agent_name == "school_of_hard_knocks":
    st.subheader("Parameters")
    params["prompt"] = st.text_area(
        "What are you struggling with?",
        height=180,
        placeholder="e.g. I’m stuck, distracted, and unsure what to focus on.",
    )
    msg = build_message_from_params(agent_name, params)

else:
    msg = st.text_input("Message", value="Summarize tools and then call ping.")

with st.expander("Request payload preview"):
    preview_payload = {"message": msg, "run_id": st.session_state.run_id, "agent_name": agent_name}
    st.json(preview_payload)

col1, col2 = st.columns(2)

if col1.button("Send"):
    payload = {"message": msg, "run_id": st.session_state.run_id, "agent_name": agent_name}
    r = requests.post(f"{API_URL}/chat", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    st.session_state.run_id = data.get("run_id")

    st.subheader("Answer")
    render_output(data.get("output"))

    if data.get("tool_calls"):
        st.subheader("Tool calls")
        st.json(data["tool_calls"])

    with st.expander("Full response JSON"):
        st.json(data)

if col2.button("Show tools"):
    r = requests.get(f"{API_URL}/mcp/tools", timeout=30)
    r.raise_for_status()
    st.subheader("Tools")
    st.json(r.json())

if st.button("Recent events"):
    params_q = {}
    if st.session_state.run_id:
        params_q["run_id"] = st.session_state.run_id
    r = requests.get(f"{API_URL}/events", params=params_q, timeout=30)
    r.raise_for_status()
    st.subheader("Events")
    st.json(r.json())

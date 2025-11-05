from __future__ import annotations

from typing import Dict
import os
import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Load local .env if present so OPENAI_API_KEY in project root is picked up
load_dotenv(find_dotenv())

# Configuration and defaults

# Pricing per 1K tokens for various models.
PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}

DEFAULT_MODEL: str = "gpt-4o-mini"
DEFAULT_TEMPERATURE: float = 0.2
DEFAULT_STREAM: bool = True
DEFAULT_SYSTEM_PROMPT: str = (
    "You are a Senior PowerBI Expert. Reply concisely with: Goal, Plan, "
    "Be pragmatic and minimal."
)

# Helper functions
def ensure_state() -> None:
    """Initialize Streamlit session state with sensible defaults.

    This function ensures that required keys exist in `st.session_state`. If they
    do not, they are set to default values. The `messages` key always begins
    with the current system prompt.
    """
    if "model" not in st.session_state:
        st.session_state.model = DEFAULT_MODEL
    if "temperature" not in st.session_state:
        st.session_state.temperature = DEFAULT_TEMPERATURE
    if "stream" not in st.session_state:
        st.session_state.stream = DEFAULT_STREAM
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": st.session_state.system_prompt}
        ]
    if "usage" not in st.session_state:
        st.session_state.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
        }


def reset_chat() -> None:
    """Reset the conversation while retaining the current system prompt."""
    st.session_state.messages = [
        {"role": "system", "content": st.session_state.system_prompt}
    ]


def approx_tokens(text: str) -> int:
    """Approximate the number of tokens in a given string.

    This heuristic divides the length of the string by four. It is used when
    token counts are not returned by the API (e.g., in streaming mode).
    """
    return max(1, len(text) // 4)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate the estimated cost of a prompt/response pair.

    Parameters
    ----------
    model: str
        The model name (must exist in PRICING or will fallback to the default).
    prompt_tokens: int
        Number of tokens in the prompt.
    completion_tokens: int
        Number of tokens in the model's completion.

    Returns
    -------
    float
        Estimated USD cost for the tokens consumed.
    """
    price = PRICING.get(model, PRICING[DEFAULT_MODEL])
    return (prompt_tokens / 1000.0) * price["input"] + (
        completion_tokens / 1000.0
    ) * price["output"]


def create_client(api_key: str) -> OpenAI:
    """Instantiate the OpenAI client using the official endpoint."""
    return OpenAI(api_key=api_key)


def call_completion(
    client: OpenAI, messages, *, model: str, temperature: float, stream: bool
):
    """Wrapper around the OpenAI chat completion API.

    This function centralizes the call so that streaming and non‑streaming
    invocations remain consistent and concise.
    """
    return client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, stream=stream
    )


# Streamlit application

# Set up the page
st.set_page_config(
    page_title="Chatbot Assistant",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://github.com/uzuntasgokberk',
        'Report a bug': "https://github.com/uzuntasgokberk",
        'About': "This is a Chatbot Assistant app."
    }
)
st.title("🔍 Chatbot Assistant")
# st.caption("Engineering Expert assistant.")

# Initialize session state
ensure_state()

# Sidebar for configuration
with st.sidebar:
    st.subheader("Settings")

    # Obtain API key from the environment, with optional override. We do not
    # persist overrides on disk.
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    api_key_override = ""

    # Inform the user whether an env key is present
    if env_key:
        st.success("Using API key from environment.")
    else:
        st.warning("No API key found in environment. Provide one below.")

    api_key_override = st.text_input(
        "OpenAI API key",
        type="password",
        placeholder="sk-...",
        help="Not persisted; overrides the environment key for this session.",
        autocomplete="off",
    )

    # Model settings
    model = st.selectbox(
        "Model",
        options=list(PRICING.keys()),
        index=list(PRICING.keys()).index(st.session_state.model)
        if st.session_state.model in PRICING
        else 0,
    )
    temperature = st.slider(
        "Temperature", 0.0, 1.0, st.session_state.temperature, 0.05,
        help= "Controls the randomness of the output. Lower is more deterministic."
    )
    stream = st.checkbox(
        "Stream responses", value=st.session_state.stream,
        help="If enabled, partial responses will be displayed as they arrive."
    )
    system_prompt = st.text_area(
        "System prompt",
        value=st.session_state.system_prompt,
        height=140,
    )

    # Buttons to apply changes and reset chat
    col_apply, col_reset = st.columns(2)
    if col_apply.button("Apply"):
        st.session_state.model = model
        st.session_state.temperature = temperature
        st.session_state.stream = stream
        # Use default if the prompt is empty after stripping
        st.session_state.system_prompt = (
            system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
        )
        # Update the system message in place if present
        if st.session_state.messages and st.session_state.messages[0]["role"] == "system":
            st.session_state.messages[0]["content"] = st.session_state.system_prompt
        else:
            reset_chat()
    if col_reset.button("Reset chat"):
        reset_chat()

    # Display cumulative usage as rows for clarity
    st.markdown("---")
    # create the placeholder here so the dataframe appears below the settings
    usage_placeholder = st.empty()
    usage = st.session_state.usage
    df_usage = pd.DataFrame(
        {
            "value": [
                usage["prompt_tokens"],
                usage["completion_tokens"],
                usage["total_tokens"],
                round(usage["cost"], 6),
            ]
        },
        index=["prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"],
    ).rename_axis("metric")
    # render initial usage table in placeholder (will be updated after each prompt)
    usage_placeholder.dataframe(df_usage)


# Render chat history without system messages
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    st.chat_message(message["role"]).write(message["content"])


# Chat input and inference
prompt = st.chat_input("Type your message…")
if prompt:
    # Determine API key
    api_key = (api_key_override or env_key).strip()
    if not api_key:
        st.error(
            "Missing OPENAI_API_KEY. Set it in your environment or paste one in the sidebar."
        )
        st.stop()
    client = create_client(api_key=api_key)

    # Append user message and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    try:
        # Choose streaming or non‑streaming call
        if st.session_state.stream:
            stream_resp = call_completion(
                client=client,
                messages=st.session_state.messages,
                model=st.session_state.model,
                temperature=st.session_state.temperature,
                stream=True,
            )
            assistant_text = ""
            with st.chat_message("assistant"):
                placeholder = st.empty()
                for chunk in stream_resp:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    content_piece = getattr(delta, "content", None)
                    if content_piece:
                        assistant_text += content_piece
                        placeholder.markdown(assistant_text)
            # Add assistant message
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_text}
            )
            # Estimate tokens and cost heuristically
            in_tok = approx_tokens(prompt)
            out_tok = approx_tokens(assistant_text)
            cost = estimate_cost(st.session_state.model, in_tok, out_tok)
            # Update usage
            st.session_state.usage["prompt_tokens"] += in_tok
            st.session_state.usage["completion_tokens"] += out_tok
            st.session_state.usage["total_tokens"] += in_tok + out_tok
            st.session_state.usage["cost"] += cost
            st.caption(
                f"≈ tokens in/out/total: {in_tok}/{out_tok}/{in_tok + out_tok} "
                f"• cost: ${cost:.6f}"
            )
            # update sidebar usage table immediately
            try:
                df_usage_new = pd.DataFrame(
                    {
                        "value": [
                            st.session_state.usage["prompt_tokens"],
                            st.session_state.usage["completion_tokens"],
                            st.session_state.usage["total_tokens"],
                            round(st.session_state.usage["cost"], 6),
                        ]
                    },
                    index=["prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"],
                ).rename_axis("metric")
                usage_placeholder.dataframe(df_usage_new)
            except Exception:
                pass
        else:
            resp = call_completion(
                client=client,
                messages=st.session_state.messages,
                model=st.session_state.model,
                temperature=st.session_state.temperature,
                stream=False,
            )
            assistant_text = resp.choices[0].message.content or ""
            st.chat_message("assistant").write(assistant_text)
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_text}
            )
            # Use accurate usage if available
            usage_data = getattr(resp, "usage", None)
            if usage_data:
                pt = usage_data.prompt_tokens
                ct = usage_data.completion_tokens
                tt = usage_data.total_tokens
            else:
                pt = approx_tokens(prompt)
                ct = approx_tokens(assistant_text)
                tt = pt + ct
            cost = estimate_cost(st.session_state.model, pt, ct)
            st.session_state.usage["prompt_tokens"] += pt
            st.session_state.usage["completion_tokens"] += ct
            st.session_state.usage["total_tokens"] += tt
            st.session_state.usage["cost"] += cost
            st.caption(
                f"tokens in/out/total: {pt}/{ct}/{tt} • cost: ${cost:.6f}"
            )
            # update sidebar usage table immediately
            try:
                df_usage_new = pd.DataFrame(
                    {
                        "value": [
                            st.session_state.usage["prompt_tokens"],
                            st.session_state.usage["completion_tokens"],
                            st.session_state.usage["total_tokens"],
                            round(st.session_state.usage["cost"], 6),
                        ]
                    },
                    index=["prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"],
                ).rename_axis("metric")
                usage_placeholder.dataframe(df_usage_new)
            except Exception:
                pass
    except Exception as err:
        st.error(f"OpenAI error: {err}")
"""
app.py

Streamlit conversational UI for the Agentic Data Analyst - Phase 1, Week 4.

Run with:
    streamlit run app.py

Architecture:
  • _init_session_state() - called once per browser session, initialises
    thread_id, chat_history, DuckDB connection, and compiled graph.
  • _reset_conversation() - atomically resets BOTH st.session_state.thread_id
    AND st.session_state.chat_history (Fix 1 - prevents UI/backend sync trap).
  • _render_sidebar() - schema viewer, session info, debug toggle, reset button.
  • _render_chat_history() - replays all prior turns from chat_history.
  • _run_agent() - invokes the LangGraph graph, extracts state, appends to
    chat_history.
  • main() - wires everything together in Streamlit's execution model.

session_state keys:
  thread_id    : str          - LangGraph checkpointer thread identifier
  chat_history : list[dict]   - UI representation of conversation turns
                                each dict: {role, content, sql, data_result,
                                            data_summary, forecast_result}
  db_conn      : DuckDBPyConnection
  db_schema    : str
  graph        : CompiledGraph
  debug_mode   : bool
"""

import logging
import os
import sys
import uuid

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found. Add it to your .env file.")
    st.stop()

from src.database import get_db_schema, init_db
from src.graph import build_graph
from src.visualiser import build_chart

#  Logging 
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


# Session state management

def _init_session_state() -> None:
    """
    Initialise all session_state keys exactly once per browser session.

    DuckDB connection and compiled graph are stored in session_state so they
    are created once and reused across every Streamlit rerun - not rebuilt
    on every user interaction.
    """
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False

    if "db_conn" not in st.session_state:
        try:
            st.session_state.db_conn   = init_db()
            st.session_state.db_schema = get_db_schema(st.session_state.db_conn)
        except FileNotFoundError as e:
            st.error(f"Database error: {e}")
            st.stop()

    if "graph" not in st.session_state:
        st.session_state.graph = build_graph(st.session_state.db_conn)


def _reset_conversation() -> None:
    """
    Atomically reset both the LangGraph thread and the Streamlit UI history.

    Fix 1: Both states must be cleared in the same function call.
    If only thread_id is regenerated, the UI shows old messages while the
    backend thinks it's a fresh conversation - they fall permanently out of sync.
    If only chat_history is cleared, the backend checkpointer still carries
    the old conversation under the old thread_id.
    """
    st.session_state.thread_id    = str(uuid.uuid4())
    st.session_state.chat_history = []
    logger.info("[app] Conversation reset. New thread_id: %s", st.session_state.thread_id[:8])


# Sidebar

def _render_sidebar() -> None:
    with st.sidebar:
        st.title("⚙️ Settings")

        # New conversation button 
        if st.button("🔄 New Conversation", use_container_width=True):
            _reset_conversation()
            st.rerun()

        st.divider()

        # Session info
        st.caption("Session")
        st.code(f"thread_id: {st.session_state.thread_id[:8]}...", language=None)

        st.divider()

        # Debug toggle
        st.session_state.debug_mode = st.toggle(
            "Show debug info",
            value=st.session_state.debug_mode,
        )

        st.divider()

        # Schema viewer 
        with st.expander("📋 Database Schema", expanded=False):
            st.code(st.session_state.db_schema, language=None)


# Chat rendering helpers

def _render_agent_turn(turn: dict) -> None:
    """
    Render one complete agent turn:
      1. Narrative response text
      2. SQL accordion
      3. Plotly chart (if data shape supports it)
      4. Data table (preview rows or full small result)
      5. Debug state snapshot (if debug_mode on)
    """
    with st.chat_message("assistant"):
        #  1. Narrative ─
        st.markdown(turn["content"])

        #  2. SQL accordion ─
        if turn.get("sql"):
            with st.expander("🔍 Show generated SQL"):
                st.code(turn["sql"], language="sql")

        #  3. Plotly chart 
        data_result     = turn.get("data_result", [])
        forecast_result = turn.get("forecast_result", {})

        fig = build_chart(
            data_result = data_result,
            query = turn.get("user_question", ""),
            forecast_result = forecast_result if forecast_result else None,
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        #  4. Data table 
        data_summary = turn.get("data_summary", "")

        if data_summary:
            with st.expander("📊 Statistical Summary", expanded=False):
                st.text(data_summary)
            if data_result:
                with st.expander(f"👁️ Preview ({len(data_result)} rows shown)", expanded=False):
                    st.dataframe(pd.DataFrame(data_result), use_container_width=True)

        elif data_result:
            st.dataframe(pd.DataFrame(data_result), use_container_width=True)

        #  5. Debug snapshot 
        if st.session_state.debug_mode and turn.get("debug"):
            with st.expander("🛠️ Debug State", expanded=False):
                debug = turn["debug"]
                col1, col2, col3 = st.columns(3)
                col1.metric("Retry Count", debug.get("retry_count", 0))
                col2.metric("Rows in Preview", debug.get("preview_rows", 0))
                col3.metric("Prediction Flag", str(debug.get("prediction_required", False)))

                if debug.get("sql_error"):
                    st.error(f"Last SQL error: {debug['sql_error']}")

                if forecast_result and not forecast_result.get("error"):
                    st.caption(
                        f"Model: {forecast_result.get('model_type')} | "
                        f"Training rows: {forecast_result.get('training_rows')}"
                    )


def _render_chat_history() -> None:
    """Replay all prior turns from st.session_state.chat_history."""
    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            with st.chat_message("user"):
                st.markdown(turn["content"])
        else:
            _render_agent_turn(turn)


# Agent invocation

def _run_agent(user_input: str) -> None:
    """
    Invoke the LangGraph graph, extract final state, and append both the
    user turn and agent turn to st.session_state.chat_history.
    """
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # Per-turn state - resets every turn, conversation memory via checkpointer
    turn_state = {
        "messages": [HumanMessage(content=user_input)],
        "db_schema": st.session_state.db_schema,
        "generated_sql": "",
        "sql_error": "",
        "retry_count": 0,
        "data_result": [],
        "data_summary": "",
        "prediction_required": False,
        "forecast_result": {},
    }

    with st.spinner("Agent thinking..."):
        try:
            final_state = st.session_state.graph.invoke(turn_state, config=config)
        except Exception as e:
            logger.error("Graph error: %s", e, exc_info=True)
            st.error(f"An unexpected error occurred: {e}")
            return

    #  Extract AI response 
    ai_messages = [
        m for m in final_state.get("messages", [])
        if hasattr(m, "type") and m.type == "ai"
    ]
    response_text = ai_messages[-1].content if ai_messages else "(no response)"

    #  Append user turn ─
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
    })

    #  Append agent turn 
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response_text,
        "user_question": user_input,
        "sql": final_state.get("generated_sql", ""),
        "data_result": final_state.get("data_result", []),
        "data_summary": final_state.get("data_summary", ""),
        "forecast_result": final_state.get("forecast_result", {}),
        "debug": {
            "retry_count": final_state.get("retry_count", 0),
            "sql_error": final_state.get("sql_error", ""),
            "prediction_required": final_state.get("prediction_required", False),
            "preview_rows": len(final_state.get("data_result", [])),
        },
    })


# Main

def main() -> None:
    st.set_page_config(
        page_title = "Agentic Data Analyst",
        page_icon = "📊",
        layout = "wide",
    )

    _init_session_state()
    _render_sidebar()

    #  Header ─
    st.title("📊 Agentic Data Analyst")
    st.caption(
        "Ask questions about your data in plain English. "
        "The agent generates SQL, self-heals on errors, and forecasts future trends."
    )
    st.divider()

    #  Welcome message on empty state 
    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            st.markdown(
                "Hello! I'm your Agentic Data Analyst. I have access to three tables:\n\n"
                "- **users** - 500 customers across 4 regions\n"
                "- **products** - 50 products across 5 categories\n"
                "- **orders** - 50,000 orders with revenue data\n\n"
                "Ask me anything - historical queries or future forecasts. "
                "You can also type `schema` to see the full database structure."
            )

    #  Render prior turns ─
    _render_chat_history()

    #  Chat input ─
    if user_input := st.chat_input("Ask a question about your data..."):

        # Special command: schema
        if user_input.strip().lower() == "schema":
            with st.chat_message("user"):
                st.markdown("schema")
            with st.chat_message("assistant"):
                st.code(st.session_state.db_schema, language=None)
            st.session_state.chat_history.append({"role": "user",      "content": "schema"})
            st.session_state.chat_history.append({"role": "assistant", 
                                                  "content": "*(schema displayed above)*",
                                                   "sql": "", "data_result": [], 
                                                   "data_summary": "",
                                                   "forecast_result": {}, 
                                                   "debug": {}
                                                })
            st.rerun()

        # Render user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run agent and render response
        _run_agent(user_input)

        # Render the latest agent turn
        if st.session_state.chat_history:
            last_turn = st.session_state.chat_history[-1]
            if last_turn["role"] == "assistant":
                _render_agent_turn(last_turn)


if __name__ == "__main__":
    main()